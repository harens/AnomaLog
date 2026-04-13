"""Naive Bayes detector over extracted template phrases."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from experiments import ConfigError
from experiments.models.base import (
    ExperimentDetector,
    ModelManifest,
    PhraseModelConfig,
    PredictionOutcome,
    SequenceSummary,
)

if TYPE_CHECKING:
    from anomalog.representations import TemplatePhraseRepresentation
    from anomalog.sequences import TemplateSequence


class NaiveBayesModelConfig(
    PhraseModelConfig,
    tag="naive_bayes",
    frozen=True,
):
    """Multinomial Naive Bayes classifier over extracted template phrases."""

    top_k_phrases: int = 5
    anomalous_posterior_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Validate detector-specific model settings."""
        self._validate_phrase_features()
        if self.top_k_phrases < 1:
            msg = "model.top_k_phrases must be at least 1."
            raise ConfigError(msg)
        if not 0.0 <= self.anomalous_posterior_threshold <= 1.0:
            msg = "model.anomalous_posterior_threshold must be between 0.0 and 1.0."
            raise ConfigError(msg)

    def build_detector(self) -> NaiveBayesDetector:
        """Construct the configured Naive Bayes detector."""
        return NaiveBayesDetector(
            smoothing=self.smoothing,
            phrase_ngram_min=self.phrase_ngram_min,
            phrase_ngram_max=self.phrase_ngram_max,
            top_k_phrases=self.top_k_phrases,
            anomalous_posterior_threshold=self.anomalous_posterior_threshold,
            representation=self.representation(),
        )


@dataclass(slots=True)
class NaiveBayesDetector(ExperimentDetector):
    """Multinomial Naive Bayes classifier over extracted template phrases."""

    detector_name = "naive_bayes"
    smoothing: float
    phrase_ngram_min: int
    phrase_ngram_max: int
    top_k_phrases: int
    anomalous_posterior_threshold: float
    representation: TemplatePhraseRepresentation
    class_priors: dict[int, float] = field(default_factory=dict)
    phrase_counts_by_class: dict[int, Counter[str]] = field(default_factory=dict)
    total_phrases_by_class: dict[int, int] = field(default_factory=dict)
    vocabulary: set[str] = field(default_factory=set)

    def fit(self, train_sequences: list[TemplateSequence]) -> None:
        """Fit class priors and phrase likelihoods from train sequences."""
        class_counts: Counter[int] = Counter()
        phrase_counts_by_class = {0: Counter(), 1: Counter()}
        total_phrases_by_class = {0: 0, 1: 0}
        vocabulary: set[str] = set()
        for sequence in train_sequences:
            class_counts[sequence.label] += 1
            sequence_phrases = Counter(self.representation.represent(sequence))
            phrase_counts_by_class[sequence.label].update(sequence_phrases)
            total_phrases_by_class[sequence.label] += sum(sequence_phrases.values())
            vocabulary.update(sequence_phrases)

        if class_counts[0] == 0 or class_counts[1] == 0:
            msg = (
                "Naive Bayes requires both normal and anomalous sequences in the "
                "training split."
            )
            raise ValueError(msg)

        self.class_priors = {
            0: class_counts[0] / sum(class_counts.values()),
            1: class_counts[1] / sum(class_counts.values()),
        }
        self.phrase_counts_by_class = phrase_counts_by_class
        self.total_phrases_by_class = total_phrases_by_class
        self.vocabulary = vocabulary

    def predict(self, sequence: TemplateSequence) -> PredictionOutcome:
        """Return anomalous posterior and most informative phrases."""
        sequence_phrases = Counter(self.representation.represent(sequence))
        anomalous_posterior = self.score(sequence_phrases)
        predicted_label = int(
            anomalous_posterior >= self.anomalous_posterior_threshold,
        )
        return PredictionOutcome(
            predicted_label=predicted_label,
            score=anomalous_posterior,
            key_phrases=self._key_phrases_for_prediction(
                sequence_phrases,
                predicted_label=predicted_label,
            ),
        )

    def model_manifest(self, *, sequence_summary: SequenceSummary) -> ModelManifest:
        """Return serializable model metadata and globally informative phrases."""
        return NaiveBayesManifest.from_sequence_summary(
            detector=self.detector_name,
            sequence_summary=sequence_summary,
            smoothing=self.smoothing,
            phrase_ngram_min=self.phrase_ngram_min,
            phrase_ngram_max=self.phrase_ngram_max,
            top_k_phrases=self.top_k_phrases,
            anomalous_posterior_threshold=self.anomalous_posterior_threshold,
            train_template_phrase_vocabulary=len(self.vocabulary),
            class_priors=self.class_priors,
            key_phrases_by_class={
                "normal": self._top_global_phrases(label=0),
                "anomalous": self._top_global_phrases(label=1),
            },
        )

    def score(self, sequence_phrases: Counter[str]) -> float:
        """Return anomalous posterior probability for a phrase bag."""
        log_prob_normal = self._log_joint_probability(
            label=0,
            phrases=sequence_phrases,
        )
        log_prob_anomalous = self._log_joint_probability(
            label=1,
            phrases=sequence_phrases,
        )
        max_log_prob = max(log_prob_normal, log_prob_anomalous)
        normal_weight = math.exp(log_prob_normal - max_log_prob)
        anomalous_weight = math.exp(log_prob_anomalous - max_log_prob)
        return anomalous_weight / (normal_weight + anomalous_weight)

    def _log_joint_probability(self, *, label: int, phrases: Counter[str]) -> float:
        prior = self.class_priors[label]
        log_probability = math.log(prior)
        for phrase, count in phrases.items():
            log_probability += count * self._log_phrase_probability(
                phrase=phrase,
                label=label,
            )
        return log_probability

    def _log_phrase_probability(self, *, phrase: str, label: int) -> float:
        vocabulary_size = max(len(self.vocabulary), 1)
        numerator = self.phrase_counts_by_class[label].get(phrase, 0) + self.smoothing
        denominator = self.total_phrases_by_class[label] + (
            self.smoothing * vocabulary_size
        )
        return math.log(numerator / denominator)

    def _key_phrases_for_prediction(
        self,
        sequence_phrases: Counter[str],
        *,
        predicted_label: int,
    ) -> list[str]:
        ranked_phrases: list[tuple[float, str]] = []
        for phrase, count in sequence_phrases.items():
            normal_log_prob = self._log_phrase_probability(phrase=phrase, label=0)
            anomalous_log_prob = self._log_phrase_probability(phrase=phrase, label=1)
            directional_score = count * (
                (anomalous_log_prob - normal_log_prob)
                if predicted_label == 1
                else (normal_log_prob - anomalous_log_prob)
            )
            ranked_phrases.append((directional_score, phrase))
        ranked_phrases.sort(
            key=lambda item: _phrase_rank(
                score=item[0],
                phrase=item[1],
            ),
            reverse=True,
        )
        positive_phrases = [phrase for score, phrase in ranked_phrases if score > 0]
        if positive_phrases:
            return positive_phrases[: self.top_k_phrases]
        return [phrase for _, phrase in ranked_phrases[: self.top_k_phrases]]

    def _top_global_phrases(self, *, label: int) -> list[str]:
        ranked_phrases: list[tuple[float, str]] = []
        other_label = 1 - label
        for phrase in self.vocabulary:
            label_log_prob = self._log_phrase_probability(phrase=phrase, label=label)
            other_log_prob = self._log_phrase_probability(
                phrase=phrase,
                label=other_label,
            )
            ranked_phrases.append((label_log_prob - other_log_prob, phrase))
        ranked_phrases.sort(
            key=lambda item: _phrase_rank(
                score=item[0],
                phrase=item[1],
            ),
            reverse=True,
        )
        return [phrase for _, phrase in ranked_phrases[: self.top_k_phrases]]


def _phrase_rank(*, score: float, phrase: str) -> tuple[float, float, int, str]:
    """Prefer informative higher-order phrases in explanation outputs."""
    token_count = phrase.count(" ") + 1
    return (score * token_count, score, token_count, phrase)


class NaiveBayesManifest(ModelManifest, frozen=True):
    """Serializable Naive Bayes metadata."""

    smoothing: float
    phrase_ngram_min: int
    phrase_ngram_max: int
    top_k_phrases: int
    anomalous_posterior_threshold: float
    train_template_phrase_vocabulary: int
    class_priors: dict[int, float]
    key_phrases_by_class: dict[str, list[str]]
