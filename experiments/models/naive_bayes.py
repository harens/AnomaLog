"""Naive Bayes detector over extracted template phrases."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, ClassVar

import msgspec

from experiments.models.base import (
    ExperimentDetector,
    ModelManifest,
    PhraseModelConfig,
    PositiveInt,
    PredictionOutcome,
    Probability,
    SequenceSummary,
    SingleFitMixin,
)

if TYPE_CHECKING:
    import logging
    from collections.abc import Iterable

    from rich.progress import Progress

    from anomalog.representations import TemplatePhraseRepresentation
    from anomalog.sequences import TemplateSequence


@dataclass(frozen=True, slots=True)
class NaiveBayesPredictionOutcome(PredictionOutcome):
    """Naive Bayes prediction with phrase-level explanation fields.

    Attributes:
        key_phrases (list[str]): Most informative phrases supporting the predicted
            class for this sequence.
    """

    key_phrases: list[str]


class NaiveBayesModelConfig(
    PhraseModelConfig,
    tag="naive_bayes",
    frozen=True,
):
    """Multinomial Naive Bayes classifier over extracted template phrases.

    Attributes:
        top_k_phrases: Number of explanatory phrases to include per prediction.
        anomalous_posterior_threshold: Posterior threshold for predicting the
            anomalous class.
    """

    top_k_phrases: Annotated[
        PositiveInt,
        msgspec.Meta(
            description="Number of most informative phrases to report per prediction.",
        ),
    ] = 5
    anomalous_posterior_threshold: Annotated[
        Probability,
        msgspec.Meta(
            description=(
                "Posterior probability threshold for predicting the anomalous class."
            ),
        ),
    ] = 0.5

    def build_detector(self) -> NaiveBayesDetector:
        """Construct the configured Naive Bayes detector.

        Returns:
            NaiveBayesDetector: Configured detector instance.
        """
        return NaiveBayesDetector(
            smoothing=self.smoothing,
            phrase_ngram_min=self.phrase_ngram_min,
            phrase_ngram_max=self.phrase_ngram_max,
            top_k_phrases=self.top_k_phrases,
            anomalous_posterior_threshold=self.anomalous_posterior_threshold,
            representation=self.representation(),
        )


@dataclass(slots=True)
class NaiveBayesDetector(SingleFitMixin, ExperimentDetector):
    """Multinomial Naive Bayes classifier over extracted template phrases.

    Attributes:
        detector_name (ClassVar[str]): Stable detector name for manifests/logging.
        smoothing (float): Additive smoothing applied to phrase counts.
        phrase_ngram_min (int): Minimum token n-gram size in the representation.
        phrase_ngram_max (int): Maximum token n-gram size in the representation.
        top_k_phrases (int): Number of explanatory phrases to return.
        anomalous_posterior_threshold (float): Posterior threshold for predicting
            the anomalous class.
        representation (TemplatePhraseRepresentation): Phrase representation used
            for fitting and inference.
        class_priors (dict[int, float]): Learned class priors.
        phrase_counts_by_class (dict[int, Counter[str]]): Learned phrase counts
            per class.
        total_phrases_by_class (dict[int, int]): Total phrase counts per class.
        vocabulary (set[str]): Learned phrase vocabulary.
    """

    detector_name: ClassVar[str] = "naive_bayes"
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

    def fit(
        self,
        train_sequences: Iterable[TemplateSequence],
        *,
        progress: Progress,
        logger: logging.Logger | None = None,
    ) -> None:
        """Fit class priors and phrase likelihoods from train sequences.

        Args:
            train_sequences (Iterable[TemplateSequence]): Training split
                sequences.
            progress (Progress): Progress reporter.
            logger (logging.Logger | None): Optional logger for fit diagnostics.

        Raises:
            ValueError: If the training split does not contain both classes.
        """
        del logger
        self._ensure_unfit(detector_name=self.detector_name)
        class_counts: Counter[int] = Counter()
        phrase_counts_by_class = {0: Counter(), 1: Counter()}
        total_phrases_by_class = {0: 0, 1: 0}
        vocabulary: set[str] = set()
        for sequence in progress.track(
            train_sequences,
            description="Fitting naive_bayes sequences",
        ):
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
        self._mark_fit_complete()

    def predict(self, sequence: TemplateSequence) -> NaiveBayesPredictionOutcome:
        """Return anomalous posterior and most informative phrases.

        Args:
            sequence (TemplateSequence): Sequence to classify.

        Returns:
            NaiveBayesPredictionOutcome: Predicted label, score, and explanation
                phrases for the sequence.
        """
        sequence_phrases = Counter(self.representation.represent(sequence))
        anomalous_posterior = self.score(sequence_phrases)
        predicted_label = int(
            anomalous_posterior >= self.anomalous_posterior_threshold,
        )
        return NaiveBayesPredictionOutcome(
            predicted_label=predicted_label,
            score=anomalous_posterior,
            key_phrases=self._key_phrases_for_prediction(
                sequence_phrases,
                predicted_label=predicted_label,
            ),
        )

    def model_manifest(self, *, sequence_summary: SequenceSummary) -> ModelManifest:
        """Return serialisable model metadata and globally informative phrases.

        Args:
            sequence_summary (SequenceSummary): Aggregate split and label counts
                for the run.

        Returns:
            ModelManifest: Serialisable Naive Bayes manifest for the run.
        """
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
        """Return anomalous posterior probability for a phrase bag.

        Args:
            sequence_phrases (Counter[str]): Phrase counts extracted from one
                sequence.

        Returns:
            float: Posterior probability of the anomalous class.
        """
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
    """Prefer informative higher-order phrases in explanation outputs.

    Args:
        score (float): Phrase informativeness score.
        phrase (str): Phrase to rank among explanation candidates.

    Returns:
        tuple[float, float, int, str]: Sort key favoring informative phrases.
    """
    token_count = phrase.count(" ") + 1
    return (score * token_count, score, token_count, phrase)


class NaiveBayesManifest(ModelManifest, frozen=True):
    """Serialisable Naive Bayes metadata.

    Attributes:
        smoothing (float): Additive smoothing used during fitting.
        phrase_ngram_min (int): Minimum token n-gram size in the representation.
        phrase_ngram_max (int): Maximum token n-gram size in the representation.
        top_k_phrases (int): Number of explanatory phrases reported.
        anomalous_posterior_threshold (float): Posterior threshold for anomaly
            predictions.
        train_template_phrase_vocabulary (int): Learned phrase vocabulary size.
        class_priors (dict[int, float]): Learned class priors.
        key_phrases_by_class (dict[str, list[str]]): Globally informative phrases
            per class.
    """

    smoothing: float
    phrase_ngram_min: int
    phrase_ngram_max: int
    top_k_phrases: int
    anomalous_posterior_threshold: float
    train_template_phrase_vocabulary: int
    class_priors: dict[int, float]
    key_phrases_by_class: dict[str, list[str]]
