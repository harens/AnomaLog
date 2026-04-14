"""River-backed detector support for mapping-style sequence features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

from river.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB

from experiments import ConfigError
from experiments.models.base import (
    ExperimentDetector,
    ModelManifest,
    PhraseModelConfig,
    PredictionOutcome,
    SequenceSummary,
)

if TYPE_CHECKING:
    from collections import Counter
    from collections.abc import Callable

    from anomalog.representations import TemplatePhraseRepresentation
    from anomalog.sequences import TemplateSequence


class RiverEstimator(Protocol):
    """Minimal River classifier protocol used by the experiment layer."""

    def learn_one(self, x: Counter[str], y: int) -> RiverEstimator:
        """Update the estimator from one labeled example."""

    def predict_proba_one(self, x: Counter[str]) -> dict[int, float]:
        """Return per-class probabilities for one example."""


class RiverModelConfig(
    PhraseModelConfig,
    tag="river",
    frozen=True,
):
    """River-backed classifier over sequence phrase counts."""

    estimator: str = "naive_bayes.MultinomialNB"
    anomalous_posterior_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Validate detector-specific model settings.

        Raises:
            ConfigError: If detector-specific settings are invalid.
        """
        self._validate_phrase_features()
        if self.estimator not in _RIVER_ESTIMATORS:
            msg = f"Unsupported river estimator: {self.estimator!r}"
            raise ConfigError(msg)
        if not 0.0 <= self.anomalous_posterior_threshold <= 1.0:
            msg = "model.anomalous_posterior_threshold must be between 0.0 and 1.0."
            raise ConfigError(msg)

    def build_detector(self) -> RiverDetector:
        """Construct the configured River-backed detector.

        Returns:
            RiverDetector: Configured detector instance.
        """
        return RiverDetector(
            estimator_name=self.estimator,
            smoothing=self.smoothing,
            phrase_ngram_min=self.phrase_ngram_min,
            phrase_ngram_max=self.phrase_ngram_max,
            anomalous_posterior_threshold=self.anomalous_posterior_threshold,
            representation=self.representation(),
        )


@dataclass(slots=True)
class RiverDetector(ExperimentDetector):
    """Run a River classifier over sequence phrase-count features."""

    detector_name: ClassVar[str] = "river"
    estimator_name: str
    smoothing: float
    phrase_ngram_min: int
    phrase_ngram_max: int
    anomalous_posterior_threshold: float
    representation: TemplatePhraseRepresentation
    model: RiverEstimator | None = None
    feature_count: int = 0

    def fit(self, train_sequences: list[TemplateSequence]) -> None:
        """Train the River classifier over the training split.

        Args:
            train_sequences (list[TemplateSequence]): Training split sequences.

        Raises:
            ValueError: If the training split does not contain both classes.
        """
        model = _RIVER_ESTIMATORS[self.estimator_name](self.smoothing)
        label_counts = {0: 0, 1: 0}
        feature_names: set[str] = set()
        for sequence in train_sequences:
            features = self.representation.represent(sequence)
            model.learn_one(features, sequence.label)
            label_counts[sequence.label] = label_counts.get(sequence.label, 0) + 1
            feature_names.update(features)

        if label_counts.get(0, 0) == 0 or label_counts.get(1, 0) == 0:
            msg = (
                f"{self.detector_name} requires both normal and anomalous sequences "
                "in the training split."
            )
            raise ValueError(msg)

        self.model = model
        self.feature_count = len(feature_names)

    def predict(self, sequence: TemplateSequence) -> PredictionOutcome:
        """Return anomalous posterior probability from the River classifier.

        Args:
            sequence (TemplateSequence): Sequence to score.

        Returns:
            PredictionOutcome: Predicted label and anomalous posterior score.

        Raises:
            ValueError: If the detector has not been fit yet.
        """
        if self.model is None:
            msg = f"{self.detector_name} must be fit before prediction."
            raise ValueError(msg)
        features = self.representation.represent(sequence)
        probabilities = self.model.predict_proba_one(features)
        anomalous_posterior = float(probabilities.get(1, 0.0))
        predicted_label = int(
            anomalous_posterior >= self.anomalous_posterior_threshold,
        )
        return PredictionOutcome(
            predicted_label=predicted_label,
            score=anomalous_posterior,
        )

    def model_manifest(self, *, sequence_summary: SequenceSummary) -> ModelManifest:
        """Return serializable River detector metadata."""
        return RiverManifest.from_sequence_summary(
            detector=self.detector_name,
            sequence_summary=sequence_summary,
            backend="river",
            river_estimator=self.estimator_name,
            smoothing=self.smoothing,
            phrase_ngram_min=self.phrase_ngram_min,
            phrase_ngram_max=self.phrase_ngram_max,
            anomalous_posterior_threshold=self.anomalous_posterior_threshold,
            train_template_phrase_vocabulary=self.feature_count,
        )


_RIVER_ESTIMATORS: dict[str, Callable[[float], RiverEstimator]] = {
    "naive_bayes.BernoulliNB": lambda smoothing: BernoulliNB(alpha=smoothing),
    "naive_bayes.ComplementNB": lambda smoothing: ComplementNB(alpha=smoothing),
    "naive_bayes.MultinomialNB": lambda smoothing: MultinomialNB(alpha=smoothing),
}


class RiverManifest(ModelManifest, frozen=True):
    """Serializable River detector metadata."""

    backend: str
    river_estimator: str
    smoothing: float
    phrase_ngram_min: int
    phrase_ngram_max: int
    anomalous_posterior_threshold: float
    train_template_phrase_vocabulary: int
