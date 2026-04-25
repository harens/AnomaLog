"""River-backed detector support for mapping-style sequence features."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated, ClassVar, Protocol

import msgspec
from river.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB

from experiments import ConfigError
from experiments.models.base import (
    ExperimentDetector,
    ModelManifest,
    PhraseModelConfig,
    PredictionOutcome,
    Probability,
    SequenceSummary,
    SingleFitMixin,
)

if TYPE_CHECKING:
    import logging
    from collections import Counter
    from collections.abc import Callable, Iterable

    from rich.progress import Progress

    from anomalog.representations import TemplatePhraseRepresentation
    from anomalog.sequences import TemplateSequence


class RiverEstimator(Protocol):
    """Minimal River classifier protocol used by the experiment layer."""

    def learn_one(self, x: Counter[str], y: int) -> RiverEstimator:
        """Update the estimator from one labeled example.

        Args:
            x (Counter[str]): Sparse phrase-count feature vector.
            y (int): Binary anomaly label for the example.

        Returns:
            RiverEstimator: Updated estimator instance.
        """

    def predict_proba_one(self, x: Counter[str]) -> dict[int, float]:
        """Return per-class probabilities for one example.

        Args:
            x (Counter[str]): Sparse phrase-count feature vector.

        Returns:
            dict[int, float]: Predicted probability per class label.
        """


class RiverEstimatorName(str, Enum):
    """Registered River estimator names used as config and runtime keys."""

    BERNOULLI_NB = "naive_bayes.BernoulliNB"
    COMPLEMENT_NB = "naive_bayes.ComplementNB"
    MULTINOMIAL_NB = "naive_bayes.MultinomialNB"


_RIVER_ESTIMATORS: dict[RiverEstimatorName, Callable[[float], RiverEstimator]] = {
    RiverEstimatorName.BERNOULLI_NB: lambda smoothing: BernoulliNB(alpha=smoothing),
    RiverEstimatorName.COMPLEMENT_NB: lambda smoothing: ComplementNB(alpha=smoothing),
    RiverEstimatorName.MULTINOMIAL_NB: lambda smoothing: MultinomialNB(alpha=smoothing),
}


class RiverModelConfig(
    PhraseModelConfig,
    tag="river",
    frozen=True,
):
    """River-backed classifier over sequence phrase counts.

    Attributes:
        estimator: Registered River estimator to train.
        anomalous_posterior_threshold: Posterior threshold for predicting the
            anomalous class.
    """

    estimator: Annotated[
        RiverEstimatorName,
        msgspec.Meta(description="Registered River estimator to train."),
    ] = RiverEstimatorName.MULTINOMIAL_NB
    anomalous_posterior_threshold: Annotated[
        Probability,
        msgspec.Meta(
            description=(
                "Posterior probability threshold for predicting the anomalous class."
            ),
        ),
    ] = 0.5

    def __post_init__(self) -> None:
        """Validate detector-specific model settings.

        Raises:
            ConfigError: If detector-specific settings are invalid.
        """
        super().__post_init__()
        if self.estimator not in _RIVER_ESTIMATORS:
            msg = f"Unsupported river estimator: {self.estimator!r}"
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
class RiverDetector(SingleFitMixin, ExperimentDetector):
    """Run a River classifier over sequence phrase-count features.

    Attributes:
        detector_name (ClassVar[str]): Stable detector name for manifests/logging.
        estimator_name (str): Registered River estimator to instantiate.
        smoothing (float): Smoothing parameter forwarded to the estimator.
        phrase_ngram_min (int): Minimum token n-gram size in the representation.
        phrase_ngram_max (int): Maximum token n-gram size in the representation.
        anomalous_posterior_threshold (float): Posterior threshold for anomaly
            predictions.
        representation (TemplatePhraseRepresentation): Phrase representation used
            for fitting and inference.
        model (RiverEstimator | None): Fitted River estimator once training
            completes.
        feature_count (int): Learned phrase vocabulary size seen during fitting.
    """

    detector_name: ClassVar[str] = "river"
    estimator_name: RiverEstimatorName
    smoothing: float
    phrase_ngram_min: int
    phrase_ngram_max: int
    anomalous_posterior_threshold: float
    representation: TemplatePhraseRepresentation
    model: RiverEstimator | None = None
    feature_count: int = 0

    def fit(
        self,
        train_sequences: Iterable[TemplateSequence],
        *,
        progress: Progress,
        logger: logging.Logger | None = None,
    ) -> None:
        """Train the River classifier over the training split.

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
        model = _RIVER_ESTIMATORS[self.estimator_name](self.smoothing)
        label_counts = {0: 0, 1: 0}
        feature_names: set[str] = set()
        for sequence in progress.track(
            train_sequences,
            description=f"Fitting {self.detector_name} sequences",
        ):
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
        self._mark_fit_complete()

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
        """Return serialisable River detector metadata.

        Args:
            sequence_summary (SequenceSummary): Aggregate split and label counts
                for the run.

        Returns:
            ModelManifest: Serialisable River manifest for the run.
        """
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


class RiverManifest(ModelManifest, frozen=True):
    """Serialisable River detector metadata.

    Attributes:
        backend (str): Backend family used for the detector.
        river_estimator (str): Concrete River estimator name.
        smoothing (float): Smoothing parameter forwarded during fitting.
        phrase_ngram_min (int): Minimum token n-gram size in the representation.
        phrase_ngram_max (int): Maximum token n-gram size in the representation.
        anomalous_posterior_threshold (float): Posterior threshold for anomaly
            predictions.
        train_template_phrase_vocabulary (int): Learned phrase vocabulary size.
    """

    backend: str
    river_estimator: str
    smoothing: float
    phrase_ngram_min: int
    phrase_ngram_max: int
    anomalous_posterior_threshold: float
    train_template_phrase_vocabulary: int
