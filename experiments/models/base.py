"""Shared model runtime types and config bases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypedDict, TypeVar, runtime_checkable

import msgspec
from typing_extensions import Self

from anomalog.representations import TemplatePhraseRepresentation
from experiments import ConfigError

if TYPE_CHECKING:
    import logging
    from collections.abc import Iterable

    from rich.progress import Progress

    from anomalog.sequences import TemplateSequence

TModelManifest = TypeVar("TModelManifest", bound="ModelManifest")


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    """Shared prediction fields for a single sequence."""

    predicted_label: int
    score: float

    def to_prediction_record(self, sequence: TemplateSequence) -> SequencePrediction:
        """Return a serializable prediction record for one sequence.

        Args:
            sequence (TemplateSequence): Sequence being serialized.

        Returns:
            SequencePrediction: Shared serialized prediction record.
        """
        return SequencePrediction(
            **SequencePrediction.shared_fields(sequence, outcome=self),
        )


class DeepLogTopPrediction(msgspec.Struct, frozen=True):
    """Top-`g` key prediction candidate from DeepLog's next-event model.

    This corresponds to the paper's log-key anomaly detection component, which
    treats an event as normal when the observed next log key is among the
    highest-probability predicted candidates.
    """

    template: str
    probability: float


class DeepLogKeyFinding(msgspec.Struct, frozen=True):
    """Event-level finding from DeepLog's stacked-LSTM key predictor.

    This captures the paper's next-log-key anomaly decision for one target
    event, including the observed template, whether it was out of vocabulary,
    how unknown history items were handled, and the ranked top-`g` candidates
    produced by the model.
    """

    event_index: int
    history_templates: list[str]
    unknown_history_templates: list[str]
    actual_template: str
    actual_probability: float | None
    is_anomalous: bool
    is_oov: bool
    top_predictions: list[DeepLogTopPrediction]


class DeepLogParameterFinding(msgspec.Struct, frozen=True):
    """Event-level finding from a per-template DeepLog parameter model.

    This corresponds to the paper's parameter-value anomaly detection model:
    one LSTM per log key, residual scored by MSE, and thresholded by a
    Gaussian fitted on validation residuals.
    """

    event_index: int
    template: str
    feature_names: list[str]
    observed_vector: list[float | None]
    predicted_vector: list[float | None]
    residual_mse: float
    gaussian_mean: float
    gaussian_stddev: float
    gaussian_lower_bound: float
    gaussian_upper_bound: float
    most_anomalous_feature: str | None
    is_anomalous: bool


class DeepLogEventFinding(msgspec.Struct, frozen=True):
    """Combined DeepLog event-level anomaly finding.

    DeepLog reports anomalies at the event level, then aggregates them into a
    sequence-level prediction for the surrounding AnomaLog experiment contract.
    """

    event_index: int
    template: str
    key_model_finding: DeepLogKeyFinding | None = None
    parameter_model_finding: DeepLogParameterFinding | None = None


class DeepLogSequenceDetails(msgspec.Struct, frozen=True):
    """DeepLog-specific sequence prediction payload.

    This preserves the scoped DeepLog event-level outputs while allowing the
    shared experiment layer to continue writing one serialized prediction
    record per `TemplateSequence`.
    """

    is_sequence_anomalous: bool
    triggered_by_key_model: bool
    triggered_by_parameter_model: bool
    findings: list[DeepLogEventFinding] = msgspec.field(default_factory=list)


class _SharedPredictionFields(TypedDict):
    """Common serialized prediction fields."""

    window_id: int
    split_label: str
    label: int
    predicted_label: int
    score: float
    entity_ids: list[str]
    event_count: int


class SequencePrediction(msgspec.Struct, frozen=True):
    """Serializable prediction record for a single sequence."""

    window_id: int
    split_label: str
    label: int
    predicted_label: int
    score: float
    entity_ids: list[str]
    event_count: int

    @staticmethod
    def shared_fields(
        sequence: TemplateSequence,
        *,
        outcome: PredictionOutcome,
    ) -> _SharedPredictionFields:
        """Build shared serialized prediction fields from detector output.

        Args:
            sequence (TemplateSequence): Sequence being serialized.
            outcome (PredictionOutcome): Detector output for the sequence.

        Returns:
            _SharedPredictionFields: Shared serialized prediction fields.
        """
        return {
            "window_id": sequence.window_id,
            "split_label": sequence.split_label.value,
            "label": sequence.label,
            "predicted_label": outcome.predicted_label,
            "score": outcome.score,
            "entity_ids": sequence.entity_ids,
            "event_count": len(sequence.events),
        }


@dataclass(frozen=True, slots=True)
class SequenceSummary:
    """Counts describing the generated sequence dataset."""

    sequence_count: int
    train_sequence_count: int
    test_sequence_count: int
    train_label_counts: dict[int, int]
    test_label_counts: dict[int, int]


class ModelManifest(msgspec.Struct, frozen=True):
    """Serializable manifest for one detector run."""

    detector: str
    train_sequence_count: int
    test_sequence_count: int
    train_label_counts: dict[int, int]
    test_label_counts: dict[int, int]

    @classmethod
    def from_sequence_summary(
        cls,
        *,
        detector: str,
        sequence_summary: SequenceSummary,
        **detector_fields: object,
    ) -> Self:
        """Build a manifest by combining shared run summary and model metadata.

        Args:
            detector (str): Detector name for the manifest.
            sequence_summary (SequenceSummary): Aggregate split and label counts.
            **detector_fields (object): Detector-specific manifest fields.

        Returns:
            Self: Manifest with shared run summary and detector-specific fields.
        """
        return cls(
            detector=detector,
            train_sequence_count=sequence_summary.train_sequence_count,
            test_sequence_count=sequence_summary.test_sequence_count,
            train_label_counts=sequence_summary.train_label_counts,
            test_label_counts=sequence_summary.test_label_counts,
            **detector_fields,
        )


@dataclass(frozen=True, slots=True)
class ModelRunSummary:
    """Detector outputs and run summaries."""

    metrics: dict[str, int | float]
    model_manifest: ModelManifest
    sequence_summary: SequenceSummary


class ExperimentModelConfig(msgspec.Struct, frozen=True, tag_field="detector"):
    """Tagged experiment-model config base."""

    name: str
    description: str | None = None

    @property
    def detector(self) -> str:
        """Return the detector name encoded in the tagged config type.

        Raises:
            ConfigError: If the config type does not define a string detector tag.
        """
        detector = self.__struct_config__.tag
        if not isinstance(detector, str):
            msg = f"{type(self).__name__} does not define a string detector tag."
            raise ConfigError(msg)
        return detector

    def build_detector(self) -> ExperimentDetector:
        """Construct the runtime detector for this config."""
        msg = f"{type(self).__name__} must implement build_detector()."
        raise NotImplementedError(msg)


class PhraseModelConfig(ExperimentModelConfig, frozen=True):
    """Shared phrase-feature config for bag-of-phrases detectors."""

    smoothing: float = 1.0
    phrase_ngram_min: int = 1
    phrase_ngram_max: int = 2

    def _validate_phrase_features(self) -> None:
        if self.smoothing <= 0:
            msg = "model.smoothing must be positive."
            raise ConfigError(msg)
        if self.phrase_ngram_min < 1:
            msg = "model.phrase_ngram_min must be at least 1."
            raise ConfigError(msg)
        if self.phrase_ngram_max < self.phrase_ngram_min:
            msg = "model.phrase_ngram_max must be >= phrase_ngram_min."
            raise ConfigError(msg)

    def representation(self) -> TemplatePhraseRepresentation:
        """Return the phrase representation required by the detector."""
        return TemplatePhraseRepresentation(
            phrase_ngram_min=self.phrase_ngram_min,
            phrase_ngram_max=self.phrase_ngram_max,
        )


@runtime_checkable
class ExperimentDetector(Protocol):
    """Common detector runtime interface."""

    detector_name: str

    def fit(
        self,
        train_sequences: Iterable[TemplateSequence],
        *,
        progress: Progress,
        logger: logging.Logger | None = None,
    ) -> None:
        """Fit the detector from the training split."""

    def predict(self, sequence: TemplateSequence) -> PredictionOutcome:
        """Return detector-specific output for one sequence."""

    def model_manifest(self, *, sequence_summary: SequenceSummary) -> ModelManifest:
        """Return serializable detector metadata."""
