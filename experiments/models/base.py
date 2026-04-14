"""Shared model runtime types and config bases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

import msgspec
from typing_extensions import Self

from anomalog.representations import TemplatePhraseRepresentation
from experiments import ConfigError

if TYPE_CHECKING:
    from anomalog.sequences import TemplateSequence

TModelManifest = TypeVar("TModelManifest", bound="ModelManifest")


class PredictionOutcome(msgspec.Struct, frozen=True):
    """Detector-specific prediction fields for a single sequence."""

    predicted_label: int
    score: float
    key_phrases: list[str] = msgspec.field(default_factory=list)


class SequencePrediction(msgspec.Struct, frozen=True):
    """Serializable prediction record for a single sequence."""

    window_id: int
    split_label: str
    label: int
    predicted_label: int
    score: float
    entity_ids: list[str]
    event_count: int
    key_phrases: list[str]

    @classmethod
    def from_sequence(
        cls,
        sequence: TemplateSequence,
        *,
        outcome: PredictionOutcome,
    ) -> SequencePrediction:
        """Build a serialized prediction from a sequence and detector outcome.

        Returns:
            SequencePrediction: Serializable prediction record.
        """
        return cls(
            window_id=sequence.window_id,
            split_label=sequence.split_label.value,
            label=sequence.label,
            predicted_label=outcome.predicted_label,
            score=outcome.score,
            entity_ids=sequence.entity_ids,
            event_count=len(sequence.events),
            key_phrases=outcome.key_phrases,
        )


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

    def fit(self, train_sequences: list[TemplateSequence]) -> None:
        """Fit the detector from the training split."""

    def predict(self, sequence: TemplateSequence) -> PredictionOutcome:
        """Return detector-specific output for one sequence."""

    def model_manifest(self, *, sequence_summary: SequenceSummary) -> ModelManifest:
        """Return serializable detector metadata."""
