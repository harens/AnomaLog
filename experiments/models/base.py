"""Shared model runtime types and config bases."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Protocol,
    TypeVar,
    runtime_checkable,
)

import msgspec
from typing_extensions import Self

from anomalog.representations import TemplatePhraseRepresentation
from experiments import ConfigError

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from rich.progress import Progress

    from anomalog.sequences import TemplateSequence

TModelManifest = TypeVar("TModelManifest", bound="ModelManifest")
TExperimentModelConfig = TypeVar(
    "TExperimentModelConfig",
    bound="ExperimentModelConfig",
)
_MODEL_CONFIG_DECODING: ContextVar[bool] = ContextVar(
    "_MODEL_CONFIG_DECODING",
    default=False,
)

NonNegativeFloat = Annotated[float, msgspec.Meta(ge=0.0)]
NonNegativeInt = Annotated[int, msgspec.Meta(ge=0)]
OpenProbability = Annotated[float, msgspec.Meta(gt=0.0, lt=1.0)]
Probability = Annotated[float, msgspec.Meta(ge=0.0, le=1.0)]
PositiveFloat = Annotated[float, msgspec.Meta(gt=0.0)]
PositiveInt = Annotated[int, msgspec.Meta(gt=0)]
PositiveSeconds = PositiveFloat


@dataclass(frozen=True, slots=True)
class PredictionOutcome:
    """In-memory detector output for one scored sequence.

    `PredictionOutcome` is the runtime-facing form: the detector returns this
    after it has made a decision. It keeps only the fields needed for
    evaluation and allows detector-specific subclasses to attach richer
    explanation payloads before anything is serialised.

    Attributes:
        predicted_label (int): Detector-predicted anomaly label.
        score (float): Detector-specific anomaly score.
    """

    predicted_label: int
    score: float

    def to_prediction_record(self, sequence: TemplateSequence) -> SequencePrediction:
        """Return a serialisable prediction record for one sequence.

        Args:
            sequence (TemplateSequence): Sequence being serialised.

        Returns:
            SequencePrediction: Shared serialised prediction record.
        """
        return SequencePrediction.from_sequence(
            sequence,
            outcome=self,
            detector_fields=self.detector_prediction_fields(),
        )

    def detector_prediction_fields(self) -> dict[str, Any]:
        """Return detector-specific fields to flatten into prediction output.

        Returns:
            dict[str, Any]: Dataclass fields defined by detector-specific
                outcome subclasses, excluding the shared label and score.
        """
        prediction_outcome_fields = {item.name for item in fields(PredictionOutcome)}
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name not in prediction_outcome_fields
        }


@dataclass(slots=True)
class SingleFitMixin:
    """Shared state for detectors that may only be fit once.

    The experiment layer treats fitting as a one-way transition. Re-running fit
    on the same detector instance would otherwise mix stale and fresh state, so
    detectors call `_ensure_unfit()` before any mutation and `_mark_fit_complete()`
    only after a successful fit.
    """

    _fit_completed: bool = field(default=False, init=False, repr=False)

    def _ensure_unfit(self, *, detector_name: str) -> None:
        """Reject repeated fitting on the same detector instance.

        Args:
            detector_name (str): Human-readable detector name used in the
                error message.

        Raises:
            RuntimeError: If fitting has already completed once.
        """
        if self._fit_completed:
            msg = f"{detector_name} can only be fit once."
            raise RuntimeError(msg)

    def _mark_fit_complete(self) -> None:
        """Record that the detector has finished a successful fit."""
        self._fit_completed = True


@dataclass(frozen=True, slots=True)
class _SharedPredictionFields:
    """Common serialised prediction fields.

    Attributes:
        window_id (int): Stable sequence window identifier.
        split_label (str): Train/test split label.
        label (int): Ground-truth sequence label.
        predicted_label (int): Detector-predicted label.
        score (float): Detector-specific anomaly score.
        entity_ids (list[str]): Entity ids present in the sequence.
        event_count (int): Number of events in the sequence.
    """

    window_id: int
    split_label: str
    label: int
    predicted_label: int
    score: float
    entity_ids: list[str]
    event_count: int

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Return shared prediction field names in serialised order.

        Returns:
            tuple[str, ...]: Ordered shared prediction field names.
        """
        return tuple(item.name for item in fields(cls))

    def to_shared_dict(self) -> dict[str, Any]:
        """Return shared prediction fields as a JSON-ready mapping.

        Returns:
            dict[str, Any]: Shared prediction fields ready for serialisation.
        """
        return {
            field_name: getattr(self, field_name)
            for field_name in _SharedPredictionFields.field_names()
        }


@dataclass(frozen=True, slots=True)
class SequencePrediction(_SharedPredictionFields):
    """Serializable prediction record for a single sequence.

    This is the persisted/output-facing form, not the detector's internal
    return type. The split from `PredictionOutcome` keeps runtime detector code
    focused on prediction logic while giving the experiment runner one stable
    JSON-friendly schema to write to disk.

    Attributes:
        detector_fields (dict[str, Any]): Detector-specific serialised fields to
            merge into the shared prediction payload.
    """

    detector_fields: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sequence(
        cls,
        sequence: TemplateSequence,
        *,
        outcome: PredictionOutcome,
        detector_fields: dict[str, Any] | None = None,
    ) -> SequencePrediction:
        """Build a prediction record from a sequence and detector output.

        Args:
            sequence (TemplateSequence): Sequence being serialised.
            outcome (PredictionOutcome): Detector output for the sequence.
            detector_fields (dict[str, Any] | None): Detector-specific fields
                to flatten into serialised output.

        Returns:
            SequencePrediction: Shared record plus detector-specific fields.
        """
        shared_fields = cls.shared_fields(sequence, outcome=outcome)
        return cls(
            **shared_fields.to_shared_dict(),
            detector_fields=detector_fields or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a flattened JSON-ready prediction record.

        Returns:
            dict[str, Any]: Shared prediction fields with detector-specific
                fields merged at the top level.
        """
        return {
            **self.to_shared_dict(),
            **self.detector_fields,
        }

    @staticmethod
    def shared_fields(
        sequence: TemplateSequence,
        *,
        outcome: PredictionOutcome,
    ) -> _SharedPredictionFields:
        """Build shared serialised prediction fields from detector output.

        Args:
            sequence (TemplateSequence): Sequence being serialised.
            outcome (PredictionOutcome): Detector output for the sequence.

        Returns:
            _SharedPredictionFields: Shared serialised prediction fields.
        """
        return _SharedPredictionFields(
            window_id=sequence.window_id,
            split_label=sequence.split_label.value,
            label=sequence.label,
            predicted_label=outcome.predicted_label,
            score=outcome.score,
            entity_ids=sequence.entity_ids,
            event_count=len(sequence.events),
        )


def require_entity_local_sequences(
    sequences: Iterable[TemplateSequence],
    *,
    detector_name: str,
) -> None:
    """Reject sequences that already span multiple entity ids.

    Args:
        sequences (Iterable[TemplateSequence]): Sequences to validate.
        detector_name (str): Detector name used in the validation error.

    Raises:
        ValueError: If any sequence contains multiple entity ids.
    """
    for sequence in sequences:
        if len(sequence.entity_ids) <= 1:
            continue
        msg = (
            f"{detector_name} requires entity-local sequences. "
            f"Found multiple entity_ids in one sequence: {sequence.entity_ids!r}."
        )
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class SequenceSummary:
    """Counts describing the generated sequence dataset.

    Attributes:
        sequence_count (int): Total number of generated sequences.
        train_sequence_count (int): Number of train-split sequences.
        test_sequence_count (int): Number of test-split sequences.
        train_label_counts (dict[int, int]): Train label histogram.
        test_label_counts (dict[int, int]): Test label histogram.
        ignored_label_counts (dict[int, int]): Label histogram for sequences
            withheld from the current train prefix.
        ignored_sequence_count (int): Number of sequences withheld from the
            current train prefix between the train pool and the fixed test
            suffix.
    """

    sequence_count: int
    train_sequence_count: int
    test_sequence_count: int
    train_label_counts: dict[int, int]
    test_label_counts: dict[int, int]
    ignored_label_counts: dict[int, int] = field(default_factory=dict)
    ignored_sequence_count: int = 0


class ModelManifest(msgspec.Struct, frozen=True, kw_only=True):
    """Serialisable manifest for one detector run.

    Attributes:
        detector (str): Stable detector name.
        train_sequence_count (int): Number of train-split sequences.
        test_sequence_count (int): Number of test-split sequences.
        train_label_counts (dict[int, int]): Train label histogram.
        test_label_counts (dict[int, int]): Test label histogram.
        ignored_sequence_count (int): Number of sequences withheld from the
            current training prefix.
    """

    detector: str
    train_sequence_count: int
    test_sequence_count: int
    train_label_counts: dict[int, int]
    test_label_counts: dict[int, int]
    ignored_sequence_count: int = 0

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
            ignored_sequence_count=sequence_summary.ignored_sequence_count,
            **detector_fields,
        )


@dataclass(frozen=True, slots=True)
class ModelRunSummary:
    """Detector outputs and run summaries.

    Attributes:
        metrics (dict[str, int | float | dict[int, int]]): Aggregate run
            metrics.
        model_manifest (ModelManifest): Detector manifest for the run.
        sequence_summary (SequenceSummary): Split and label counts for the run.
    """

    metrics: dict[str, int | float | dict[int, int]]
    model_manifest: ModelManifest
    sequence_summary: SequenceSummary


class ExperimentModelConfig(msgspec.Struct, frozen=True, tag_field="detector"):
    """Tagged experiment-model config base."""  # noqa: DOC601 DOC603: attribute docs live in Annotated metadata.

    name: Annotated[
        str,
        msgspec.Meta(description="Human-readable model config name."),
    ]
    description: Annotated[
        str | None,
        msgspec.Meta(description="Optional free-text model config description."),
    ] = None

    def __post_init__(self) -> None:
        """Reject direct construction so msgspec metadata remains authoritative.

        Raises:
            TypeError: If a model config is constructed directly.
        """
        if _MODEL_CONFIG_DECODING.get():
            return
        msg = (
            "Experiment model configs must be decoded with "
            "decode_experiment_model_config()."
        )
        raise TypeError(msg)

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
        """Construct the runtime detector for this config.

        Raises:
            NotImplementedError: Always, until implemented by a concrete model config.
        """  # noqa: DOC201, DOC203 - No return doc since base method always raises.
        msg = f"{type(self).__name__} must implement build_detector()."
        raise NotImplementedError(msg)


class PhraseModelConfig(ExperimentModelConfig, frozen=True):
    """Shared phrase-feature config for bag-of-phrases detectors."""  # noqa: DOC601 DOC603: attribute docs live in Annotated metadata.

    smoothing: Annotated[
        PositiveFloat,
        msgspec.Meta(description="Additive smoothing value for phrase counts."),
    ] = 1.0
    phrase_ngram_min: Annotated[
        PositiveInt,
        msgspec.Meta(description="Smallest template phrase n-gram size to extract."),
    ] = 1
    phrase_ngram_max: Annotated[
        PositiveInt,
        msgspec.Meta(description="Largest template phrase n-gram size to extract."),
    ] = 2

    def __post_init__(self) -> None:
        """Validate shared phrase-feature settings.

        Raises:
            ConfigError: If phrase n-gram bounds are invalid.
        """
        super().__post_init__()
        if self.phrase_ngram_max < self.phrase_ngram_min:
            msg = "model.phrase_ngram_max must be >= phrase_ngram_min."
            raise ConfigError(msg)

    def representation(self) -> TemplatePhraseRepresentation:
        """Return the phrase representation required by the detector.

        Returns:
            TemplatePhraseRepresentation: Phrase representation configured from
                the shared phrase settings.
        """
        return TemplatePhraseRepresentation(
            phrase_ngram_min=self.phrase_ngram_min,
            phrase_ngram_max=self.phrase_ngram_max,
        )


def decode_experiment_model_config(
    raw_config: Mapping[str, Any],
    *,
    config_type: type[TExperimentModelConfig],
    dec_hook: Callable[[type, object], object] | None = None,
) -> TExperimentModelConfig:
    """Decode an experiment model config through msgspec validation.

    Args:
        raw_config (Mapping[str, Any]): Raw decoded config mapping.
        config_type (type[TExperimentModelConfig]): Concrete model config type.
        dec_hook (Callable[[type, object], object] | None): Optional msgspec
            decode hook.

    Returns:
        TExperimentModelConfig: Validated model config.
    """
    token = _MODEL_CONFIG_DECODING.set(True)
    try:
        return msgspec.convert(raw_config, type=config_type, dec_hook=dec_hook)
    finally:
        _MODEL_CONFIG_DECODING.reset(token)


@runtime_checkable
class ExperimentDetector(Protocol):
    """Common detector runtime interface.

    Attributes:
        detector_name (str): Stable detector name for manifests and logging.
    """

    detector_name: str

    def fit(
        self,
        train_sequences: Iterable[TemplateSequence],
        *,
        progress: Progress,
        logger: logging.Logger | None = None,
    ) -> None:
        """Fit the detector from the training split.

        Args:
            train_sequences (Iterable[TemplateSequence]): Training sequences to fit on.
            progress (Progress): Progress reporter supplied by the runner.
            logger (logging.Logger | None): Optional logger for fit diagnostics.
        """

    def predict(self, sequence: TemplateSequence) -> PredictionOutcome:
        """Return detector-specific output for one sequence.

        Args:
            sequence (TemplateSequence): Sequence to score.

        Returns:
            PredictionOutcome: Detector-specific runtime prediction outcome.
        """

    def model_manifest(self, *, sequence_summary: SequenceSummary) -> ModelManifest:
        """Return serialisable detector metadata.

        Args:
            sequence_summary (SequenceSummary): Aggregate split and label counts
                for the run.

        Returns:
            ModelManifest: Serialisable detector manifest for the run.
        """


@runtime_checkable
class BatchExperimentDetector(Protocol):
    """Optional detector interface for bulk sequence scoring.

    Detectors that can score more efficiently in bulk than through repeated
    per-sequence calls may implement this protocol. The experiment runner still
    preserves sequence order and emits one prediction record per test sequence.
    """

    def predict_all(
        self,
        sequences: Iterable[TemplateSequence],
    ) -> Iterator[tuple[TemplateSequence, PredictionOutcome]]:
        """Return predictions for sequences in the same order they were given.

        Args:
            sequences (Iterable[TemplateSequence]): Test sequences to score.

        Returns:
            Iterator[tuple[TemplateSequence, PredictionOutcome]]: Scored
            sequence/output pairs in input order.
        """
