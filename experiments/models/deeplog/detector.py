# ruff: noqa: D101
"""DeepLog detector orchestration and manifest reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import msgspec
import torch

from anomalog.parsers.structured.contracts import is_anomalous_label
from experiments.models.base import (
    ExperimentDetector,
    ExperimentModelConfig,
    OpenProbability,
    PositiveFloat,
    PositiveInt,
    PredictionOutcome,
    SequenceSummary,
    SingleFitMixin,
)
from experiments.models.deeplog.key import (
    KeyScoringContext,
    fit_key_model,
    score_key_sequence,
)
from experiments.models.deeplog.parameters import (
    fit_parameter_models,
    parameter_anomaly_score,
    parameter_covered_event_count,
    parameter_model_input_size,
    score_parameter_sequence,
)
from experiments.models.deeplog.shared import (
    DeepLogEventFinding,
    DeepLogKeyFinding,
    DeepLogManifest,
    DeepLogParameterFinding,
    KeyLSTM,
    ParameterModelManifestEntry,
    ParameterModelState,
    SkippedParameterModelEntry,
    build_normal_training_corpus,
    evaluation_event_index_mask,
)
from experiments.models.next_event_metrics import (
    NextEventPredictionDiagnostics,
    NextEventPredictionExclusionReason,
    NextEventPredictionSegmentBoundaryReason,
    NextEventPredictionSegmentDiagnostics,
    NextEventPredictionSegmentSourceType,
    NextEventPredictionSegmentSummary,
    NextEventPredictionState,
    VocabularyPolicy,
)
from experiments.models.torch_runtime import (
    TorchDeviceName,
    resolve_torch_device,
    set_torch_seed,
)

if TYPE_CHECKING:
    import logging
    from collections.abc import Iterable

    from rich.progress import Progress

    from anomalog.sequences import TemplateSequence


@dataclass(frozen=True, slots=True)
class DeepLogPredictionOutcome(PredictionOutcome):
    """DeepLog runtime prediction plus detector-specific explanation fields.

    This keeps DeepLog-specific explanation machinery isolated from the generic
    experiment contract. The base prediction serialiser then flattens these fields
    into persisted sequence records.

    Attributes:
        triggered_by_key_model (bool): Whether the key model flagged the
            sequence.
        triggered_by_parameter_model (bool): Whether the parameter model
            flagged the sequence.
        findings (list[DeepLogEventFinding]): Event-level DeepLog findings.
    """

    triggered_by_key_model: bool
    triggered_by_parameter_model: bool
    findings: list[DeepLogEventFinding]


class DeepLogRunMetrics(msgspec.Struct, frozen=True):
    """DeepLog-specific run metrics for a single evaluation.

    Attributes:
        next_event_prediction (NextEventPredictionDiagnostics | None): Latest
            key-model next-event diagnostics.
        event_level_detection (DeepLogEventLevelDetectionDiagnostics | None):
            Event-level anomaly metrics derived from labelled log entries.
        sequence_trigger_breakdown (DeepLogSequenceTriggerBreakdown | None):
            Sequence-level trigger source counts split by actual label. This
            remains meaningful for session-based runs, but is suppressed for
            continuous stream batches where sequence boundaries are internal.
    """

    next_event_prediction: NextEventPredictionDiagnostics | None
    event_level_detection: DeepLogEventLevelDetectionDiagnostics | None
    sequence_trigger_breakdown: DeepLogSequenceTriggerBreakdown | None


class DeepLogEventLevelDetectionDiagnostics(msgspec.Struct, frozen=True):
    """Event-level DeepLog metrics derived from labelled log entries.

    Attributes:
        task (str): Stable task label for downstream reporting.
        events_seen (int): Labelled events encountered in the scored split.
        events_eligible (int): Labelled events with a full history window.
        tp (int): True positives at the event level.
        tn (int): True negatives at the event level.
        fp (int): False positives at the event level.
        fn (int): False negatives at the event level.
        normal_event_count (int): Number of normal labelled events seen.
        anomalous_event_count (int): Number of anomalous labelled events seen.
        precision (float): Event-level precision.
        recall (float): Event-level recall.
        f1 (float): Event-level F1 score.
    """

    task: str
    events_seen: int
    events_eligible: int
    tp: int
    tn: int
    fp: int
    fn: int
    normal_event_count: int
    anomalous_event_count: int
    precision: float
    recall: float
    f1: float


class DeepLogSequenceTriggerBreakdown(msgspec.Struct, frozen=True):
    """Sequence-level trigger source counts split by ground-truth label.

    Attributes:
        total_sequences (int): Number of scored test sequences.
        normal_sequences (int): Number of scored normal test sequences.
        anomalous_sequences (int): Number of scored anomalous test sequences.
        key_only_normal_sequences (int): Normal sequences flagged only by the key model.
        key_only_anomalous_sequences (int): Anomalous sequences flagged only
            by the key model.
        parameter_only_normal_sequences (int): Normal sequences flagged only
            by the parameter model.
        parameter_only_anomalous_sequences (int): Anomalous sequences flagged
            only by the parameter model.
        both_normal_sequences (int): Normal sequences flagged by both models.
        both_anomalous_sequences (int): Anomalous sequences flagged by both models.
        neither_normal_sequences (int): Normal sequences flagged by neither model.
        neither_anomalous_sequences (int): Anomalous sequences flagged by neither model.
    """

    total_sequences: int
    normal_sequences: int
    anomalous_sequences: int
    key_only_normal_sequences: int
    key_only_anomalous_sequences: int
    parameter_only_normal_sequences: int
    parameter_only_anomalous_sequences: int
    both_normal_sequences: int
    both_anomalous_sequences: int
    neither_normal_sequences: int
    neither_anomalous_sequences: int


@dataclass(slots=True)
class DeepLogStreamContext:
    """Cached history carried between chronological stream boundaries.

    Attributes:
        key_templates (list[str]): Tail of observed template names carried
            across chronological batch boundaries.
        parameter_events_by_template (dict[str, list[tuple[list[str], int | None]]]):
            Per-template tails of observed parameter events carried across
            chronological batch boundaries.
    """

    key_templates: list[str] = field(default_factory=list)
    parameter_events_by_template: dict[
        str,
        list[tuple[list[str], int | None]],
    ] = field(default_factory=dict)


@dataclass(slots=True)
class _NextEventPredictionSegmentState:
    """Track segment-level next-event warm-up across streamed sequences.

    Attributes:
        history_size (int): DeepLog history length used for scoring.
        segment_id (int): Monotonic segment identifier within the run.
        active (bool): Whether a segment is currently open.
        warmup_remaining (int): Remaining warm-up events for the active
            segment.
        current_length (int): Number of scored events in the active segment.
        current_insufficient_history (int): Number of events excluded from the
            active segment because the carried history was insufficient.
        current_continuous_context (bool): Whether the active segment should
            continue across the next sequence boundary.
        current_boundary_reason (NextEventPredictionSegmentBoundaryReason):
            Why the active segment started.
        current_source_object_type (NextEventPredictionSegmentSourceType):
            Best-effort source-object label for the active segment.
        summaries (list[NextEventPredictionSegmentSummary]): Finalised segment
            summaries for the run.
    """

    history_size: int
    segment_id: int = 0
    active: bool = False
    warmup_remaining: int = 0
    current_length: int = 0
    current_insufficient_history: int = 0
    current_continuous_context: bool = False
    current_boundary_reason: NextEventPredictionSegmentBoundaryReason = (
        NextEventPredictionSegmentBoundaryReason.UNKNOWN
    )
    current_source_object_type: NextEventPredictionSegmentSourceType = (
        NextEventPredictionSegmentSourceType.UNKNOWN
    )
    summaries: list[NextEventPredictionSegmentSummary] = field(
        default_factory=list,
    )

    def start_segment(
        self,
        *,
        prefix_length: int,
        continuous_context: bool,
        boundary_reason: NextEventPredictionSegmentBoundaryReason,
        source_object_type: NextEventPredictionSegmentSourceType,
    ) -> None:
        """Begin a fresh prediction segment.

        Args:
            prefix_length (int): Number of carried templates already available
                before the active segment starts.
            continuous_context (bool): Whether the segment should continue
                across later sequence boundaries.
            boundary_reason (NextEventPredictionSegmentBoundaryReason): Why the
                segment starts.
            source_object_type (NextEventPredictionSegmentSourceType): Best-
                effort source-object label for the segment.
        """
        if self.active:
            self.finalise_segment()
        self.segment_id += 1
        self.active = True
        self.warmup_remaining = max(0, self.history_size - prefix_length)
        self.current_length = 0
        self.current_insufficient_history = 0
        self.current_continuous_context = continuous_context
        self.current_boundary_reason = boundary_reason
        self.current_source_object_type = source_object_type

    def record_event(self, *, has_history: bool) -> bool:
        """Advance the current segment by one scored event.

        Args:
            has_history (bool): Whether the event already has enough carried
                context to be scored normally.

        Returns:
            bool: True when the event contributes to the insufficient-history
            count.

        Raises:
            RuntimeError: If the segment has not been started.
        """
        if not self.active:
            msg = "prediction segment state must be started before recording events."
            raise RuntimeError(msg)
        self.current_length += 1
        if has_history:
            return False
        self.current_insufficient_history += 1
        if self.warmup_remaining > 0:
            self.warmup_remaining -= 1
        return True

    def finalise_segment(self) -> None:
        """Persist the active segment summary, if any."""
        if not self.active:
            return
        self.summaries.append(
            NextEventPredictionSegmentSummary(
                segment_id=self.segment_id,
                length=self.current_length,
                insufficient_history=self.current_insufficient_history,
                boundary_reason=self.current_boundary_reason,
                source_object_type=self.current_source_object_type,
            ),
        )
        self.active = False
        self.warmup_remaining = 0
        self.current_length = 0
        self.current_insufficient_history = 0
        self.current_continuous_context = False
        self.current_boundary_reason = NextEventPredictionSegmentBoundaryReason.UNKNOWN
        self.current_source_object_type = NextEventPredictionSegmentSourceType.UNKNOWN

    def snapshot(self) -> NextEventPredictionSegmentDiagnostics | None:
        """Return an aggregate view of the recorded segments.

        Returns:
            NextEventPredictionSegmentDiagnostics | None: Segment-level
            diagnostics for the run, or `None` when no segments were recorded.
        """
        if self.active:
            self.finalise_segment()
        if not self.summaries:
            return None
        sorted_by_length = sorted(
            self.summaries,
            key=lambda item: (item.length, item.segment_id),
        )
        largest_segments = sorted(
            self.summaries,
            key=lambda item: (item.length, item.segment_id),
            reverse=True,
        )[:5]
        histogram: dict[str, int] = {}
        for summary in self.summaries:
            key = str(summary.length)
            histogram[key] = histogram.get(key, 0) + 1
        return NextEventPredictionSegmentDiagnostics(
            segment_count=len(self.summaries),
            history_size=self.history_size,
            expected_insufficient_history_from_segments=sum(
                summary.insufficient_history for summary in self.summaries
            ),
            largest_segments=largest_segments,
            smallest_segments=sorted_by_length[:5],
            segment_length_histogram=histogram,
        )


class DeepLogModelConfig(
    ExperimentModelConfig,
    tag="deeplog",
    frozen=True,
):
    history_size: Annotated[
        PositiveInt,
        msgspec.Meta(
            description="Number of prior log keys used to predict the next key. "
            "In the paper, the history size `h` is set to 10.",
        ),
    ] = 10
    top_g: Annotated[
        PositiveInt,
        msgspec.Meta(
            description="Number of top next-key predictions accepted as normal. "
            "In the paper, `g` is set to 9, but evaluates values from 7 to 12 "
            "in Figure 8.",
        ),
    ] = 9
    num_layers: Annotated[
        PositiveInt,
        msgspec.Meta(
            description="Number of LSTM layers in the key model. "
            "In the paper, the number of layers `L` is set to 2, but evaluates "
            "values from 1 to 5 in Figure 8.",
        ),
    ] = 2
    hidden_size: Annotated[
        PositiveInt,
        msgspec.Meta(
            description="Hidden dimension for DeepLog LSTM models. "
            "The paper uses 64 memory units per LSTM layer, but evaluates "
            "values from 32, 64, 128, 192 and 256 in Figure 8.",
        ),
    ] = 64
    epochs: Annotated[
        PositiveInt,
        msgspec.Meta(
            description="Training epochs for DeepLog neural models. "
            "This is not defined in the paper.",
        ),
    ] = 300
    batch_size: Annotated[
        PositiveInt,
        msgspec.Meta(
            description="Training batch size for DeepLog neural models. "
            "This is not defined in the paper.",
        ),
    ] = 2048
    learning_rate: Annotated[
        PositiveFloat,
        msgspec.Meta(
            description="Optimiser learning rate for DeepLog neural models. "
            "This is not defined in the paper, but 1e-3 is a common default "
            "for Adam-based training.",
        ),
    ] = 1e-3
    validation_fraction: Annotated[
        OpenProbability,
        msgspec.Meta(
            description=(
                "Fraction of normal training data held out to fit Gaussian "
                "parameter thresholds. The paper requires a validation set for "
                "modeling MSE distributions but does not define a split fraction."
            ),
        ),
    ] = 0.1
    gaussian_confidence: Annotated[
        OpenProbability,
        msgspec.Meta(
            description="Gaussian confidence interval used for parameter scoring. "
            "This is not defined in the paper, but Figure 9 evaluates different "
            "levels. Default to 99%, the middle of the paper's evaluated CIs: "
            "98%, 99%, 99.9%.",
        ),
    ] = 0.99
    parameter_detection_enabled: Annotated[
        bool,
        msgspec.Meta(
            description=(
                "Whether to fit and apply the per-template parameter anomaly "
                "models. The HDFS DeepLog paper benchmark reports the key "
                "model only, while parameter-value detection is evaluated "
                "separately on OpenStack."
            ),
        ),
    ] = True
    include_elapsed_time: Annotated[
        bool,
        msgspec.Meta(
            description="Whether parameter models include elapsed-time features. "
            "The paper includes elapsed time as one of the modeled quantitative "
            "parameters, so this is set to True for paper-faithful modeling.",
        ),
    ] = True
    random_seed: Annotated[
        int,
        msgspec.Meta(description="Random seed used for deterministic torch training."),
    ] = 0
    vocabulary_policy: Annotated[
        VocabularyPolicy,
        msgspec.Meta(
            description=(
                "Vocabulary policy used for next-event diagnostics. "
                "DeepLog defaults to full-dataset diagnostics for direct "
                "comparison with DeepCASE, while train-only remains available "
                "when a closed-world scope is preferred."
            ),
        ),
    ] = VocabularyPolicy.FULL_DATASET
    device: Annotated[
        TorchDeviceName,
        msgspec.Meta(description="Torch device selection: auto, cpu, cuda, or mps."),
    ] = "auto"
    short_session_padding_fidelity: Annotated[
        bool,
        msgspec.Meta(
            description=(
                "Enable paper-script short-session fallback for key scoring. "
                "When enabled, test sessions shorter than history_size+1 still "
                "produce one top-g decision using left-padded history."
            ),
        ),
    ] = True

    def build_detector(self) -> DeepLogDetector:
        """Construct a DeepLog detector for experiment execution.

        Returns:
            DeepLogDetector: Configured detector instance.
        """
        return DeepLogDetector(config=self)


@dataclass(slots=True)
class DeepLogDetector(SingleFitMixin, ExperimentDetector):
    """Scoped DeepLog detector for AnomaLog experiment runs.

    The implementation mirrors the paper's two-stage inference logic:

    1. score the next-log-key model on each eligible event
    2. only if the key looks normal, score the parameter model for that event

    Attributes:
        detector_name (ClassVar[str]): Stable detector registry name.
        config (DeepLogModelConfig): Immutable detector configuration.
        key_model (KeyLSTM | None): Fitted next-key model.
        template_to_index (dict[str, int]): Template-to-index vocabulary map.
        index_to_template (dict[int, str]): Reverse key vocabulary map.
        parameter_models (dict[str, ParameterModelState]): Fitted per-template
            parameter models.
        skipped_parameter_models (dict[str, str]): Reasons template models were
            skipped during fitting.
        train_event_count (int): Number of training events seen.
        train_parameter_covered_event_count (int): Number of training events
            covered by parameter models.
        test_event_count (int): Number of test events seen.
        scored_parameter_event_count (int): Number of scored test events passed
            to parameter models.
        device (torch.device): Resolved runtime torch device.
    """

    detector_name: ClassVar[str] = "deeplog"
    config: DeepLogModelConfig
    key_model: KeyLSTM | None = None
    template_to_index: dict[str, int] = field(default_factory=dict)
    index_to_template: dict[int, str] = field(default_factory=dict)
    parameter_models: dict[str, ParameterModelState] = field(default_factory=dict)
    skipped_parameter_models: dict[str, str] = field(default_factory=dict)
    train_event_count: int = 0
    train_parameter_covered_event_count: int = 0
    test_event_count: int = 0
    scored_parameter_event_count: int = 0
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    _next_event_prediction_state: NextEventPredictionState | None = field(
        default=None,
        repr=False,
    )
    _event_level_events_seen: int = field(default=0, init=False, repr=False)
    _event_level_events_eligible: int = field(default=0, init=False, repr=False)
    _event_level_tp: int = field(default=0, init=False, repr=False)
    _event_level_tn: int = field(default=0, init=False, repr=False)
    _event_level_fp: int = field(default=0, init=False, repr=False)
    _event_level_fn: int = field(default=0, init=False, repr=False)
    _event_level_normal_count: int = field(default=0, init=False, repr=False)
    _event_level_anomalous_count: int = field(default=0, init=False, repr=False)
    _sequence_total_count: int = field(default=0, init=False, repr=False)
    _sequence_normal_count: int = field(default=0, init=False, repr=False)
    _sequence_anomalous_count: int = field(default=0, init=False, repr=False)
    _sequence_key_only_normal_count: int = field(default=0, init=False, repr=False)
    _sequence_key_only_anomalous_count: int = field(
        default=0,
        init=False,
        repr=False,
    )
    _sequence_parameter_only_normal_count: int = field(
        default=0,
        init=False,
        repr=False,
    )
    _sequence_parameter_only_anomalous_count: int = field(
        default=0,
        init=False,
        repr=False,
    )
    _sequence_both_normal_count: int = field(default=0, init=False, repr=False)
    _sequence_both_anomalous_count: int = field(default=0, init=False, repr=False)
    _sequence_neither_normal_count: int = field(default=0, init=False, repr=False)
    _sequence_neither_anomalous_count: int = field(
        default=0,
        init=False,
        repr=False,
    )
    _stream_context: DeepLogStreamContext | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _next_event_prediction_segment_state: _NextEventPredictionSegmentState | None = (
        field(
            default=None,
            init=False,
            repr=False,
        )
    )
    _sequence_trigger_breakdown_applicable: bool = field(
        default=True,
        init=False,
        repr=False,
    )

    def fit(
        self,
        train_sequences: Iterable[TemplateSequence],
        *,
        progress: Progress,
        logger: logging.Logger | None = None,
    ) -> None:
        """Fit the DeepLog key and parameter models from normal sequences.

        Args:
            train_sequences (Iterable[TemplateSequence]): Training split.
            progress (Progress): Progress reporter.
            logger (logging.Logger | None): Optional logger for fit diagnostics.
        """
        self._ensure_unfit(detector_name=self.detector_name)
        stream_context = DeepLogStreamContext()
        stream_context_enabled = False

        def observe_sequence(sequence: TemplateSequence) -> None:
            nonlocal stream_context_enabled
            if not sequence.continuous_context:
                return
            stream_context_enabled = True
            self._update_stream_context(
                sequence=sequence,
                stream_context=stream_context,
            )

        training_corpus = build_normal_training_corpus(
            train_sequences,
            progress=progress,
            observe_sequence=observe_sequence,
        )
        set_torch_seed(self.config.random_seed)
        device = resolve_torch_device(self.config.device)
        if logger is not None:
            logger.info("DeepLog resolved torch device: %s", device)
        key_model, template_to_index, index_to_template = fit_key_model(
            training_corpus=training_corpus,
            config=self.config,
            device=device,
            progress=progress,
        )
        if self.config.parameter_detection_enabled:
            parameter_models, skipped_parameter_models = fit_parameter_models(
                training_corpus=training_corpus,
                config=self.config,
                device=device,
                progress=progress,
            )
        else:
            parameter_models = {}
            skipped_parameter_models = {}
        train_event_count = training_corpus.event_count
        train_parameter_covered_event_count = (
            parameter_covered_event_count(
                sequences=training_corpus.sequences,
                parameter_models=parameter_models,
            )
            if self.config.parameter_detection_enabled
            else 0
        )
        self.device = device
        self.key_model = key_model
        self.template_to_index = template_to_index
        self.index_to_template = index_to_template
        self.parameter_models = parameter_models
        self.skipped_parameter_models = skipped_parameter_models
        self.train_event_count = train_event_count
        self.train_parameter_covered_event_count = train_parameter_covered_event_count
        self._stream_context = stream_context if stream_context_enabled else None
        self._sequence_trigger_breakdown_applicable = True
        self._reset_next_event_prediction_state()
        self._reset_event_level_state()
        self._mark_fit_complete()

    def predict(self, sequence: TemplateSequence) -> DeepLogPredictionOutcome:
        """Return DeepLog findings aggregated to one sequence-level prediction.

        Args:
            sequence (TemplateSequence): Sequence to score.

        Returns:
            DeepLogPredictionOutcome: Sequence-level DeepLog output.

        Raises:
            ValueError: If the detector has not been fit yet.
        """
        if self.key_model is None:
            msg = "deeplog must be fit before prediction."
            raise ValueError(msg)

        findings: list[DeepLogEventFinding] = []
        key_triggered = False
        parameter_triggered = False
        scores: list[float] = []
        evaluation_event_indexes = set(evaluation_event_index_mask(sequence))
        self.test_event_count += len(evaluation_event_indexes)
        self._prepare_next_event_prediction_segment(
            sequence=sequence,
            prefix_length=self._prediction_prefix_length(sequence),
        )

        # DeepLog makes one decision per event after the initial warm-up
        # window. First it asks whether the next log key itself looks normal.
        key_findings = score_key_sequence(
            sequence=sequence,
            context=KeyScoringContext(
                model=self.key_model,
                template_to_index=self.template_to_index,
                index_to_template=self.index_to_template,
                history_size=self.config.history_size,
                top_g=self.config.top_g,
            ),
            prefix_templates=self._prediction_prefix_templates(sequence),
            include_short_session_padding_fallback=(
                self.config.short_session_padding_fidelity
            ),
        )
        self._record_next_event_predictions(
            sequence=sequence,
            key_findings=key_findings,
            evaluation_event_indexes=evaluation_event_indexes,
            segment_state=self._next_event_prediction_segment_state,
        )
        # The paper's inference path is "key first, parameters second". We
        # therefore only pay the parameter-model cost for events whose key
        # history was accepted as normal by the key model.
        if self.config.parameter_detection_enabled:
            parameter_eligible_event_indexes = {
                event_index
                for event_index, key_finding in key_findings.items()
                if (
                    not key_finding.is_anomalous
                    and event_index in evaluation_event_indexes
                )
            }
            parameter_findings = score_parameter_sequence(
                sequence=sequence,
                parameter_models=self.parameter_models,
                history_size=self.config.history_size,
                eligible_event_indexes=parameter_eligible_event_indexes,
                prefix_events_by_template=(
                    None
                    if not sequence.continuous_context or self._stream_context is None
                    else self._stream_context.parameter_events_by_template
                ),
            )
        else:
            parameter_findings = {}

        event_indexes = sorted(
            (set(key_findings) | set(parameter_findings)) & evaluation_event_indexes,
        )
        for event_index in event_indexes:
            key_finding = key_findings.get(event_index)
            parameter_finding = parameter_findings.get(event_index)
            if key_finding is not None and key_finding.is_anomalous:
                key_triggered = True
                key_score = (
                    1.0
                    if key_finding.actual_probability is None
                    else (1.0 - key_finding.actual_probability)
                )
                scores.append(key_score)
            if parameter_finding is not None:
                self.scored_parameter_event_count += 1
                anomaly_score = parameter_anomaly_score(parameter_finding)
                if anomaly_score > 0.0:
                    scores.append(anomaly_score)
                if parameter_finding.is_anomalous:
                    parameter_triggered = True
            findings.append(
                DeepLogEventFinding(
                    event_index=event_index,
                    template=sequence.events[event_index][0],
                    key_model_finding=key_finding,
                    parameter_model_finding=parameter_finding,
                ),
            )
            self._record_event_level_decision(
                sequence=sequence,
                event_index=event_index,
                key_finding=key_finding,
                parameter_finding=parameter_finding,
            )

        if self._stream_context is not None:
            self._update_stream_context(
                sequence=sequence,
                stream_context=self._stream_context,
            )

        # An AnomaLog run still needs one sequence-level label and score, so we
        # aggregate the paper's event-level decisions here:
        #
        # - the sequence is anomalous if any event was anomalous
        # - the sequence score is the strongest event-level anomaly signal
        predicted_label = int(key_triggered or parameter_triggered)
        if sequence.continuous_context:
            self._sequence_trigger_breakdown_applicable = False
        else:
            self._record_sequence_trigger_breakdown(
                actual_is_anomalous=is_anomalous_label(sequence.label),
                key_triggered=key_triggered,
                parameter_triggered=parameter_triggered,
            )
            self._finalise_next_event_prediction_segment()
        return DeepLogPredictionOutcome(
            predicted_label=predicted_label,
            score=max(scores, default=0.0),
            triggered_by_key_model=key_triggered,
            triggered_by_parameter_model=parameter_triggered,
            findings=findings,
        )

    def model_manifest(self, *, sequence_summary: SequenceSummary) -> DeepLogManifest:
        """Return manifest metadata for the fitted DeepLog models.

        Args:
            sequence_summary (SequenceSummary): Shared sequence-count and label
                summary for the experiment run.

        Returns:
            DeepLogManifest: Serialisable metadata describing the fitted
            DeepLog run.
        """
        return DeepLogManifest.from_sequence_summary(
            detector=self.detector_name,
            sequence_summary=sequence_summary,
            implementation_scope="Scoped DeepLog core v1",
            parameter_schema_policy=(
                "disabled for this reproduction"
                if not self.config.parameter_detection_enabled
                else (
                    "strict: include only template parameter positions that are "
                    "always numeric in normal training data"
                )
            ),
            parameter_validation_policy=(
                "not applicable: HDFS paper reproduction uses key-only anomaly "
                "detection"
                if not self.config.parameter_detection_enabled
                else (
                    "per-template temporal tail split over history-target pairs; "
                    "Gaussian residuals come from held-out validation pairs "
                    "scored after training on each series prefix"
                )
            ),
            history_size=self.config.history_size,
            top_g=self.config.top_g,
            num_layers=self.config.num_layers,
            hidden_size=self.config.hidden_size,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            validation_fraction=self.config.validation_fraction,
            gaussian_confidence=self.config.gaussian_confidence,
            parameter_detection_enabled=self.config.parameter_detection_enabled,
            include_elapsed_time=self.config.include_elapsed_time,
            train_key_vocabulary_size=len(self.template_to_index),
            trained_parameter_model_count=len(self.parameter_models),
            skipped_parameter_model_count=len(self.skipped_parameter_models),
            train_parameter_covered_event_count=self.train_parameter_covered_event_count,
            train_parameter_covered_event_fraction=_fraction(
                self.train_parameter_covered_event_count,
                self.train_event_count,
            ),
            scored_parameter_event_count=self.scored_parameter_event_count,
            scored_parameter_event_fraction=_fraction(
                self.scored_parameter_event_count,
                self.test_event_count,
            ),
            parameter_models=[
                ParameterModelManifestEntry(
                    template=template,
                    feature_count=len(state.schema.feature_names),
                    input_feature_count=parameter_model_input_size(
                        feature_count=len(state.schema.feature_names),
                    ),
                    feature_names=state.schema.feature_names,
                    numeric_parameter_positions=state.schema.numeric_parameter_positions,
                    dropped_parameter_positions=state.schema.dropped_parameter_positions,
                    gaussian_mean=state.gaussian.mean,
                    gaussian_stddev=state.gaussian.stddev,
                    gaussian_lower_bound=state.gaussian.lower_bound,
                    gaussian_upper_bound=state.gaussian.upper_bound,
                )
                for template, state in sorted(self.parameter_models.items())
            ],
            skipped_parameter_models=[
                SkippedParameterModelEntry(template=template, reason=reason)
                for template, reason in sorted(self.skipped_parameter_models.items())
            ],
        )

    def run_metrics(self, *, run_metrics: dict[str, Any]) -> DeepLogRunMetrics:
        """Return DeepLog-specific run metrics for the latest evaluation.

        Args:
            run_metrics (dict[str, Any]): Generic run metrics accumulated by
                the shared evaluator.

        Returns:
            DeepLogRunMetrics: DeepLog-owned metrics for the latest scoring
            run.
        """
        del run_metrics
        segment_diagnostics = self._next_event_prediction_segment_snapshot()
        next_event_prediction = self._next_event_prediction_state_snapshot(
            segment_diagnostics=segment_diagnostics,
        )
        event_level_detection = self._event_level_state_snapshot()
        sequence_trigger_breakdown = (
            self._sequence_trigger_breakdown_snapshot()
            if self._sequence_trigger_breakdown_applicable
            else None
        )
        self._stream_context = None
        self._reset_next_event_prediction_state()
        self._reset_event_level_state()
        self._reset_sequence_trigger_breakdown()
        self._reset_next_event_prediction_segment_state()
        self._sequence_trigger_breakdown_applicable = True
        return DeepLogRunMetrics(
            next_event_prediction=next_event_prediction,
            event_level_detection=event_level_detection,
            sequence_trigger_breakdown=sequence_trigger_breakdown,
        )

    def _record_next_event_predictions(
        self,
        *,
        sequence: TemplateSequence,
        key_findings: dict[int, DeepLogKeyFinding],
        evaluation_event_indexes: set[int],
        segment_state: _NextEventPredictionSegmentState | None,
    ) -> None:
        state = self._ensure_next_event_prediction_state()
        if segment_state is None:
            msg = "prediction segment state must be initialised before scoring."
            raise RuntimeError(msg)
        for event_index, template in enumerate(sequence.templates):
            if event_index not in evaluation_event_indexes:
                continue
            key_finding = key_findings.get(event_index)
            if key_finding is None:
                state.record_exclusion(
                    NextEventPredictionExclusionReason.INSUFFICIENT_HISTORY,
                )
                segment_state.record_event(has_history=False)
                continue
            segment_state.record_event(has_history=True)
            state.record_observation(
                actual_label=template,
                predicted_labels=[
                    prediction.template for prediction in key_finding.top_predictions
                ],
                target_is_known=not key_finding.is_oov,
                history_is_known=not key_finding.unknown_history_templates,
            )

    def _ensure_next_event_prediction_state(self) -> NextEventPredictionState:
        state = self._next_event_prediction_state
        if state is None:
            state = NextEventPredictionState.create(
                k_values=_next_event_k_values(self.config.top_g),
                vocabulary_policy=self.config.vocabulary_policy,
            )
            self._next_event_prediction_state = state
        return state

    def _reset_next_event_prediction_state(self) -> None:
        """Reset next-event diagnostics before a fresh scoring run."""
        self._next_event_prediction_state = NextEventPredictionState.create(
            k_values=_next_event_k_values(self.config.top_g),
            vocabulary_policy=self.config.vocabulary_policy,
        )

    def _reset_next_event_prediction_segment_state(self) -> None:
        """Reset segment diagnostics before a fresh scoring run."""
        self._next_event_prediction_segment_state = None

    def _prepare_next_event_prediction_segment(
        self,
        *,
        sequence: TemplateSequence,
        prefix_length: int,
    ) -> None:
        state = self._next_event_prediction_segment_state
        source_object_type = _prediction_segment_source_object_type(sequence)
        if state is None:
            state = _NextEventPredictionSegmentState(
                history_size=self.config.history_size,
            )
            self._next_event_prediction_segment_state = state
        if sequence.continuous_context:
            if state.active:
                return
            boundary_reason = (
                NextEventPredictionSegmentBoundaryReason.STREAM_START
                if not state.summaries
                else NextEventPredictionSegmentBoundaryReason.CONTEXT_RESET
            )
            state.start_segment(
                prefix_length=prefix_length,
                continuous_context=True,
                boundary_reason=boundary_reason,
                source_object_type=source_object_type,
            )
            return
        if state.active:
            state.finalise_segment()
        state.start_segment(
            prefix_length=prefix_length,
            continuous_context=False,
            boundary_reason=NextEventPredictionSegmentBoundaryReason.STANDALONE_SEQUENCE,
            source_object_type=source_object_type,
        )

    def _prediction_prefix_templates(
        self,
        sequence: TemplateSequence,
    ) -> list[str] | None:
        """Return carried key-history templates for one predicted sequence.

        Args:
            sequence (TemplateSequence): Sequence being scored.

        Returns:
            list[str] | None: Carried template history, or `None` when the
            sequence should not inherit previous context.
        """
        if not sequence.continuous_context or self._stream_context is None:
            return None
        return self._stream_context.key_templates

    def _prediction_prefix_length(self, sequence: TemplateSequence) -> int:
        """Return the number of carried key-history templates for one sequence.

        Args:
            sequence (TemplateSequence): Sequence being scored.

        Returns:
            int: Number of carried templates available to the sequence.
        """
        prefix_templates = self._prediction_prefix_templates(sequence)
        if prefix_templates is None:
            return 0
        return len(prefix_templates)

    def _finalise_next_event_prediction_segment(self) -> None:
        state = self._next_event_prediction_segment_state
        if state is None:
            return
        if state.active and not state.current_continuous_context:
            state.finalise_segment()

    def _next_event_prediction_segment_snapshot(
        self,
    ) -> NextEventPredictionSegmentDiagnostics | None:
        """Return the recorded segment diagnostics for the latest run.

        Returns:
            NextEventPredictionSegmentDiagnostics | None: Finalised segment
            diagnostics, or `None` when no scored segments were recorded.
        """
        state = self._next_event_prediction_segment_state
        if state is None:
            return None
        return state.snapshot()

    def _reset_event_level_state(self) -> None:
        self._event_level_events_seen = 0
        self._event_level_events_eligible = 0
        self._event_level_tp = 0
        self._event_level_tn = 0
        self._event_level_fp = 0
        self._event_level_fn = 0
        self._event_level_normal_count = 0
        self._event_level_anomalous_count = 0

    def _next_event_prediction_state_snapshot(
        self,
        *,
        segment_diagnostics: NextEventPredictionSegmentDiagnostics | None = None,
    ) -> NextEventPredictionDiagnostics | None:
        """Return next-event diagnostics for the latest scoring run.

        Args:
            segment_diagnostics (NextEventPredictionSegmentDiagnostics | None):
                Optional segment-level diagnostics to attach.

        Returns:
            NextEventPredictionDiagnostics | None: Latest next-event
            diagnostics, or `None` when no eligible events were observed.
        """
        state = self._next_event_prediction_state
        if state is None:
            return None
        return state.snapshot(segment_diagnostics=segment_diagnostics)

    def _event_level_state_snapshot(
        self,
    ) -> DeepLogEventLevelDetectionDiagnostics | None:
        if self._event_level_events_seen <= 0:
            return None
        precision = (
            self._event_level_tp / (self._event_level_tp + self._event_level_fp)
            if (self._event_level_tp + self._event_level_fp)
            else 0.0
        )
        recall = (
            self._event_level_tp / (self._event_level_tp + self._event_level_fn)
            if (self._event_level_tp + self._event_level_fn)
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return DeepLogEventLevelDetectionDiagnostics(
            task="event_level_detection",
            events_seen=self._event_level_events_seen,
            events_eligible=self._event_level_events_eligible,
            tp=self._event_level_tp,
            tn=self._event_level_tn,
            fp=self._event_level_fp,
            fn=self._event_level_fn,
            normal_event_count=self._event_level_normal_count,
            anomalous_event_count=self._event_level_anomalous_count,
            precision=round(precision, 8),
            recall=round(recall, 8),
            f1=round(f1, 8),
        )

    def _sequence_trigger_breakdown_snapshot(
        self,
    ) -> DeepLogSequenceTriggerBreakdown | None:
        if self._sequence_total_count <= 0:
            return None
        return DeepLogSequenceTriggerBreakdown(
            total_sequences=self._sequence_total_count,
            normal_sequences=self._sequence_normal_count,
            anomalous_sequences=self._sequence_anomalous_count,
            key_only_normal_sequences=self._sequence_key_only_normal_count,
            key_only_anomalous_sequences=self._sequence_key_only_anomalous_count,
            parameter_only_normal_sequences=self._sequence_parameter_only_normal_count,
            parameter_only_anomalous_sequences=self._sequence_parameter_only_anomalous_count,
            both_normal_sequences=self._sequence_both_normal_count,
            both_anomalous_sequences=self._sequence_both_anomalous_count,
            neither_normal_sequences=self._sequence_neither_normal_count,
            neither_anomalous_sequences=self._sequence_neither_anomalous_count,
        )

    def _reset_sequence_trigger_breakdown(self) -> None:
        self._sequence_total_count = 0
        self._sequence_normal_count = 0
        self._sequence_anomalous_count = 0
        self._sequence_key_only_normal_count = 0
        self._sequence_key_only_anomalous_count = 0
        self._sequence_parameter_only_normal_count = 0
        self._sequence_parameter_only_anomalous_count = 0
        self._sequence_both_normal_count = 0
        self._sequence_both_anomalous_count = 0
        self._sequence_neither_normal_count = 0
        self._sequence_neither_anomalous_count = 0

    def _update_stream_context(
        self,
        *,
        sequence: TemplateSequence,
        stream_context: DeepLogStreamContext,
    ) -> None:
        """Advance the cached stream history with one chronological batch.

        Args:
            sequence (TemplateSequence): Newly scored chronological batch.
            stream_context (DeepLogStreamContext): Mutable cache carrying the
                prior stream tail.
        """
        stream_context.key_templates.extend(sequence.templates)
        if len(stream_context.key_templates) > self.config.history_size:
            stream_context.key_templates = stream_context.key_templates[
                -self.config.history_size :
            ]
        for template, parameters, dt_prev_ms in sequence.events:
            history = stream_context.parameter_events_by_template.setdefault(
                template,
                [],
            )
            history.append((parameters, dt_prev_ms))
            if len(history) > self.config.history_size:
                stream_context.parameter_events_by_template[template] = history[
                    -self.config.history_size :
                ]

    def _record_sequence_trigger_breakdown(
        self,
        *,
        actual_is_anomalous: bool,
        key_triggered: bool,
        parameter_triggered: bool,
    ) -> None:
        self._sequence_total_count += 1
        label_attr = _SEQUENCE_LABEL_COUNT_ATTRS[int(actual_is_anomalous)]
        trigger_attr = _SEQUENCE_TRIGGER_COUNT_ATTRS[
            key_triggered,
            parameter_triggered,
        ][int(actual_is_anomalous)]
        setattr(self, label_attr, getattr(self, label_attr) + 1)
        setattr(self, trigger_attr, getattr(self, trigger_attr) + 1)

    def _record_event_level_decision(
        self,
        *,
        sequence: TemplateSequence,
        event_index: int,
        key_finding: DeepLogKeyFinding | None,
        parameter_finding: DeepLogParameterFinding | None,
    ) -> None:
        if sequence.event_labels is None:
            return
        if event_index >= len(sequence.event_labels):
            return
        actual_label = sequence.event_labels[event_index]
        if actual_label is None:
            return
        self._event_level_events_seen += 1
        actual_is_anomalous = is_anomalous_label(actual_label)
        if actual_is_anomalous:
            self._event_level_anomalous_count += 1
        else:
            self._event_level_normal_count += 1
        if key_finding is None and parameter_finding is None:
            return
        self._event_level_events_eligible += 1
        predicted_is_anomalous = (
            key_finding is not None and key_finding.is_anomalous
        ) or (parameter_finding is not None and parameter_finding.is_anomalous)
        if actual_is_anomalous and predicted_is_anomalous:
            self._event_level_tp += 1
        elif not actual_is_anomalous and not predicted_is_anomalous:
            self._event_level_tn += 1
        elif not actual_is_anomalous and predicted_is_anomalous:
            self._event_level_fp += 1
        else:
            self._event_level_fn += 1


def _prediction_segment_source_object_type(
    sequence: TemplateSequence,
) -> NextEventPredictionSegmentSourceType:
    """Infer a stable source-object label for one prediction segment.

    Args:
        sequence (TemplateSequence): Sequence being scored.

    Returns:
        NextEventPredictionSegmentSourceType: Best-effort source-object label
        for the segment.
    """
    if sequence.continuous_context:
        return NextEventPredictionSegmentSourceType.CHRONOLOGICAL_CHUNK
    if sequence.sole_entity_id is not None:
        return NextEventPredictionSegmentSourceType.ENTITY
    if not sequence.entity_ids:
        return NextEventPredictionSegmentSourceType.CACHED_SEQUENCE
    if len(sequence.entity_ids) == 1:
        return NextEventPredictionSegmentSourceType.FILE_CHUNK
    return NextEventPredictionSegmentSourceType.STREAM_PARTITION


def _fraction(numerator: int, denominator: int) -> float:
    """Return a rounded fraction for manifest reporting.

    Args:
        numerator (int): Numerator for the fraction.
        denominator (int): Denominator for the fraction.

    Returns:
        float: Fraction value, or `0.0` when the denominator is zero.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _next_event_k_values(top_g: int) -> tuple[int, ...]:
    """Return the DeepLog next-event reporting cut-offs.

    Args:
        top_g (int): DeepLog top-g reporting threshold from the config.

    Returns:
        tuple[int, ...]: Ordered top-k cut-offs up to `top_g`.
    """
    return tuple(sorted({k for k in (1, 2, 3, 5, top_g) if 0 < k <= top_g}))


_SEQUENCE_LABEL_COUNT_ATTRS: tuple[str, str] = (
    "_sequence_normal_count",
    "_sequence_anomalous_count",
)
_SEQUENCE_TRIGGER_COUNT_ATTRS: dict[
    tuple[bool, bool],
    tuple[str, str],
] = {
    (False, False): (
        "_sequence_neither_normal_count",
        "_sequence_neither_anomalous_count",
    ),
    (True, False): (
        "_sequence_key_only_normal_count",
        "_sequence_key_only_anomalous_count",
    ),
    (False, True): (
        "_sequence_parameter_only_normal_count",
        "_sequence_parameter_only_anomalous_count",
    ),
    (True, True): (
        "_sequence_both_normal_count",
        "_sequence_both_anomalous_count",
    ),
}
