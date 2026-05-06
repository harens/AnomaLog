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
            Sequence-level trigger source counts split by actual label.
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
    ] = 30
    batch_size: Annotated[
        PositiveInt,
        msgspec.Meta(
            description="Training batch size for DeepLog neural models. "
            "This is not defined in the paper.",
        ),
    ] = 128
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

        training_corpus = build_normal_training_corpus(
            train_sequences,
            progress=progress,
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
        )
        self._record_next_event_predictions(
            sequence=sequence,
            key_findings=key_findings,
            evaluation_event_indexes=evaluation_event_indexes,
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

        # An AnomaLog run still needs one sequence-level label and score, so we
        # aggregate the paper's event-level decisions here:
        #
        # - the sequence is anomalous if any event was anomalous
        # - the sequence score is the strongest event-level anomaly signal
        predicted_label = int(key_triggered or parameter_triggered)
        self._record_sequence_trigger_breakdown(
            actual_is_anomalous=is_anomalous_label(sequence.label),
            key_triggered=key_triggered,
            parameter_triggered=parameter_triggered,
        )
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
        next_event_prediction = self._next_event_prediction_state_snapshot()
        event_level_detection = self._event_level_state_snapshot()
        sequence_trigger_breakdown = self._sequence_trigger_breakdown_snapshot()
        self._reset_next_event_prediction_state()
        self._reset_event_level_state()
        self._reset_sequence_trigger_breakdown()
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
    ) -> None:
        state = self._ensure_next_event_prediction_state()
        for event_index, template in enumerate(sequence.templates):
            if event_index not in evaluation_event_indexes:
                continue
            if event_index < self.config.history_size:
                state.record_exclusion(
                    NextEventPredictionExclusionReason.INSUFFICIENT_HISTORY,
                )
                continue
            key_finding = key_findings.get(event_index)
            if key_finding is None:
                state.record_exclusion(
                    NextEventPredictionExclusionReason.INSUFFICIENT_HISTORY,
                )
                continue
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

    def _reset_event_level_state(self) -> None:
        self._event_level_events_seen = 0
        self._event_level_events_eligible = 0
        self._event_level_tp = 0
        self._event_level_tn = 0
        self._event_level_fp = 0
        self._event_level_fn = 0

    def _next_event_prediction_state_snapshot(
        self,
    ) -> NextEventPredictionDiagnostics | None:
        """Return next-event diagnostics for the latest scoring run.

        Returns:
            NextEventPredictionDiagnostics | None: Latest next-event
            diagnostics, or `None` when no eligible events were observed.
        """
        state = self._next_event_prediction_state
        if state is None:
            return None
        return state.snapshot()

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
        if key_finding is None and parameter_finding is None:
            return
        self._event_level_events_eligible += 1
        actual_is_anomalous = is_anomalous_label(actual_label)
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
