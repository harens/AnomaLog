"""DeepCase detector integration for AnomaLog experiments."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import msgspec
import numpy as np
import scipy.sparse as sp
import torch
from deepcase.interpreter.utils import group_by, sp_unique
from sklearn.neighbors import KDTree
from typing_extensions import override

from deepcase import DeepCASE
from experiments.models.base import (
    AbstainAwarePredictionOutcome,
    ExperimentDetector,
    ExperimentModelConfig,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    Probability,
    SequenceSummary,
    SingleFitMixin,
)
from experiments.models.deepcase.shared import (
    DeepCaseClusterScoreStrategy,
    DeepCaseEventFinding,
    DeepCaseEventIdMap,
    DeepCaseManifest,
    DeepCasePredictionDiagnostics,
    DeepCaseSequenceDecision,
    DeepCaseWorkloadAlertSampling,
    DeepCaseWorkloadMode,
    DeepCaseWorkloadReductionMetrics,
    _DeepCasePredictionDiagnosticsState,
    _DeepCasePredictionSummary,
    aggregate_sequence_score,
    build_sample_batch,
    build_training_batch_from_map,
    build_workload_reduction_metrics,
    cluster_scores_for_labels,
    decision_label_for_score,
    finding_reason_for_score,
)
from experiments.models.next_event_metrics import (
    NextEventPredictionDiagnostics,
    NextEventPredictionState,
    VocabularyPolicy,
)
from experiments.models.torch_runtime import (
    TorchDeviceName,
    resolve_torch_device,
    set_torch_seed,
)

DEEPCASE_TRAINING_SAMPLE_CHUNK_SIZE = 32_768
DEEPCASE_PREDICTION_SAMPLE_CHUNK_SIZE = 32_768

if TYPE_CHECKING:
    import logging
    from collections.abc import Iterable, Iterator, Sequence

    from deepcase.interpreter.interpreter import Interpreter
    from rich.progress import Progress

    from anomalog.sequences import TemplateSequence
    from experiments.models.deepcase.shared import DeepCaseSampleBatch


@dataclass(frozen=True, slots=True)
class DeepCasePredictionOutcome(
    AbstainAwarePredictionOutcome,
):
    """DeepCase runtime prediction plus event-level findings.

    Attributes:
        findings (list[DeepCaseEventFinding]): Event-level DeepCase findings
            for each scored event in the sequence.
        sequence_decision (DeepCaseSequenceDecision): Sequence-level decision
            category derived from the event findings.
        confident_event_count (int): Number of events with a confident label.
        abstained_event_count (int): Number of events that should be reviewed
            manually instead of being treated as automatic anomalies.
        confident_anomaly_event_count (int): Number of confidently anomalous
            events within the sequence.
    """

    findings: list[DeepCaseEventFinding]
    sequence_decision: DeepCaseSequenceDecision
    confident_event_count: int
    abstained_event_count: int
    confident_anomaly_event_count: int

    @property
    @override
    def is_abstained(self) -> bool:
        """Return whether the sequence decision deferred to manual review.

        Returns:
            bool: True when the sequence should be reviewed manually instead
                of being treated as an automatic decision.
        """
        return self.sequence_decision is DeepCaseSequenceDecision.ABSTAINED


class DeepCaseRunMetrics(msgspec.Struct, frozen=True):
    """DeepCASE-specific run metrics for a single evaluation.

    Attributes:
        auto_decision_count (int): Number of confident automatic decisions.
        abstained_prediction_count (int): Number of deferred test sequences.
        abstained_anomalous_label_count (int): Deferred anomalous sequences.
        abstained_normal_label_count (int): Deferred normal sequences.
        parent_sequence_fallback_count (int): Number of scored events that had
            to fall back to the parent sequence label.
        auto_coverage (float): Fraction of test sequences handled
            automatically.
        abstain_rate (float): Fraction of test sequences deferred for
            review.
        random_seed (int): Configured random seed for the latest run.
        prediction_diagnostics (DeepCasePredictionDiagnostics | None): Event
            and sequence diagnostics for the latest DeepCASE scoring run.
        next_event_prediction (NextEventPredictionDiagnostics | None): Latest
            Context Builder next-event diagnostics.
        manual_workload_reduction (DeepCaseWorkloadReductionMetrics | None):
            Manual-mode workload reduction summary, if available.
        semi_automatic_workload_reduction (DeepCaseWorkloadReductionMetrics |
            None): Semi-automatic workload reduction summary, if available.
    """

    auto_decision_count: int
    abstained_prediction_count: int
    abstained_anomalous_label_count: int
    abstained_normal_label_count: int
    parent_sequence_fallback_count: int
    auto_coverage: float
    abstain_rate: float
    random_seed: int
    prediction_diagnostics: DeepCasePredictionDiagnostics | None
    next_event_prediction: NextEventPredictionDiagnostics | None = None
    manual_workload_reduction: DeepCaseWorkloadReductionMetrics | None = None
    semi_automatic_workload_reduction: DeepCaseWorkloadReductionMetrics | None = None


class DeepCaseModelConfig(
    ExperimentModelConfig,
    tag="deepcase",
    frozen=True,
):
    """Configuration for the DeepCASE experiment detector.

    Parameters are grouped by subsystem from the original paper:
    sequencing events, context builder, interpreter, and torch runtime.

    Attributes:
        context_length (PositiveInt): Number of prior events retained in the context
            window.
        timeout_seconds (PositiveFloat): Maximum age of a context event.
        hidden_size (PositiveInt): Hidden dimension of the context builder
            encoder.
        label_smoothing_delta (Probability): Label smoothing used during
            training.
        confidence_threshold (Probability): Interpreter confidence threshold.
        eps (PositiveFloat): DBSCAN neighbourhood radius.
        min_samples (PositiveInt): Minimum DBSCAN cluster size.
        epochs (PositiveInt): Context-builder training epochs.
        batch_size (PositiveInt): Context-builder batch size.
        learning_rate (PositiveFloat): Context-builder optimiser learning
            rate.
        teach_ratio (Probability): Teacher-forcing ratio.
        iterations (NonNegativeInt): Interpreter query iterations used while
            building clusters.
        attention_query_iterations (NonNegativeInt): Prediction-time
            attention-query iterations. The paper-faithful default is 100;
            zero is reserved for explicit ablation or smoke-test variants.
        query_batch_size (PositiveInt): Interpreter query batch size.
        vocabulary_policy (VocabularyPolicy): Vocabulary policy for
            next-event diagnostics.
        cluster_score_strategy (DeepCaseClusterScoreStrategy): Cluster score
            labelling strategy.
        cluster_anomaly_fraction_threshold (Probability): Minimum anomaly
            fraction required by the thresholded cluster policy.
        no_score (int): Special no-score sentinel value.
        device (TorchDeviceName): Requested Torch device policy.
        random_seed (int): Deterministic random seed.
    """  # noqa: DOC605

    # Sequencing events subsystem (Section D-A)

    context_length: Annotated[
        PositiveInt,
        msgspec.Meta(
            description=(
                "Number of prior events in the same-device context window. "
                "In the original paper, context_length=10 with left-padding for "
                "shorter sequences."
            ),
        ),
    ] = 10

    timeout_seconds: Annotated[
        PositiveFloat,
        msgspec.Meta(
            description=(
                "Maximum time gap between context events and the target event. "
                "In the original paper, timeout_seconds=86,400 (24 hours)."
            ),
        ),
    ] = 86_400

    # Context builder subsystem (Section D-B)

    hidden_size: Annotated[
        PositiveInt,
        msgspec.Meta(
            description=(
                "Hidden dimension of the context builder encoder. "
                "In the original paper, hidden_size=128."
            ),
        ),
    ] = 128

    label_smoothing_delta: Annotated[
        Probability,
        msgspec.Meta(
            description=(
                "Label smoothing delta used when training the context builder. "
                "In the original paper, label_smoothing_delta=0.1."
            ),
        ),
    ] = 0.1

    # Interpreter subsystem (Section D-C)

    confidence_threshold: Annotated[
        Probability,
        msgspec.Meta(
            description=(
                "Minimum confidence required to accept a corrected attention "
                "distribution during attention querying. Lower-confidence cases "
                "are passed for manual inspection. In the original paper, "
                "confidence_threshold=0.2."
            ),
        ),
    ] = 0.2

    eps: Annotated[
        PositiveFloat,
        msgspec.Meta(
            description=(
                "DBSCAN neighborhood radius for interpreter clustering. "
                "In the original paper, eps=0.1."
            ),
        ),
    ] = 0.1

    min_samples: Annotated[
        PositiveInt,
        msgspec.Meta(
            description=(
                "Minimum cluster size for DBSCAN. Smaller groups are passed "
                "directly to the security operator. In the original paper, "
                "min_samples=5."
            ),
        ),
    ] = 5

    # ContextBuilder-specific parameters (defined in deepcase library's fit method)

    epochs: Annotated[
        PositiveInt,
        msgspec.Meta(
            description=(
                "Training epochs for the context builder. The paper-faithful "
                "path uses 100."
            ),
        ),
    ] = 100
    batch_size: Annotated[
        PositiveInt,
        msgspec.Meta(
            description=(
                "Batch size for training the context builder. In the original "
                "paper, the input is encoded into a 128-dimensional context vector."
            ),
        ),
    ] = 128
    learning_rate: Annotated[
        PositiveFloat,
        msgspec.Meta(description="Optimiser learning rate for the context builder."),
    ] = 0.01
    teach_ratio: Annotated[
        Probability,
        msgspec.Meta(description="Ratio of sequences to train with."),
    ] = 0.5

    # Interpreter-specific parameters (defined in deepcase library's fit method)

    iterations: Annotated[
        NonNegativeInt,
        msgspec.Meta(
            description=(
                "Maximum attention-querying iterations used while building "
                "DeepCase interpreter clusters. In the paper-faithful path, "
                "this remains 100 for both clustering and interpreter query "
                "refinement."
            ),
        ),
    ] = 100
    attention_query_iterations: Annotated[
        NonNegativeInt,
        msgspec.Meta(
            description=(
                "Attention-query iterations used during prediction-time "
                "scoring. The paper-faithful path uses 100; zero is reserved "
                "for ablation or smoke-test variants. If ContextBuilder.query() "
                "is called directly, pass iterations=100 explicitly because the "
                "low-level default is zero."
            ),
        ),
    ] = 100
    query_batch_size: Annotated[
        PositiveInt,
        msgspec.Meta(description="Batch size used during interpreter querying."),
    ] = 1024
    vocabulary_policy: Annotated[
        VocabularyPolicy,
        msgspec.Meta(
            description=(
                "Vocabulary policy used for next-event diagnostics. "
                "The maintained baseline uses the complete dataset "
                "vocabulary, but train-only mode is available for closed-"
                "world comparisons."
            ),
        ),
    ] = VocabularyPolicy.FULL_DATASET
    cluster_score_strategy: Annotated[
        DeepCaseClusterScoreStrategy,
        msgspec.Meta(
            description=(
                "Cluster-labelling policy used after interpreter clustering. "
                "'max' keeps the paper-faithful any-anomalous baseline, "
                "'majority_vote' labels clusters by the strict majority, "
                "'threshold_fraction' uses the configured anomaly fraction "
                "cut-off, and 'abstain_mixed' defers mixed clusters."
            ),
        ),
    ] = DeepCaseClusterScoreStrategy.ANY_ANOMALOUS
    cluster_anomaly_fraction_threshold: Annotated[
        Probability,
        msgspec.Meta(
            description=(
                "Minimum anomaly fraction required by the "
                "'threshold_fraction' cluster policy."
            ),
        ),
    ] = 0.75
    no_score: Annotated[
        int,
        msgspec.Meta(
            description="Sample has no score and is ignored during clustering.",
        ),
    ] = -1

    # Torch runtime parameters

    device: Annotated[
        TorchDeviceName,
        msgspec.Meta(description="Torch device selection: auto, cpu, cuda, or mps."),
    ] = "auto"
    random_seed: Annotated[
        int,
        msgspec.Meta(description="Random seed used for deterministic torch training."),
    ] = 0

    def build_detector(self) -> DeepCaseDetector:
        """Construct the DeepCase detector.

        Returns:
            DeepCaseDetector: Configured detector instance.
        """
        return DeepCaseDetector(config=self)


@dataclass(slots=True)
class DeepCaseDetector(SingleFitMixin, ExperimentDetector):
    """DeepCase workflow adapted to AnomaLog entity sequences.

    Attributes:
        detector_name (ClassVar[str]): Stable detector registry name.
        config (DeepCaseModelConfig): Immutable detector configuration.
        model (DeepCASE | None): Fitted upstream DeepCASE model.
        event_id_map (DeepCaseEventIdMap | None): Train-time template-to-event
            id mapping reused during prediction.
        device (torch.device): Resolved runtime torch device. The config stores
            the requested device policy, and this field stores the actual device
            used after fitting.
        train_sample_count (int): Number of event-centered samples used for
            training.
        clustered_sample_count (int): Number of training samples assigned to a
            non-noise cluster.
        known_cluster_count (int): Number of non-noise clusters learned during
            training.
        known_benign_cluster_count (int): Number of training samples whose
            cluster score was benign.
        known_malicious_cluster_count (int): Number of training samples whose
            cluster score was malicious.
        unknown_cluster_score_count (int): Number of training samples that
            remained unclustered or otherwise unscored.
    """

    detector_name: ClassVar[str] = "deepcase"
    config: DeepCaseModelConfig
    model: DeepCASE | None = None
    event_id_map: DeepCaseEventIdMap | None = None
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    train_sample_count: int = 0
    clustered_sample_count: int = 0
    known_cluster_count: int = 0
    known_benign_cluster_count: int = 0
    known_malicious_cluster_count: int = 0
    unknown_cluster_score_count: int = 0
    _next_event_prediction_state: NextEventPredictionState | None = field(
        default=None,
        repr=False,
    )
    _prediction_diagnostics_state: _DeepCasePredictionDiagnosticsState = field(
        default_factory=_DeepCasePredictionDiagnosticsState,
        repr=False,
    )

    def fit(
        self,
        train_sequences: Iterable[TemplateSequence],
        *,
        progress: Progress,
        logger: logging.Logger | None = None,
    ) -> None:
        """Fit DeepCase's Context Builder and Interpreter.

        Args:
            train_sequences (Iterable[TemplateSequence]): Training split.
            progress (Progress): Progress reporter.
            logger (logging.Logger | None): Optional logger for diagnostics.

        Raises:
            ValueError: If the train split has no event samples.
        """
        self._ensure_unfit(detector_name=self.detector_name)
        train_sequences_list = list(train_sequences)
        event_id_map = DeepCaseEventIdMap.from_sequences(train_sequences_list)
        if _training_sample_count_total(train_sequences_list) == 0:
            msg = "DeepCase requires at least one training event sample."
            raise ValueError(msg)

        set_torch_seed(self.config.random_seed)
        device = resolve_torch_device(self.config.device)
        if logger is not None:
            logger.info("DeepCase resolved torch device: %s", device)

        model = DeepCASE(
            features=len(event_id_map.event_id_to_template),
            max_length=self.config.context_length,
            hidden_size=self.config.hidden_size,
            eps=self.config.eps,
            min_samples=self.config.min_samples,
            threshold=self.config.confidence_threshold,
        ).to(str(device))
        fit_task = progress.add_task(
            "DeepCase: training context builder",
            total=self.config.epochs + 1,
        )
        if logger is not None:
            logger.info("Training DeepCase context builder")
        _fit_context_builder_in_chunks(
            model=model,
            train_sequences=train_sequences_list,
            event_id_map=event_id_map,
            config=self.config,
        )
        for _ in range(self.config.epochs):
            progress.advance(fit_task)
        progress.update(fit_task, description="DeepCase: clustering interpreter")
        if logger is not None:
            logger.info("Clustering DeepCase interpreter")
        clustered_samples = _cluster_training_samples(
            model=model,
            train_sequences=train_sequences_list,
            event_id_map=event_id_map,
            config=self.config,
            device=device,
        )
        model.interpreter.clusters = clustered_samples.clusters
        model.interpreter.vectors = sp.csc_matrix((0, model.interpreter.features))
        model.interpreter.events = np.zeros(0, dtype=int)
        progress.advance(fit_task)
        progress.update(fit_task, description="DeepCase: fit complete")

        self.model = model
        self.event_id_map = event_id_map
        self.device = device
        self.train_sample_count = int(clustered_samples.labels.shape[0])
        self.clustered_sample_count = int(
            np.count_nonzero(clustered_samples.clusters != -1),
        )
        self.known_cluster_count = len(
            {cluster for cluster in clustered_samples.clusters if cluster != -1},
        )
        self.known_benign_cluster_count = int(
            np.count_nonzero(
                (clustered_samples.clusters != -1)
                & np.isclose(clustered_samples.scores, 0.0),
            ),
        )
        self.known_malicious_cluster_count = int(
            np.count_nonzero(
                (clustered_samples.clusters != -1)
                & np.isclose(clustered_samples.scores, 1.0),
            ),
        )
        self.unknown_cluster_score_count = int(
            np.count_nonzero(
                np.isclose(clustered_samples.scores, float(self.config.no_score)),
            ),
        )
        self._reset_next_event_prediction_state()
        self._reset_prediction_diagnostics()
        self._mark_fit_complete()

    def predict(self, sequence: TemplateSequence) -> DeepCasePredictionOutcome:
        """Return DeepCase findings aggregated to one sequence prediction.

        Args:
            sequence (TemplateSequence): Sequence to score.

        Returns:
            DeepCasePredictionOutcome: Sequence-level prediction with findings.

        Raises:
            ValueError: If the detector has not been fit.
        """
        if self.model is None or self.event_id_map is None:
            msg = "deepcase must be fit before prediction."
            raise ValueError(msg)
        self._reset_prediction_diagnostics()
        self._reset_next_event_prediction_state()
        batch = build_sample_batch(
            (sequence,),
            event_id_map=self.event_id_map,
            context_length=self.config.context_length,
            timeout_seconds=self.config.timeout_seconds,
            unknown_event_id=self.event_id_map.no_event_id,
        )
        self._prediction_diagnostics_state.record_parent_sequence_fallback_count(
            batch.parent_sequence_fallback_count,
        )
        if batch.sample_count == 0:
            self._prediction_diagnostics_state.record_empty_sequence(
                sequence_label=sequence.label,
            )
            return DeepCasePredictionOutcome(
                predicted_label=0,
                score=0.0,
                findings=[],
                sequence_decision=DeepCaseSequenceDecision.CONFIDENT_NORMAL,
                confident_event_count=0,
                abstained_event_count=0,
                confident_anomaly_event_count=0,
            )

        self._record_next_event_predictions(
            sequences=(sequence,),
            batch=batch,
        )
        raw_scores = self._predict_batch(batch)
        findings = _findings_from_scores(batch=batch, raw_scores=raw_scores)
        summary = _summarise_findings(findings=findings, raw_scores=raw_scores)
        self._prediction_diagnostics_state.record(
            summary=summary,
            findings=findings,
            event_labels=batch.scores.tolist(),
            sequence_label=sequence.label,
        )
        return DeepCasePredictionOutcome(
            predicted_label=summary.predicted_label,
            score=summary.score,
            findings=findings,
            sequence_decision=summary.sequence_decision,
            confident_event_count=summary.confident_event_count,
            abstained_event_count=summary.abstained_event_count,
            confident_anomaly_event_count=summary.confident_anomaly_event_count,
        )

    def predict_all(
        self,
        sequences: Iterable[TemplateSequence],
    ) -> Iterator[tuple[TemplateSequence, DeepCasePredictionOutcome]]:
        """Yield sequence predictions while batching upstream DeepCASE scoring.

        DeepCASE's interpreter is vectorised over many event samples. Replaying
        one upstream prediction call per ``TemplateSequence`` makes large test
        runs spend most of their time on repeated interpreter setup and nearest
        neighbour queries rather than on the actual batched computation. This
        method keeps the experiment output stream sequence-oriented while
        scoring bounded chunks of event samples together.

        Args:
            sequences (Iterable[TemplateSequence]): Test sequences to score.

        Yields:
            tuple[TemplateSequence, DeepCasePredictionOutcome]: Scored sequences
            in input order.
        """
        self._reset_prediction_diagnostics()
        self._reset_next_event_prediction_state()
        for chunk in _chunk_prediction_sequences_by_sample_count(
            sequences,
            max_sample_count=self._prediction_chunk_sample_limit(),
        ):
            yield from self._predict_sequence_chunk(chunk)

    def model_manifest(self, *, sequence_summary: SequenceSummary) -> DeepCaseManifest:
        """Return manifest metadata for the fitted DeepCase workflow.

        Args:
            sequence_summary (SequenceSummary): Shared sequence-count and label
                summary for the experiment run.

        Returns:
            DeepCaseManifest: Serialisable metadata describing the fitted
            DeepCASE run.
        """
        if self.event_id_map is None:
            vocabulary_size = 0
        else:
            vocabulary_size = len(self.event_id_map.template_to_event_id)
        return DeepCaseManifest.from_sequence_summary(
            detector=self.detector_name,
            sequence_summary=sequence_summary,
            implementation_scope="Official DeepCase library integration",
            label_policy=(
                "event-label supervision when available: each event-centered "
                "sample uses its target event label and falls back to the "
                "parent TemplateSequence label when the event label is missing; "
                f"cluster labelling uses {self.config.cluster_score_strategy.value}"
            ),
            context_length=self.config.context_length,
            timeout_seconds=self.config.timeout_seconds,
            hidden_size=self.config.hidden_size,
            label_smoothing_delta=self.config.label_smoothing_delta,
            eps=self.config.eps,
            min_samples=self.config.min_samples,
            confidence_threshold=self.config.confidence_threshold,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            teach_ratio=self.config.teach_ratio,
            iterations=self.config.iterations,
            attention_query_iterations=self.config.attention_query_iterations,
            query_batch_size=self.config.query_batch_size,
            cluster_score_strategy=self.config.cluster_score_strategy.value,
            cluster_anomaly_fraction_threshold=self.config.cluster_anomaly_fraction_threshold,
            no_score=self.config.no_score,
            device=str(self.device),
            random_seed=self.config.random_seed,
            train_event_vocabulary_size=vocabulary_size,
            train_sample_count=self.train_sample_count,
            clustered_sample_count=self.clustered_sample_count,
            known_cluster_count=self.known_cluster_count,
            known_benign_cluster_count=self.known_benign_cluster_count,
            known_malicious_cluster_count=self.known_malicious_cluster_count,
            unknown_cluster_score_count=self.unknown_cluster_score_count,
            prediction_diagnostics=self._prediction_diagnostics_state.snapshot(),
            online_updates_status="not implemented",
            persistent_cluster_database_status="not implemented",
        )

    def run_metrics(self, *, run_metrics: dict[str, Any]) -> DeepCaseRunMetrics:
        """Return DeepCASE-specific run metrics for the latest evaluation.

        Args:
            run_metrics (dict[str, Any]): Generic run metrics accumulated by
                the shared evaluator.

        Returns:
            DeepCaseRunMetrics: DeepCASE-owned metrics for the latest scoring
            run.
        """
        test_sequence_count = int(run_metrics["test_sequence_count"])
        prediction_diagnostics = self._prediction_diagnostics_state.snapshot()
        next_event_prediction = self._next_event_prediction_state_snapshot()
        auto_decision_count = (
            0
            if prediction_diagnostics is None
            else (
                prediction_diagnostics.sequence_confident_anomaly_count
                + prediction_diagnostics.sequence_confident_normal_count
            )
        )
        abstained_prediction_count = (
            0
            if prediction_diagnostics is None
            else prediction_diagnostics.sequence_abstained_count
        )
        auto_coverage = (
            auto_decision_count / test_sequence_count if test_sequence_count else 0.0
        )
        abstain_rate = (
            abstained_prediction_count / test_sequence_count
            if test_sequence_count
            else 0.0
        )
        manual_workload_reduction = (
            None
            if prediction_diagnostics is None
            else build_workload_reduction_metrics(
                mode=DeepCaseWorkloadMode.MANUAL,
                total_contextual_sequence_count=self.train_sample_count,
                covered_contextual_sequence_count=self.clustered_sample_count,
                uncovered_contextual_sequence_count=self.unknown_cluster_score_count,
                alert_sampling=DeepCaseWorkloadAlertSampling(
                    cluster_count=self.known_cluster_count,
                    alerts_per_cluster=10,
                ),
            )
        )
        semi_automatic_workload_reduction = (
            None
            if prediction_diagnostics is None
            else build_workload_reduction_metrics(
                mode=DeepCaseWorkloadMode.SEMI_AUTOMATIC,
                total_contextual_sequence_count=prediction_diagnostics.event_count,
                covered_contextual_sequence_count=prediction_diagnostics.confident_event_count,
                uncovered_contextual_sequence_count=prediction_diagnostics.abstained_event_count,
                alert_sampling=DeepCaseWorkloadAlertSampling(
                    cluster_count=self.known_cluster_count,
                    alerts_per_cluster=10,
                ),
            )
        )
        return DeepCaseRunMetrics(
            auto_decision_count=auto_decision_count,
            abstained_prediction_count=abstained_prediction_count,
            abstained_anomalous_label_count=(
                0
                if prediction_diagnostics is None
                else prediction_diagnostics.abstained_anomalous_label_count
            ),
            abstained_normal_label_count=(
                0
                if prediction_diagnostics is None
                else prediction_diagnostics.abstained_normal_label_count
            ),
            parent_sequence_fallback_count=(
                self._prediction_diagnostics_state.parent_sequence_fallback_count
            ),
            auto_coverage=round(auto_coverage, 8),
            abstain_rate=round(abstain_rate, 8),
            random_seed=self.config.random_seed,
            prediction_diagnostics=prediction_diagnostics,
            next_event_prediction=next_event_prediction,
            manual_workload_reduction=manual_workload_reduction,
            semi_automatic_workload_reduction=semi_automatic_workload_reduction,
        )

    def _predict_batch(self, batch: DeepCaseSampleBatch) -> list[float]:
        model = self.model
        if model is None:
            msg = "deepcase must be fit before prediction."
            raise ValueError(msg)
        return _predict_batch_in_chunks(
            model=model,
            batch=batch,
            device=self.device,
            chunk_size=DEEPCASE_PREDICTION_SAMPLE_CHUNK_SIZE,
            config=self.config,
        )

    def _predict_sequence_chunk(
        self,
        sequences: Sequence[TemplateSequence],
    ) -> Iterator[tuple[TemplateSequence, DeepCasePredictionOutcome]]:
        """Score one bounded sequence chunk through a single DeepCASE call.

        Args:
            sequences (Sequence[TemplateSequence]): Test sequences to score
                together.

        Yields:
            (TemplateSequence, DeepCasePredictionOutcome): Sequence-level
            DeepCASE outcomes in input order.

        Raises:
            ValueError: If the detector has not been fit.
        """
        if self.event_id_map is None:
            msg = "deepcase must be fit before prediction."
            raise ValueError(msg)
        batch = build_sample_batch(
            sequences,
            event_id_map=self.event_id_map,
            context_length=self.config.context_length,
            timeout_seconds=self.config.timeout_seconds,
            unknown_event_id=self.event_id_map.no_event_id,
        )
        self._prediction_diagnostics_state.record_parent_sequence_fallback_count(
            batch.parent_sequence_fallback_count,
        )
        if batch.sample_count == 0:
            for sequence in sequences:
                self._prediction_diagnostics_state.record_empty_sequence(
                    sequence_label=sequence.label,
                )
                yield (
                    sequence,
                    DeepCasePredictionOutcome(
                        predicted_label=0,
                        score=0.0,
                        findings=[],
                        sequence_decision=DeepCaseSequenceDecision.CONFIDENT_NORMAL,
                        confident_event_count=0,
                        abstained_event_count=0,
                        confident_anomaly_event_count=0,
                    ),
                )
            return

        self._record_next_event_predictions(
            sequences=sequences,
            batch=batch,
        )
        raw_scores = self._predict_batch(batch)
        score_offset = 0
        for sequence in sequences:
            sample_count = _prediction_sample_count(sequence)
            sequence_scores = raw_scores[score_offset : score_offset + sample_count]
            findings = _findings_from_scores(
                batch=batch,
                raw_scores=sequence_scores,
                start_index=score_offset,
            )
            summary = _summarise_findings(
                findings=findings,
                raw_scores=sequence_scores,
            )
            self._prediction_diagnostics_state.record(
                summary=summary,
                findings=findings,
                event_labels=batch.scores[
                    score_offset : score_offset + sample_count
                ].tolist(),
                sequence_label=sequence.label,
            )
            yield (
                sequence,
                DeepCasePredictionOutcome(
                    predicted_label=summary.predicted_label,
                    score=summary.score,
                    findings=findings,
                    sequence_decision=summary.sequence_decision,
                    confident_event_count=summary.confident_event_count,
                    abstained_event_count=summary.abstained_event_count,
                    confident_anomaly_event_count=summary.confident_anomaly_event_count,
                ),
            )
            score_offset += sample_count

    def _prediction_chunk_sample_limit(self) -> int:
        """Return the bounded event-sample budget for one prediction chunk.

        Returns:
            int: Maximum number of event-centred samples to score together.
        """
        return max(
            self.config.query_batch_size,
            DEEPCASE_PREDICTION_SAMPLE_CHUNK_SIZE,
        )

    def _reset_prediction_diagnostics(self) -> None:
        """Reset accumulated prediction diagnostics before a new scoring run."""
        self._prediction_diagnostics_state.reset()

    def _reset_next_event_prediction_state(self) -> None:
        """Reset accumulated next-event diagnostics before a new scoring run."""
        self._next_event_prediction_state = NextEventPredictionState.create(
            k_values=_next_event_k_values(),
            vocabulary_policy=self.config.vocabulary_policy,
        )

    def _next_event_prediction_state_snapshot(
        self,
    ) -> NextEventPredictionDiagnostics | None:
        """Return the current next-event diagnostics without clearing them.

        Returns:
            NextEventPredictionDiagnostics | None: Latest next-event
            diagnostics, or `None` when no eligible events were observed.
        """
        state = self._next_event_prediction_state
        if state is None:
            return None
        return state.snapshot()

    def _record_next_event_predictions(
        self,
        *,
        sequences: Sequence[TemplateSequence],
        batch: DeepCaseSampleBatch,
    ) -> None:
        """Record deterministic Context Builder diagnostics for one scoring run.

        Args:
            sequences (Sequence[TemplateSequence]): Sequences being scored in
                this run.
            batch (DeepCaseSampleBatch): Materialised event-centred batch for
                the same sequences.

        Raises:
            ValueError: If the detector has not been fit or the batch cannot
                be mapped back to train vocabulary entries.
        """
        state = self._ensure_next_event_prediction_state()
        confidence = _normalise_next_event_confidence(
            self._predict_next_event_batch(batch),
        )
        sample_offset = 0
        event_id_map = self.event_id_map
        if event_id_map is None:
            msg = "deepcase must be fit before prediction."
            raise ValueError(msg)
        for sequence in sequences:
            for batch_sample_offset, event_index in enumerate(
                _prediction_sample_indexes(sequence),
            ):
                batch_index = sample_offset + batch_sample_offset
                template = sequence.templates[event_index]
                state.record_observation(
                    actual_label=template,
                    predicted_labels=_top_event_templates(
                        confidence[batch_index],
                        event_id_map.event_id_to_template,
                        k=max(state.k_values),
                    ),
                    target_is_known=batch.original_event_ids[batch_index] is not None,
                    history_is_known=not any(
                        context_event_id is None
                        for context_event_id in batch.context_original_event_ids[
                            batch_index
                        ]
                    ),
                )
            sample_offset += _prediction_sample_count(sequence)

    def _predict_next_event_batch(self, batch: DeepCaseSampleBatch) -> torch.Tensor:
        """Run the diagnostic-only Context Builder next-event pass.

        This intentionally performs a separate deterministic prediction pass
        from the interpreter-based anomaly path so the anomaly logic remains
        unchanged.
        Upstream ContextBuilder.predict returns confidence tensors of shape
        ``(n_samples, steps, output_size)``; with ``steps=1`` the final step
        slice is the next-event distribution.

        Args:
            batch (DeepCaseSampleBatch): Bounded event-centred sample batch.

        Returns:
            torch.Tensor: Context Builder confidence tensor for the batch.

        Raises:
            ValueError: If the detector has not been fit.
        """
        model = self.model
        if model is None:
            msg = "deepcase must be fit before prediction."
            raise ValueError(msg)
        return _predict_next_event_batch_in_chunks(
            model=model,
            batch=batch,
            device=self.device,
            chunk_size=DEEPCASE_PREDICTION_SAMPLE_CHUNK_SIZE,
        )

    def _ensure_next_event_prediction_state(self) -> NextEventPredictionState:
        state = self._next_event_prediction_state
        if state is None:
            state = NextEventPredictionState.create(
                k_values=_next_event_k_values(),
                vocabulary_policy=self.config.vocabulary_policy,
            )
            self._next_event_prediction_state = state
        return state


@dataclass(frozen=True, slots=True)
class _ClusteredTrainingSamples:
    clusters: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True, slots=True)
class _AttendedTrainingChunk:
    vectors: sp.csc_matrix
    sample_indexes: np.ndarray


@dataclass(frozen=True, slots=True)
class _CachedContextBuilderTrainingBatch:
    """Cached inputs for one DeepCASE context-builder training chunk.

    Attributes:
        contexts (np.ndarray): Cached context window matrix for one chunk.
        events (np.ndarray): Cached target-event ids aligned with contexts.
    """

    contexts: np.ndarray
    events: np.ndarray


def _training_sample_count(sequence: TemplateSequence) -> int:
    if sequence.training_event_mask is not None:
        return sum(1 for is_eligible in sequence.training_event_mask if is_eligible)
    return len(sequence.events)


def _training_sample_indexes(sequence: TemplateSequence) -> list[int]:
    if sequence.training_event_mask is None:
        return list(range(len(sequence.events)))
    return [
        event_index
        for event_index, is_eligible in enumerate(sequence.training_event_mask)
        if is_eligible
    ]


def _training_sample_count_total(sequences: Sequence[TemplateSequence]) -> int:
    return sum(_training_sample_count(sequence) for sequence in sequences)


def _sample_mask(
    *,
    sequence_length: int,
    sample_indexes: Sequence[int],
) -> tuple[bool, ...]:
    mask = [False] * sequence_length
    for sample_index in sample_indexes:
        mask[sample_index] = True
    return tuple(mask)


def _chunk_training_sequences_by_sample_count(
    sequences: Iterable[TemplateSequence],
    *,
    max_sample_count: int,
) -> Iterator[list[TemplateSequence]]:
    chunk: list[TemplateSequence] = []
    chunk_sample_count = 0
    for sequence in sequences:
        sample_indexes = _training_sample_indexes(sequence)
        if not sample_indexes:
            continue
        start = 0
        while start < len(sample_indexes):
            if chunk_sample_count == max_sample_count:
                yield chunk
                chunk = []
                chunk_sample_count = 0
            remaining_capacity = max_sample_count - chunk_sample_count
            remaining_sample_count = len(sample_indexes) - start
            if start == 0 and remaining_sample_count <= remaining_capacity:
                chunk.append(sequence)
                chunk_sample_count += remaining_sample_count
                break
            take = min(remaining_capacity, remaining_sample_count)
            chunk.append(
                replace(
                    sequence,
                    training_event_mask=_sample_mask(
                        sequence_length=len(sequence.events),
                        sample_indexes=sample_indexes[start : start + take],
                    ),
                ),
            )
            chunk_sample_count += take
            start += take
            if chunk_sample_count == max_sample_count:
                yield chunk
                chunk = []
                chunk_sample_count = 0
    if chunk:
        yield chunk


def _chunk_prediction_sequences_by_sample_count(
    sequences: Iterable[TemplateSequence],
    *,
    max_sample_count: int,
) -> Iterator[list[TemplateSequence]]:
    chunk: list[TemplateSequence] = []
    chunk_sample_count = 0
    for sequence in sequences:
        sample_indexes = _prediction_sample_indexes(sequence)
        if not sample_indexes:
            continue
        start = 0
        while start < len(sample_indexes):
            if chunk_sample_count == max_sample_count:
                yield chunk
                chunk = []
                chunk_sample_count = 0
            remaining_capacity = max_sample_count - chunk_sample_count
            remaining_sample_count = len(sample_indexes) - start
            if start == 0 and remaining_sample_count <= remaining_capacity:
                chunk.append(sequence)
                chunk_sample_count += remaining_sample_count
                break
            take = min(remaining_capacity, remaining_sample_count)
            chunk.append(
                replace(
                    sequence,
                    evaluation_event_mask=_sample_mask(
                        sequence_length=len(sequence.events),
                        sample_indexes=sample_indexes[start : start + take],
                    ),
                ),
            )
            chunk_sample_count += take
            start += take
            if chunk_sample_count == max_sample_count:
                yield chunk
                chunk = []
                chunk_sample_count = 0
    if chunk:
        yield chunk


def _fit_context_builder_in_chunks(
    *,
    model: DeepCASE,
    train_sequences: Sequence[TemplateSequence],
    event_id_map: DeepCaseEventIdMap,
    config: DeepCaseModelConfig,
) -> None:
    cached_batches = _materialise_context_builder_training_batches(
        train_sequences=train_sequences,
        event_id_map=event_id_map,
        context_length=config.context_length,
        timeout_seconds=config.timeout_seconds,
    )
    for _ in range(config.epochs):
        for batch in cached_batches:
            model.context_builder.fit(
                X=batch.contexts,
                y=batch.events.reshape(-1, 1),
                epochs=1,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                teach_ratio=config.teach_ratio,
                delta=config.label_smoothing_delta,
                verbose=False,
            )


def _materialise_context_builder_training_batches(
    *,
    train_sequences: Sequence[TemplateSequence],
    event_id_map: DeepCaseEventIdMap,
    context_length: int,
    timeout_seconds: float,
) -> list[_CachedContextBuilderTrainingBatch]:
    """Return the reusable training chunks for the DeepCASE context builder.

    The context windows are deterministic, so we build each chunk once and
    replay those arrays across epochs instead of reconstructing the same
    samples repeatedly.

    Args:
        train_sequences (Sequence[TemplateSequence]): Training sequences to
            chunk and materialise.
        event_id_map (DeepCaseEventIdMap): Train-time event vocabulary map.
        context_length (int): Number of prior events in each context window.
        timeout_seconds (float): Maximum time gap between context events and
            the target event.

    Returns:
        list[_CachedContextBuilderTrainingBatch]: Cached context-builder input
        chunks ready for replay across epochs.
    """
    cached_batches: list[_CachedContextBuilderTrainingBatch] = []
    for chunk in _chunk_training_sequences_by_sample_count(
        train_sequences,
        max_sample_count=DEEPCASE_TRAINING_SAMPLE_CHUNK_SIZE,
    ):
        batch = build_training_batch_from_map(
            chunk,
            event_id_map=event_id_map,
            context_length=context_length,
            timeout_seconds=timeout_seconds,
        )
        if batch.sample_count == 0:
            continue
        cached_batches.append(
            _CachedContextBuilderTrainingBatch(
                contexts=batch.contexts,
                events=batch.events,
            ),
        )
    return cached_batches


def _cluster_training_samples(
    *,
    model: DeepCASE,
    train_sequences: Sequence[TemplateSequence],
    event_id_map: DeepCaseEventIdMap,
    config: DeepCaseModelConfig,
    device: torch.device,
) -> _ClusteredTrainingSamples:
    total_sample_count = _training_sample_count_total(train_sequences)
    clusters = np.full(total_sample_count, -1, dtype=int)
    labels = np.empty(total_sample_count, dtype=float)
    cluster_scores = np.full(total_sample_count, float(config.no_score), dtype=float)
    state = _TrainingClusterState(
        interpreter=model.interpreter,
        labels=labels,
        clusters=clusters,
        cluster_scores=cluster_scores,
        config=config,
    )
    event_vectors: dict[int, list[sp.csc_matrix]] = defaultdict(list)
    event_sample_indexes: dict[int, list[np.ndarray]] = defaultdict(list)
    sample_offset = 0
    for chunk in _chunk_training_sequences_by_sample_count(
        train_sequences,
        max_sample_count=DEEPCASE_TRAINING_SAMPLE_CHUNK_SIZE,
    ):
        batch = build_training_batch_from_map(
            chunk,
            event_id_map=event_id_map,
            context_length=config.context_length,
            timeout_seconds=config.timeout_seconds,
        )
        if batch.sample_count == 0:
            continue
        labels[sample_offset : sample_offset + batch.sample_count] = batch.scores
        attended_chunk = _attend_training_chunk(
            model=model,
            batch=batch,
            config=config,
            device=device,
            sample_offset=sample_offset,
        )
        if attended_chunk is not None:
            event_ids = batch.events[attended_chunk.sample_indexes - sample_offset]
            for event in np.unique(event_ids):
                event_mask = event_ids == event
                event_vectors[event].append(attended_chunk.vectors[event_mask])
                event_sample_indexes[event].append(
                    attended_chunk.sample_indexes[event_mask],
                )
        sample_offset += batch.sample_count

    if not event_vectors:
        return _ClusteredTrainingSamples(
            clusters=clusters,
            labels=labels,
            scores=cluster_scores,
        )

    cluster_offset = 0
    for event in sorted(event_vectors):
        cluster_offset = _cluster_and_score_training_event(
            state=state,
            event=event,
            vectors=sp.vstack(event_vectors[event], format="csc"),
            sample_indexes=np.concatenate(event_sample_indexes[event]),
            cluster_offset=cluster_offset,
        )
        del event_vectors[event]
        del event_sample_indexes[event]

    return _ClusteredTrainingSamples(
        clusters=clusters,
        labels=labels,
        scores=cluster_scores,
    )


def _attend_training_chunk(
    *,
    model: DeepCASE,
    batch: DeepCaseSampleBatch,
    config: DeepCaseModelConfig,
    device: torch.device,
    sample_offset: int,
) -> _AttendedTrainingChunk | None:
    chunk_contexts = torch.as_tensor(
        batch.contexts,
        dtype=torch.int64,
        device=device,
    )
    chunk_events = torch.as_tensor(
        batch.events.reshape(-1, 1),
        dtype=torch.int64,
        device=device,
    )
    vectors, mask = model.interpreter.attended_context(
        X=chunk_contexts,
        y=chunk_events,
        threshold=config.confidence_threshold,
        iterations=config.iterations,
        batch_size=config.query_batch_size,
        verbose=False,
    )
    mask_array = mask.detach().cpu().numpy()
    if not np.any(mask_array):
        return None
    return _AttendedTrainingChunk(
        vectors=vectors,
        sample_indexes=np.arange(
            sample_offset,
            sample_offset + batch.sample_count,
            dtype=int,
        )[mask_array],
    )


def _score_interpreter_event(
    *,
    interpreter: Interpreter,
    event: int,
    vectors: sp.csc_matrix,
    scores: np.ndarray,
) -> None:
    """Build DeepCASE lookup state for one clustered event.

    This mirrors ``Interpreter.score`` but only for a single event so fit does
    not need to materialise the entire attended-vector matrix at once.

    Args:
        interpreter (Interpreter): Upstream DeepCASE interpreter to populate.
        event (int): Event identifier for the clustered vectors.
        vectors (sp.csc_matrix): Sparse attended vectors for the event.
        scores (np.ndarray): Per-sample scores aligned with ``vectors``.

    Raises:
        RuntimeError: If the KDTree contents do not match the unique sparse
            vectors used to build it.
    """
    if vectors.shape[0] == 0:
        return

    vectors, inverse, _ = sp_unique(vectors)
    interpreter.tree[event] = KDTree(vectors.toarray(), p=1)
    interpreter.labels[event] = {}
    data, index_tree, _, _ = interpreter.tree[event].get_arrays()
    _, index_vector = zip(*group_by(inverse), strict=True)
    if not np.all(data == vectors.toarray()):
        msg = "DeepCase interpreter tree vectors did not round-trip as expected."
        raise RuntimeError(msg)

    for index, mapping in zip(index_tree, index_vector, strict=True):
        interpreter.labels[event][index] = scores[mapping].max()


@dataclass(slots=True)
class _TrainingClusterState:
    interpreter: Interpreter
    labels: np.ndarray
    clusters: np.ndarray
    cluster_scores: np.ndarray
    config: DeepCaseModelConfig


def _cluster_and_score_training_event(
    *,
    state: _TrainingClusterState,
    event: int,
    vectors: sp.csc_matrix,
    sample_indexes: np.ndarray,
    cluster_offset: int,
) -> int:
    """Cluster one event's attended training vectors and build lookup state.

    Args:
        state (_TrainingClusterState): Mutable shared fit state.
        event (int): Event identifier for the current sparse vector batch.
        vectors (sp.csc_matrix): Sparse attended vectors for this event.
        sample_indexes (np.ndarray): Global sample indexes for ``vectors``.
        cluster_offset (int): Running offset for non-noise cluster ids.

    Returns:
        int: Updated global cluster offset after the event's non-noise clusters
        have been assigned.
    """
    event_clusters = state.interpreter.dbscan.dbscan(
        X=vectors,
        eps=state.config.eps,
        min_samples=state.config.min_samples,
        verbose=False,
    )
    non_noise = event_clusters != -1
    if np.any(non_noise):
        event_clusters[non_noise] += cluster_offset
        cluster_offset = int(event_clusters[non_noise].max()) + 1
    state.clusters[sample_indexes] = event_clusters
    event_scores = cluster_scores_for_labels(
        event_clusters,
        state.labels[sample_indexes],
        strategy=state.config.cluster_score_strategy,
        anomaly_fraction_threshold=state.config.cluster_anomaly_fraction_threshold,
        no_score=state.config.no_score,
    )
    state.cluster_scores[sample_indexes] = event_scores
    _score_interpreter_event(
        interpreter=state.interpreter,
        event=event,
        vectors=vectors[event_clusters != -1],
        scores=event_scores[event_clusters != -1],
    )
    return cluster_offset


def _predict_batch_in_chunks(
    *,
    model: DeepCASE,
    batch: DeepCaseSampleBatch,
    device: torch.device,
    chunk_size: int,
    config: DeepCaseModelConfig,
) -> list[float]:
    raw_scores: list[float] = []
    for start in range(0, batch.sample_count, chunk_size):
        end = min(start + chunk_size, batch.sample_count)
        chunk_predictions = model.predict(
            X=torch.as_tensor(
                batch.contexts[start:end],
                dtype=torch.int64,
                device=device,
            ),
            y=torch.as_tensor(
                batch.events[start:end].reshape(-1, 1),
                dtype=torch.int64,
                device=device,
            ),
            iterations=config.attention_query_iterations,
            batch_size=config.query_batch_size,
            verbose=False,
        )
        raw_scores.extend(float(score) for score in chunk_predictions)
    return raw_scores


def _predict_next_event_batch_in_chunks(
    *,
    model: DeepCASE,
    batch: DeepCaseSampleBatch,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    confidence_chunks: list[torch.Tensor] = []
    for start in range(0, batch.sample_count, chunk_size):
        end = min(start + chunk_size, batch.sample_count)
        confidence, _ = model.context_builder.predict(
            X=torch.as_tensor(
                batch.contexts[start:end],
                dtype=torch.int64,
                device=device,
            ),
            steps=1,
        )
        confidence_chunks.append(confidence)
    if not confidence_chunks:
        output_size = model.context_builder.decoder_event.out.out_features
        return torch.empty((0, 1, output_size), device=device)
    return torch.cat(confidence_chunks, dim=0)


def _findings_from_scores(
    *,
    batch: DeepCaseSampleBatch,
    raw_scores: Sequence[float],
    start_index: int = 0,
) -> list[DeepCaseEventFinding]:
    findings: list[DeepCaseEventFinding] = []
    for relative_index, raw_score in enumerate(raw_scores):
        index = start_index + relative_index
        reason = finding_reason_for_score(raw_score)
        findings.append(
            DeepCaseEventFinding(
                event_index=batch.event_indexes[index],
                template=batch.templates[index],
                event_id=batch.original_event_ids[index],
                raw_score=raw_score,
                reason=reason,
                predicted_label=decision_label_for_score(raw_score),
                is_abstained=reason.is_abstained,
            ),
        )
    return findings


def _summarise_findings(
    *,
    findings: Sequence[DeepCaseEventFinding],
    raw_scores: Sequence[float],
) -> _DeepCasePredictionSummary:
    """Summarise event findings into one sequence-level DeepCase decision.

    This mirrors the paper's semi-automatic flow: confident positive scores
    are treated as anomaly candidates, while low-confidence or out-of-vocabulary
    cases are preserved as abstentions for manual inspection.

    Args:
        findings (Sequence[DeepCaseEventFinding]): Event-level findings for
            one scored sequence.
        raw_scores (Sequence[float]): Raw DeepCASE event scores for the same
            sequence.

    Returns:
        _DeepCasePredictionSummary: Sequence-level binary decision together
        with DeepCASE abstain and confidence counts.
    """
    abstained_event_count = 0
    confident_anomaly_event_count = 0
    for finding in findings:
        if finding.is_abstained:
            abstained_event_count += 1
        elif finding.predicted_label == 1:
            confident_anomaly_event_count += 1
    confident_event_count = len(findings) - abstained_event_count
    if confident_anomaly_event_count > 0:
        sequence_decision = DeepCaseSequenceDecision.CONFIDENT_ANOMALY
        predicted_label = 1
    elif abstained_event_count > 0:
        sequence_decision = DeepCaseSequenceDecision.ABSTAINED
        predicted_label = 0
    else:
        sequence_decision = DeepCaseSequenceDecision.CONFIDENT_NORMAL
        predicted_label = 0
    return _DeepCasePredictionSummary(
        predicted_label=predicted_label,
        score=aggregate_sequence_score(raw_scores),
        sequence_decision=sequence_decision,
        confident_event_count=confident_event_count,
        abstained_event_count=abstained_event_count,
        confident_anomaly_event_count=confident_anomaly_event_count,
    )


def _normalise_next_event_confidence(confidence: torch.Tensor) -> torch.Tensor:
    """Return a 2D next-event confidence tensor.

    Args:
        confidence (torch.Tensor): Raw context-builder confidence tensor.

    Returns:
        torch.Tensor: Two-dimensional confidence tensor with one row per
            scored event.

    Raises:
        ValueError: If the DeepCASE context builder returns an unexpected tensor
            rank.
    """
    if confidence.ndim == _NEXT_EVENT_CONFIDENCE_THREE_D:
        return confidence[:, -1, :]
    if confidence.ndim == _NEXT_EVENT_CONFIDENCE_TWO_D:
        return confidence
    msg = f"Unexpected DeepCase confidence shape: {tuple(confidence.shape)}"
    raise ValueError(msg)


def _prediction_sample_count(sequence: TemplateSequence) -> int:
    """Return how many prediction samples one sequence contributes.

    DeepCASE prediction batches may exclude some events via
    ``evaluation_event_mask`` while still keeping the parent sequence intact.
    Score slicing must follow the emitted sample count rather than the raw
    event count; otherwise later sequences consume the wrong score segment.

    Args:
        sequence (TemplateSequence): Sequence being scored.

    Returns:
        int: Number of event-centred samples emitted for prediction.
    """
    if sequence.evaluation_event_mask is None:
        return len(sequence.events)
    return sum(1 for is_eligible in sequence.evaluation_event_mask if is_eligible)


def _prediction_sample_indexes(sequence: TemplateSequence) -> list[int]:
    """Return prediction-time event indexes for one sequence.

    Args:
        sequence (TemplateSequence): Sequence being scored.

    Returns:
        list[int]: Event indexes emitted by DeepCASE prediction batching.
    """
    if sequence.evaluation_event_mask is None:
        return list(range(len(sequence.events)))
    return [
        event_index
        for event_index, is_eligible in enumerate(sequence.evaluation_event_mask)
        if is_eligible
    ]


def _top_event_templates(
    confidence: torch.Tensor,
    event_id_to_template: dict[int, str],
    *,
    k: int,
) -> list[str]:
    """Return the top predicted next-event templates for one sample.

    Args:
        confidence (torch.Tensor): One sample's next-event confidence vector.
        event_id_to_template (dict[int, str]): Mapping from event ids to
            training templates.
        k (int): Number of candidate templates to return.

    Returns:
        list[str]: Ranked predicted templates, highest confidence first.
    """
    top_k = min(k, confidence.shape[-1])
    if top_k == 0:
        return []
    _, top_indexes = torch.topk(confidence, k=top_k)
    return [event_id_to_template[int(index)] for index in top_indexes.tolist()]


def _next_event_k_values() -> tuple[int, ...]:
    """Return the DeepCASE next-event reporting cut-offs.

    Returns:
        tuple[int, ...]: Standard top-k cut-offs for DeepCASE diagnostics.
    """
    return (1, 2, 3, 5)


_NEXT_EVENT_CONFIDENCE_THREE_D = 3
_NEXT_EVENT_CONFIDENCE_TWO_D = 2
