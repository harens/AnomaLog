"""DeepCase detector integration for AnomaLog experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

import msgspec
import numpy as np
import torch
from deepcase import DeepCASE
from typing_extensions import override

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
    DeepCaseEventFinding,
    DeepCaseEventIdMap,
    DeepCaseManifest,
    DeepCasePredictionDiagnostics,
    DeepCaseSequenceDecision,
    _DeepCasePredictionDiagnosticsState,
    _DeepCasePredictionSummary,
    aggregate_sequence_score,
    build_sample_batch,
    build_training_batch,
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

if TYPE_CHECKING:
    import logging
    from collections.abc import Iterable, Iterator, Sequence

    from rich.progress import Progress

    from anomalog.sequences import TemplateSequence
    from experiments.models.deepcase.shared import DeepCaseSampleBatch

DeepCaseClusterScoreStrategy = Literal["max", "min", "avg"]
"""Defined within DeepCase Library's fit method
for how event scores are aggregated within a cluster."""


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
        auto_coverage (float): Fraction of test sequences handled
            automatically.
        abstain_rate (float): Fraction of test sequences deferred for
            review.
        prediction_diagnostics (DeepCasePredictionDiagnostics | None): Event
            and sequence diagnostics for the latest DeepCASE scoring run.
        next_event_prediction (NextEventPredictionDiagnostics | None): Latest
            Context Builder next-event diagnostics.
    """

    auto_decision_count: int
    abstained_prediction_count: int
    abstained_anomalous_label_count: int
    abstained_normal_label_count: int
    auto_coverage: float
    abstain_rate: float
    prediction_diagnostics: DeepCasePredictionDiagnostics | None
    next_event_prediction: NextEventPredictionDiagnostics | None = None


class DeepCaseModelConfig(
    ExperimentModelConfig,
    tag="deepcase",
    frozen=True,
):
    """Configuration for the DeepCASE experiment detector.

    Parameters are grouped by subsystem from the original paper:
    sequencing events, context builder, interpreter, and torch runtime.

    """  # noqa: DOC601 DOC603: attribute docs live in Annotated metadata.

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
        msgspec.Meta(description="Training epochs for the context builder."),
    ] = 10
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
                "DeepCase interpreter clusters and during prediction-time "
                "attention queries."
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
        msgspec.Meta(description="How event scores are aggregated within a cluster."),
    ] = "max"
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
    _prediction_sample_chunk_size: ClassVar[int] = 32_768

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
        event_id_map, batch = build_training_batch(
            train_sequences,
            context_length=self.config.context_length,
            timeout_seconds=self.config.timeout_seconds,
            progress=progress,
        )
        if batch.sample_count == 0:
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
        contexts = batch.contexts.to(device)
        events = batch.events.reshape(-1, 1).to(device)
        fit_task = progress.add_task(
            "DeepCase: training context builder",
            total=self.config.epochs + 1,
        )
        if logger is not None:
            logger.info("Training DeepCase context builder")
        for _ in range(self.config.epochs):
            model.context_builder.fit(
                X=contexts,
                y=events,
                epochs=1,
                batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                teach_ratio=self.config.teach_ratio,
                delta=self.config.label_smoothing_delta,
                verbose=False,
            )
            progress.advance(fit_task)
        progress.update(fit_task, description="DeepCase: clustering interpreter")
        if logger is not None:
            logger.info("Clustering DeepCase interpreter")
        model.interpreter.fit(
            X=contexts,
            y=events,
            scores=batch.scores,
            iterations=self.config.iterations,
            batch_size=self.config.query_batch_size,
            strategy=self.config.cluster_score_strategy,
            NO_SCORE=self.config.no_score,
            verbose=False,
        )
        progress.advance(fit_task)
        progress.update(fit_task, description="DeepCase: fit complete")

        self.model = model
        self.event_id_map = event_id_map
        self.device = device
        self.train_sample_count = batch.sample_count
        self.clustered_sample_count = int(
            np.count_nonzero(model.interpreter.clusters != -1),
        )
        self.known_cluster_count = len(
            {cluster for cluster in model.interpreter.clusters if cluster != -1},
        )
        scored_clusters = model.interpreter.score_clusters(
            scores=batch.scores,
            strategy=self.config.cluster_score_strategy,
            NO_SCORE=self.config.no_score,
        )
        self.known_benign_cluster_count = int(
            np.count_nonzero(scored_clusters == 0),
        )
        self.known_malicious_cluster_count = int(
            np.count_nonzero(scored_clusters > 0),
        )
        self.unknown_cluster_score_count = int(
            np.count_nonzero(scored_clusters == self.config.no_score),
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
        if batch.sample_count == 0:
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
        buffered_sequences: list[TemplateSequence] = []
        buffered_sample_count = 0
        for sequence in sequences:
            buffered_sequences.append(sequence)
            buffered_sample_count += len(sequence.events)
            if buffered_sample_count < self._prediction_chunk_sample_limit():
                continue
            yield from self._predict_sequence_chunk(buffered_sequences)
            buffered_sequences = []
            buffered_sample_count = 0
        if buffered_sequences:
            yield from self._predict_sequence_chunk(buffered_sequences)

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
                "sequence-label supervision: every event-centered sample inherits "
                "the source TemplateSequence label"
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
            query_batch_size=self.config.query_batch_size,
            cluster_score_strategy=self.config.cluster_score_strategy,
            no_score=self.config.no_score,
            device=str(self.device),
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
            auto_coverage=round(auto_coverage, 8),
            abstain_rate=round(abstain_rate, 8),
            prediction_diagnostics=prediction_diagnostics,
            next_event_prediction=next_event_prediction,
        )

    def _predict_batch(self, batch: DeepCaseSampleBatch) -> list[float]:
        model = self.model
        if model is None:
            msg = "deepcase must be fit before prediction."
            raise ValueError(msg)
        raw_predictions = model.predict(
            X=batch.contexts.to(self.device),
            y=batch.events.reshape(-1, 1).to(self.device),
            iterations=self.config.iterations,
            batch_size=self.config.query_batch_size,
            verbose=False,
        )
        return [float(score) for score in raw_predictions]

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
        if batch.sample_count == 0:
            for sequence in sequences:
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
            sample_count = len(sequence.events)
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
        return max(self.config.query_batch_size, self._prediction_sample_chunk_size)

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
            for event_index, template in enumerate(sequence.templates):
                batch_index = sample_offset + event_index
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
            sample_offset += len(sequence.events)

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
        confidence, _ = model.context_builder.predict(
            X=batch.contexts.to(self.device),
            steps=1,
        )
        return confidence

    def _ensure_next_event_prediction_state(self) -> NextEventPredictionState:
        state = self._next_event_prediction_state
        if state is None:
            state = NextEventPredictionState.create(
                k_values=_next_event_k_values(),
                vocabulary_policy=self.config.vocabulary_policy,
            )
            self._next_event_prediction_state = state
        return state


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
        sequence_decision: Literal[DeepCaseSequenceDecision.CONFIDENT_ANOMALY] = (
            DeepCaseSequenceDecision.CONFIDENT_ANOMALY
        )
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
