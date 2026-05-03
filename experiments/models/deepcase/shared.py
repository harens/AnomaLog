"""Shared DeepCase runtime structs and sample helpers.

This module adapts AnomaLog ``TemplateSequence`` objects into the tensor inputs
expected by the upstream DeepCASE library while preserving enough metadata to
explain event-level outcomes back in AnomaLog terms.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sized
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import msgspec
import numpy as np
import torch

from anomalog.parsers.structured.contracts import is_anomalous_label
from experiments.models.base import ModelManifest, require_entity_local_sequences

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from rich.progress import Progress

    from anomalog.sequences import TemplateSequence
DEEPCASE_NO_EVENT = -1337
"""Upstream DeepCASE sentinel for missing or timed-out context events."""


class ScoreReason(str, Enum):
    """Stable serialised reason labels for DeepCase event findings.

    Attributes:
        KNOWN_BENIGN_CLUSTER: Event matched a known benign cluster.
        KNOWN_MALICIOUS_CLUSTER: Event matched a known malicious cluster.
        NOT_CONFIDENT_ENOUGH: Interpreter confidence was too low to trust the
            cluster correction, so the sample should be treated as abstained
            rather than anomalous.
        EVENT_NOT_IN_TRAINING_VOCABULARY: Prediction-time event template was
            unseen during training.
        CLOSEST_CLUSTER_OUTSIDE_EPSILON: Nearest cluster was outside the
            interpreter epsilon threshold, so the sample should be treated as
            abstained rather than anomalous.
    """

    KNOWN_BENIGN_CLUSTER = "known_benign_cluster"
    KNOWN_MALICIOUS_CLUSTER = "known_malicious_cluster"
    NOT_CONFIDENT_ENOUGH = "not_confident_enough"
    EVENT_NOT_IN_TRAINING_VOCABULARY = "event_not_in_training_vocabulary"
    CLOSEST_CLUSTER_OUTSIDE_EPSILON = "closest_cluster_outside_epsilon"

    @property
    def is_abstained(self) -> bool:
        """Return whether this score reason should abstain from automation."""
        return self not in {
            ScoreReason.KNOWN_BENIGN_CLUSTER,
            ScoreReason.KNOWN_MALICIOUS_CLUSTER,
        }


class DeepCaseSequenceDecision(str, Enum):
    """Stable sequence-level decision categories for DeepCASE.

    Attributes:
        CONFIDENT_NORMAL: All decisive event findings were normal.
        ABSTAINED: At least one event required manual review and none were
            confidently anomalous.
        CONFIDENT_ANOMALY: At least one event was confidently anomalous.
    """

    CONFIDENT_NORMAL = "confident_normal"
    ABSTAINED = "abstained"
    CONFIDENT_ANOMALY = "confident_anomaly"


# Sentinel scores emitted by the DeepCase library for non-cluster outcomes.
SPECIAL_SCORE_REASONS: dict[float, ScoreReason] = {
    -1.0: ScoreReason.NOT_CONFIDENT_ENOUGH,
    -2.0: ScoreReason.EVENT_NOT_IN_TRAINING_VOCABULARY,
    -3.0: ScoreReason.CLOSEST_CLUSTER_OUTSIDE_EPSILON,
}


@dataclass(frozen=True, slots=True)
class DeepCaseSampleBatch:
    """Tensor inputs and aligned metadata for DeepCase event-centered samples.

    DeepCASE trains and predicts over individual target events paired with a
    fixed-width context window. This batch keeps those tensor inputs together
    with the event-level metadata needed to map predictions back onto an
    AnomaLog ``TemplateSequence``.

    Attributes:
        contexts (torch.Tensor): Integer context windows of shape
            ``(sample_count, context_length)``.
        context_original_event_ids (list[list[int | None]]): Original
            train-vocabulary ids for each context slot, with ``None`` used for
            padding slots and unknown prediction-time templates.
        events (torch.Tensor): Integer target event ids of shape
            ``(sample_count,)``.
        scores (np.ndarray): Per-sample labels or scores aligned with
            ``contexts`` and ``events``.
        event_indexes (list[int]): Index of each target event within its source
            sequence.
        templates (list[str]): Original target event templates.
        original_event_ids (list[int | None]): Train-vocabulary event ids for
            the target templates. Prediction-time unknown templates remain
            ``None`` because they were never assigned a train id, which keeps
            unseen templates distinct from known templates that happen to map
            to the unknown-event sentinel during scoring.
        parent_sequence_fallback_count (int): Number of target events that had
            to fall back to the parent sequence label.
    """

    contexts: torch.Tensor
    context_original_event_ids: list[list[int | None]]
    events: torch.Tensor
    scores: np.ndarray
    event_indexes: list[int]
    templates: list[str]
    original_event_ids: list[int | None]
    parent_sequence_fallback_count: int

    @property
    def sample_count(self) -> int:
        """Return the number of event-centered samples in this batch."""
        return len(self.event_indexes)


@dataclass(frozen=True, slots=True)
class ContextWindowPolicy:
    """Shared context-window rules for turning sequences into DeepCase samples.

    Attributes:
        context_length (int): Number of prior events to retain for each target.
        timeout_ms (int): Maximum target-to-context age before a context event
            is replaced with the DeepCASE no-event sentinel.
        no_event_id (int): Event id written when a context slot has no valid
            preceding event.
    """

    context_length: int
    timeout_ms: int
    no_event_id: int


@dataclass(frozen=True, slots=True)
class DeepCaseEventIdMap:
    """Train-time mapping between templates and DeepCase event ids.

    DeepCASE expects small contiguous integer ids rather than the original
    template strings used throughout AnomaLog. This mapping is learned from the
    train split and then reused during prediction so previously unseen templates
    can be reported explicitly.

    Attributes:
        template_to_event_id (dict[str, int]): Mapping from train template to
            contiguous DeepCase event id.
        event_id_to_template (dict[int, str]): Reverse mapping used for
            manifest/reporting purposes, including the no-event sentinel id.
        no_event_id (int): Contiguous event id reserved for missing or stale
            context slots.
    """

    template_to_event_id: dict[str, int]
    """Mapping from event template to contiguous event id."""
    event_id_to_template: dict[int, str]
    """Mapping from contiguous event id to event template, including no-event id."""
    no_event_id: int
    """Contiguous event id used for no-event contexts, equal to vocabulary size."""

    @classmethod
    def from_sequences(
        cls,
        sequences: Iterable[TemplateSequence],
    ) -> DeepCaseEventIdMap:
        """Build a deterministic train-only event-id map from sequences.

        Args:
            sequences (Iterable[TemplateSequence]): Train sequences.

        Returns:
            DeepCaseEventIdMap: Contiguous event-id mappings plus NO_EVENT id.
        """
        templates = sorted(
            {template for sequence in sequences for template in sequence.templates},
        )
        template_to_event_id = {
            template: event_id for event_id, template in enumerate(templates)
        }
        no_event_id = len(template_to_event_id)
        event_id_to_template = {
            event_id: template for template, event_id in template_to_event_id.items()
        }
        event_id_to_template[no_event_id] = str(DEEPCASE_NO_EVENT)
        return cls(
            template_to_event_id=template_to_event_id,
            event_id_to_template=event_id_to_template,
            no_event_id=no_event_id,
        )


class DeepCaseEventFinding(msgspec.Struct, frozen=True):
    """Event-level DeepCase prediction finding.

    Attributes:
        event_index (int): Position of the target event within its source
            sequence.
        template (str): Original target event template string.
        event_id (int | None): Train-vocabulary event id for ``template``.
            Unknown prediction-time templates remain ``None``.
        raw_score (float): Raw DeepCASE event score or special code.
        reason (ScoreReason): Stable explanation label derived from
            ``raw_score``.
        predicted_label (int): Conservative AnomaLog binary label derived from
            ``raw_score``. Only confident positive scores count as anomalies.
        is_abstained (bool): Whether the event should be surfaced for manual
            review instead of being folded into the automated anomaly label.
    """

    event_index: int
    template: str
    event_id: int | None
    raw_score: float
    reason: ScoreReason
    predicted_label: int
    is_abstained: bool


class DeepCasePredictionDiagnostics(msgspec.Struct, frozen=True):
    """Serialisable DeepCASE prediction diagnostics for model manifests.

    Attributes:
        event_count (int): Total number of scored events.
        confident_event_count (int): Number of events with a decisive label.
        abstained_event_count (int): Number of events that required review.
        abstained_anomalous_label_count (int): Number of abstained
            predictions with anomalous ground-truth labels.
        abstained_normal_label_count (int): Number of abstained predictions
            with normal ground-truth labels.
        confident_anomaly_event_count (int): Number of decisive anomalous
            events.
        sequence_confident_anomaly_count (int): Number of sequences whose
            final decision was confident anomaly.
        sequence_confident_normal_count (int): Number of sequences whose
            final decision was confident normal.
        sequence_abstained_count (int): Number of sequences whose final
            decision was abstained.
        reason_counts (dict[str, int]): Histogram of event finding reasons.
    """

    event_count: int
    confident_event_count: int
    abstained_event_count: int
    abstained_anomalous_label_count: int
    abstained_normal_label_count: int
    confident_anomaly_event_count: int
    sequence_confident_anomaly_count: int
    sequence_confident_normal_count: int
    sequence_abstained_count: int
    reason_counts: dict[str, int]


@dataclass(frozen=True, slots=True)
class _DeepCasePredictionSummary:
    predicted_label: int
    score: float
    sequence_decision: DeepCaseSequenceDecision
    confident_event_count: int
    abstained_event_count: int
    confident_anomaly_event_count: int


@dataclass(slots=True)
class _DeepCasePredictionDiagnosticsState:
    event_count: int = 0
    confident_event_count: int = 0
    abstained_event_count: int = 0
    abstained_anomalous_label_count: int = 0
    abstained_normal_label_count: int = 0
    confident_anomaly_event_count: int = 0
    sequence_confident_anomaly_count: int = 0
    sequence_confident_normal_count: int = 0
    sequence_abstained_count: int = 0
    parent_sequence_fallback_count: int = 0
    reason_counts: Counter[str] = field(default_factory=Counter)

    def record(
        self,
        *,
        summary: _DeepCasePredictionSummary,
        findings: Sequence[DeepCaseEventFinding],
        sequence_label: int,
    ) -> None:
        self.event_count += len(findings)
        self.confident_event_count += summary.confident_event_count
        self.abstained_event_count += summary.abstained_event_count
        if summary.sequence_decision is DeepCaseSequenceDecision.ABSTAINED:
            if is_anomalous_label(sequence_label):
                self.abstained_anomalous_label_count += 1
            else:
                self.abstained_normal_label_count += 1
        self.confident_anomaly_event_count += summary.confident_anomaly_event_count
        if summary.sequence_decision is DeepCaseSequenceDecision.CONFIDENT_ANOMALY:
            self.sequence_confident_anomaly_count += 1
        elif summary.sequence_decision is DeepCaseSequenceDecision.CONFIDENT_NORMAL:
            self.sequence_confident_normal_count += 1
        else:
            self.sequence_abstained_count += 1
        self.reason_counts.update(finding.reason.value for finding in findings)

    def reset(self) -> None:
        self.event_count = 0
        self.confident_event_count = 0
        self.abstained_event_count = 0
        self.abstained_anomalous_label_count = 0
        self.abstained_normal_label_count = 0
        self.confident_anomaly_event_count = 0
        self.sequence_confident_anomaly_count = 0
        self.sequence_confident_normal_count = 0
        self.sequence_abstained_count = 0
        self.parent_sequence_fallback_count = 0
        self.reason_counts = Counter()

    def record_parent_sequence_fallback_count(self, count: int) -> None:
        """Record how many samples in the latest run used parent labels.

        Args:
            count (int): Number of samples that fell back to the parent
                sequence label.
        """
        self.parent_sequence_fallback_count += count

    def snapshot(self) -> DeepCasePredictionDiagnostics | None:
        if self.event_count == 0:
            return None
        return DeepCasePredictionDiagnostics(
            event_count=self.event_count,
            confident_event_count=self.confident_event_count,
            abstained_event_count=self.abstained_event_count,
            abstained_anomalous_label_count=self.abstained_anomalous_label_count,
            abstained_normal_label_count=self.abstained_normal_label_count,
            confident_anomaly_event_count=self.confident_anomaly_event_count,
            sequence_confident_anomaly_count=self.sequence_confident_anomaly_count,
            sequence_confident_normal_count=self.sequence_confident_normal_count,
            sequence_abstained_count=self.sequence_abstained_count,
            reason_counts=dict(self.reason_counts),
        )


class DeepCaseManifest(ModelManifest, frozen=True):
    """Serialisable manifest for DeepCase experiment runs.

    Attributes:
        implementation_scope (str): High-level description of the integrated
            DeepCASE implementation.
        label_policy (str): Summary of how sequence labels were projected onto
            event-centered samples.
        context_length (int): Configured context length.
        timeout_seconds (float): Configured context timeout in seconds.
        hidden_size (int): Context-builder hidden size.
        label_smoothing_delta (float): Label smoothing delta used in training.
        eps (float): Interpreter DBSCAN epsilon.
        min_samples (int): Interpreter minimum cluster size.
        confidence_threshold (float): Interpreter confidence threshold.
        epochs (int): Context-builder training epochs.
        batch_size (int): Context-builder training batch size.
        learning_rate (float): Context-builder optimizer learning rate.
        teach_ratio (float): Teacher-forcing ratio.
        iterations (int): Maximum interpreter query iterations used while
            building clusters and during prediction-time attention queries.
        query_batch_size (int): Batch size used during querying/prediction.
        cluster_score_strategy (str): Cluster score aggregation strategy.
        no_score (int): Special no-score value passed to DeepCASE.
        device (str): Resolved runtime torch device.
        train_event_vocabulary_size (int): Number of train templates mapped to
            contiguous DeepCASE ids.
        train_sample_count (int): Number of event-centered training samples.
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
        prediction_diagnostics (DeepCasePredictionDiagnostics | None): Optional
            prediction-time diagnostics aggregated from the latest DeepCASE
            scoring run.
        online_updates_status (str): Status of online model updates.
        persistent_cluster_database_status (str): Status of persistent cluster
            storage support.
    """

    implementation_scope: str
    label_policy: str
    context_length: int
    timeout_seconds: float
    hidden_size: int
    label_smoothing_delta: float
    eps: float
    min_samples: int
    confidence_threshold: float
    epochs: int
    batch_size: int
    learning_rate: float
    teach_ratio: float
    iterations: int
    query_batch_size: int
    cluster_score_strategy: str
    no_score: int
    device: str
    train_event_vocabulary_size: int
    train_sample_count: int
    clustered_sample_count: int
    known_cluster_count: int
    known_benign_cluster_count: int
    known_malicious_cluster_count: int
    unknown_cluster_score_count: int
    prediction_diagnostics: DeepCasePredictionDiagnostics | None
    online_updates_status: str
    persistent_cluster_database_status: str


def build_sample_batch(
    sequences: Iterable[TemplateSequence],
    *,
    event_id_map: DeepCaseEventIdMap,
    context_length: int,
    timeout_seconds: float,
    unknown_event_id: int | None = None,
) -> DeepCaseSampleBatch:
    """Build DeepCase event-centered samples from AnomaLog sequences.

    Args:
        sequences (Iterable[TemplateSequence]): Entity-local template sequences.
        event_id_map (DeepCaseEventIdMap): Train event-id mapping.
        context_length (int): Number of previous events per context window.
        timeout_seconds (float): Maximum target-to-context age in seconds.
        unknown_event_id (int | None): Event id used for unknown templates. If
            omitted, samples containing unknown templates are rejected.

    Returns:
        DeepCaseSampleBatch: Tensor inputs plus per-event metadata.

    """
    timeout_ms = int(timeout_seconds * 1000)
    context_policy = ContextWindowPolicy(
        context_length=context_length,
        timeout_ms=timeout_ms,
        no_event_id=event_id_map.no_event_id,
    )
    batch_columns = _EmptyBatchColumns.create()

    for sequence in sequences:
        require_entity_local_sequences((sequence,), detector_name="DeepCase")
        sequence_event_ids = [
            _event_id_for_template(
                template,
                event_id_map=event_id_map,
                unknown_event_id=unknown_event_id,
            )
            for template in sequence.templates
        ]
        _append_sequence_samples(
            sequence=sequence,
            sequence_event_ids=sequence_event_ids,
            original_event_id_lookup=event_id_map.template_to_event_id,
            context_policy=context_policy,
            batch_columns=batch_columns,
        )

    return batch_columns.to_batch(context_length=context_length)


def build_training_batch(
    sequences: Iterable[TemplateSequence],
    *,
    context_length: int,
    timeout_seconds: float,
    progress: Progress | None = None,
) -> tuple[DeepCaseEventIdMap, DeepCaseSampleBatch]:
    """Build a train batch and event-id map in a single pass over sequences.

    Training needs two artifacts derived from the same input stream: the
    contiguous train-time template mapping and the event-centered DeepCASE
    samples. Building them together avoids an eager full materialization of the
    original ``TemplateSequence`` iterable before tensor construction.

    Args:
        sequences (Iterable[TemplateSequence]): Entity-local training
            sequences.
        context_length (int): Number of previous events per context window.
        timeout_seconds (float): Maximum target-to-context age in seconds.
        progress (Progress | None): Optional progress reporter used to track
            training-sequence preparation.

    Returns:
        tuple[DeepCaseEventIdMap, DeepCaseSampleBatch]: Train event-id map and
        aligned tensor batch.
    """
    template_to_event_id: dict[str, int] = {}
    timeout_ms = int(timeout_seconds * 1000)
    context_policy = ContextWindowPolicy(
        context_length=context_length,
        timeout_ms=timeout_ms,
        no_event_id=DEEPCASE_NO_EVENT,
    )
    batch_columns = _EmptyBatchColumns.create()
    prepare_task: int | None = None
    if progress is not None:
        total = len(sequences) if isinstance(sequences, Sized) else None
        prepare_task = progress.add_task(
            "Preparing DeepCase training sequences",
            total=total,
        )

    try:
        for sequence in sequences:
            require_entity_local_sequences((sequence,), detector_name="DeepCase")
            sequence_event_ids = [
                template_to_event_id.setdefault(template, len(template_to_event_id))
                for template in sequence.templates
            ]
            _append_sequence_samples(
                sequence=sequence,
                sequence_event_ids=sequence_event_ids,
                original_event_id_lookup=template_to_event_id,
                context_policy=context_policy,
                batch_columns=batch_columns,
            )
            if progress is not None and prepare_task is not None:
                progress.advance(prepare_task)
    finally:
        if progress is not None and prepare_task is not None:
            progress.remove_task(prepare_task)

    no_event_id = len(template_to_event_id)
    event_id_to_template = {
        event_id: template for template, event_id in template_to_event_id.items()
    }
    event_id_to_template[no_event_id] = str(DEEPCASE_NO_EVENT)
    batch = batch_columns.to_batch(
        context_length=context_length,
        no_event_placeholder=DEEPCASE_NO_EVENT,
        no_event_id=no_event_id,
    )
    return (
        DeepCaseEventIdMap(
            template_to_event_id=template_to_event_id,
            event_id_to_template=event_id_to_template,
            no_event_id=no_event_id,
        ),
        batch,
    )


def finding_reason_for_score(raw_score: float) -> ScoreReason:
    """Return the DeepCase outcome reason for a raw score/code.

    Args:
        raw_score (float): Raw score returned by the official DeepCase library.

    Returns:
        ScoreReason: Stable reason label for serialization.

    Examples:
        >>> from experiments.models.deepcase.shared import (
        ...     ScoreReason,
        ...     finding_reason_for_score,
        ... )
        >>> finding_reason_for_score(0.0).value
        'known_benign_cluster'
        >>> finding_reason_for_score(-2.0) is (
        ...     ScoreReason.EVENT_NOT_IN_TRAINING_VOCABULARY
        ... )
        True
    """
    for special_score, reason in SPECIAL_SCORE_REASONS.items():
        if math.isclose(raw_score, special_score):
            return reason
    if raw_score > 0:
        return ScoreReason.KNOWN_MALICIOUS_CLUSTER
    return ScoreReason.KNOWN_BENIGN_CLUSTER


def decision_label_for_score(raw_score: float) -> int:
    """Map a raw DeepCase score/code into AnomaLog's binary anomaly label.

    Args:
        raw_score (float): Raw DeepCase score.

    Returns:
        int: `1` only for confident malicious scores, otherwise `0`.

    Examples:
        >>> from experiments.models.deepcase.shared import decision_label_for_score
        >>> decision_label_for_score(0.0)
        0
        >>> decision_label_for_score(-1.0)
        0
    """
    return int(raw_score > 0)


def aggregate_sequence_score(raw_scores: Sequence[float]) -> float:
    """Aggregate event-level DeepCase scores for one sequence.

    Args:
        raw_scores (Sequence[float]): Raw event scores for one sequence.

    Returns:
        float: Sequence-level score for the experiment metrics contract.
    """
    return max((score for score in raw_scores if score > 0), default=0.0)


def _event_id_for_template(
    template: str,
    *,
    event_id_map: DeepCaseEventIdMap,
    unknown_event_id: int | None,
) -> int:
    event_id = event_id_map.template_to_event_id.get(template)
    if event_id is not None:
        return event_id
    if unknown_event_id is not None:
        return unknown_event_id
    msg = f"Training sequence contains unknown template: {template!r}."
    raise ValueError(msg)


def _context_for_target(
    *,
    sequence: TemplateSequence,
    sequence_event_ids: Sequence[int],
    target_index: int,
    context_policy: ContextWindowPolicy,
    original_event_id_lookup: dict[str, int],
) -> tuple[list[int], list[int | None]]:
    """Return the encoded context window and original ids for one target event.

    The DeepCASE paper defines an event context as the preceding events from
    the same entity. This helper adapts AnomaLog's ``dt_prev_ms`` gaps into
    that window by replacing context events with ``no_event_id`` when the
    accumulated age between the candidate context event and the target exceeds
    ``timeout_ms``.

    Args:
        sequence (TemplateSequence): Source sequence containing event gap data.
        sequence_event_ids (Sequence[int]): Event ids aligned with
            ``sequence.templates``.
        target_index (int): Index of the event whose context is being built.
        context_policy (ContextWindowPolicy): Context length, timeout, and
            no-event policy to apply.
        original_event_id_lookup (dict[str, int]): Mapping used to recover
            train-vocabulary ids for the original context templates.

    Returns:
        tuple[list[int], list[int | None]]: Encoded context event ids and the
            corresponding original train-vocabulary ids for each slot.

    Examples:
        We have a sequence of three events "A", "B", and "C" with 1_000 ms between each.
        With a context length of 2 and timeout of 2_500 ms, "C" gets "B" and No ID
        as context because "A" is too old (3_000 ms > 2_500 ms).

        >>> from types import SimpleNamespace
        >>> sequence = SimpleNamespace(
        ...     events=[("A", [], None), ("B", [], 1_000), ("C", [], 2_000)],
        ...     templates=["A", "B", "C"],
        ... )
        >>> _context_for_target(
        ...     sequence=sequence,
        ...     sequence_event_ids=[10, 11, 12],
        ...     target_index=2,
        ...     context_policy=ContextWindowPolicy(
        ...         context_length=2,
        ...         timeout_ms=2_500,
        ...         no_event_id=99,
        ...     ),
        ...     original_event_id_lookup={"A": 10, "B": 11, "C": 12},
        ... )
        ([99, 11], [10, 11])
    """
    context = [context_policy.no_event_id] * context_policy.context_length
    context_original_event_ids: list[int | None] = [
        None,
    ] * context_policy.context_length
    start_index = max(0, target_index - context_policy.context_length)
    write_offset = context_policy.context_length - (target_index - start_index)
    for context_index in range(start_index, target_index):
        event_id = sequence_event_ids[context_index]
        context_template = sequence.templates[context_index]
        if _is_stale_context_event(
            sequence=sequence,
            context_index=context_index,
            target_index=target_index,
            timeout_ms=context_policy.timeout_ms,
        ):
            event_id = context_policy.no_event_id
        context[write_offset] = event_id
        context_original_event_ids[write_offset] = original_event_id_lookup.get(
            context_template,
        )
        write_offset += 1
    return context, context_original_event_ids


def _sample_label_for_target(
    sequence: TemplateSequence,
    target_index: int,
) -> int:
    """Return the supervised label for one target event.

    Args:
        sequence (TemplateSequence): Source sequence being expanded into
            DeepCASE samples.
        target_index (int): Index of the target event within the source
            sequence.

    Returns:
        int: Binary label for the target event.
    """
    event_label = _target_event_label(sequence, target_index)
    if event_label is not None:
        return int(is_anomalous_label(event_label))
    return int(is_anomalous_label(sequence.label))


def _is_stale_context_event(
    *,
    sequence: TemplateSequence,
    context_index: int,
    target_index: int,
    timeout_ms: int,
) -> bool:
    """Return whether a context event is too old for a target event.

    The source ``TemplateSequence`` stores event-local ``dt_prev_ms`` gaps
    rather than absolute timestamps. To decide whether a context event is still
    valid, DeepCase needs the total elapsed time between that context event and
    the target event. If any intermediate gap is unknown, the helper keeps the
    context event instead of discarding it as stale.

    Args:
        sequence (TemplateSequence): Source sequence containing event gap data.
        context_index (int): Index of the candidate context event.
        target_index (int): Index of the target event.
        timeout_ms (int): Maximum allowed elapsed time between context event
            and target event.

    Returns:
        bool: ``True`` when the context event is stale and should be replaced
        with the no-event sentinel.

    Examples:
        "A" is originally stale for "C" because the total elapsed
        time is 3_000 ms > 2_500 ms. In the second case, "A" is not stale
        for "C" because the elapsed time is unknown, so we keep the context
        event just in case.

        >>> from types import SimpleNamespace
        >>> sequence = SimpleNamespace(
        ...     events=[("A", [], None), ("B", [], 1_000), ("C", [], 2_000)]
        ... )
        >>> _is_stale_context_event(
        ...     sequence=sequence,
        ...     context_index=0,
        ...     target_index=2,
        ...     timeout_ms=2_500,
        ... )
        True
        >>> _is_stale_context_event(
        ...     sequence=SimpleNamespace(
        ...         events=[("A", [], None), ("B", [], None), ("C", [], 2_000)]
        ...     ),
        ...     context_index=0,
        ...     target_index=2,
        ...     timeout_ms=2_500,
        ... )
        False
    """
    elapsed_ms = 0
    saw_elapsed = False
    for event_index in range(context_index + 1, target_index + 1):
        dt_prev_ms = sequence.events[event_index][2]
        if dt_prev_ms is None:
            return False
        elapsed_ms += dt_prev_ms
        saw_elapsed = True
    return saw_elapsed and elapsed_ms > timeout_ms


@dataclass(slots=True)
class _EmptyBatchColumns:
    """Mutable batch columns used while constructing sample tensors.

    Attributes:
        contexts (list[list[int]]): Context rows collected for each sample.
        context_original_event_ids (list[list[int | None]]): Original
            train-vocabulary ids for each collected context slot.
        events (list[int]): Target event ids for each sample.
        scores (list[float]): Per-sample labels or scores.
        event_indexes (list[int]): Target event indexes within their source
            sequences.
        templates (list[str]): Original target templates.
        original_event_ids (list[int | None]): Train-vocabulary ids for the
            target templates. Prediction-time unseen templates stay ``None``
            so the batch can preserve the distinction between an unknown event
            sentinel and a genuinely missing train-vocabulary id.
        parent_sequence_fallback_count (int): Number of samples using parent
            sequence labels.
    """

    contexts: list[list[int]]
    context_original_event_ids: list[list[int | None]]
    events: list[int]
    scores: list[float]
    event_indexes: list[int]
    templates: list[str]
    original_event_ids: list[int | None]
    parent_sequence_fallback_count: int

    @classmethod
    def create(cls) -> _EmptyBatchColumns:
        """Return a fresh mutable set of aligned batch columns.

        Returns:
            _EmptyBatchColumns: Empty aligned columns ready for sample
            accumulation.
        """
        return cls(
            contexts=[],
            context_original_event_ids=[],
            events=[],
            scores=[],
            event_indexes=[],
            templates=[],
            original_event_ids=[],
            parent_sequence_fallback_count=0,
        )

    def to_batch(
        self,
        *,
        context_length: int,
        no_event_placeholder: int | None = None,
        no_event_id: int | None = None,
    ) -> DeepCaseSampleBatch:
        """Freeze the accumulated columns into the immutable batch container.

        Args:
            context_length (int): Fixed context width used to shape an empty
                context tensor correctly.
            no_event_placeholder (int | None): Temporary placeholder value to
                replace before freezing the batch.
            no_event_id (int | None): Final no-event id that replaces
                ``no_event_placeholder`` when provided.

        Returns:
            DeepCaseSampleBatch: Immutable batch with aligned tensor columns.

        Raises:
            ValueError: If a no-event placeholder is provided without the
                replacement no-event id.
        """
        contexts_array = np.asarray(self.contexts, dtype=int)
        if contexts_array.size == 0:
            contexts_array = np.empty((0, context_length), dtype=int)
        if no_event_placeholder is not None:
            if no_event_id is None:
                msg = "no_event_id is required when replacing a no-event placeholder."
                raise ValueError(msg)
            contexts_array = contexts_array.copy()
            contexts_array[contexts_array == no_event_placeholder] = no_event_id

        return DeepCaseSampleBatch(
            contexts=torch.tensor(contexts_array, dtype=torch.long),
            context_original_event_ids=self.context_original_event_ids,
            events=torch.tensor(self.events, dtype=torch.long),
            scores=np.asarray(self.scores, dtype=float),
            event_indexes=self.event_indexes,
            templates=self.templates,
            original_event_ids=self.original_event_ids,
            parent_sequence_fallback_count=self.parent_sequence_fallback_count,
        )


def _append_sequence_samples(
    *,
    sequence: TemplateSequence,
    sequence_event_ids: Sequence[int],
    original_event_id_lookup: dict[str, int],
    context_policy: ContextWindowPolicy,
    batch_columns: _EmptyBatchColumns,
) -> None:
    """Append one sequence's event-centered samples into aligned batch columns.

    Args:
        sequence (TemplateSequence): Source sequence being expanded into event
            samples.
        sequence_event_ids (Sequence[int]): Event ids aligned with
            ``sequence.templates``.
        original_event_id_lookup (dict[str, int]): Mapping used to recover the
            train-vocabulary id for each target template.
        context_policy (ContextWindowPolicy): Context construction policy.
        batch_columns (_EmptyBatchColumns): Mutable aligned columns receiving
            the generated samples.
    """
    for target_index, event_id in enumerate(sequence_event_ids):
        if _target_event_label(sequence, target_index) is None:
            batch_columns.parent_sequence_fallback_count += 1
        sample_label = _sample_label_for_target(sequence, target_index)
        context_ids, context_original_event_ids = _context_for_target(
            sequence=sequence,
            sequence_event_ids=sequence_event_ids,
            target_index=target_index,
            context_policy=context_policy,
            original_event_id_lookup=original_event_id_lookup,
        )
        batch_columns.contexts.append(context_ids)
        batch_columns.context_original_event_ids.append(
            context_original_event_ids,
        )
        batch_columns.events.append(event_id)
        batch_columns.scores.append(float(sample_label))
        batch_columns.event_indexes.append(target_index)
        batch_columns.templates.append(sequence.templates[target_index])
        batch_columns.original_event_ids.append(
            original_event_id_lookup.get(sequence.templates[target_index]),
        )


def _target_event_label(
    sequence: TemplateSequence,
    target_index: int,
) -> int | None:
    """Return the label attached directly to one event, if present.

    Args:
        sequence (TemplateSequence): Source sequence being expanded into
            DeepCASE samples.
        target_index (int): Index of the target event within the source
            sequence.

    Returns:
        int | None: Event-level label for the target event, or ``None`` when
        no event-specific label exists.
    """
    event_labels = sequence.event_labels
    if event_labels is None:
        return None
    return event_labels[target_index]
