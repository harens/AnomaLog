"""Metric block schema and validation helpers for experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import msgspec

from experiments.models.metric_schema import (
    AggregationPolicy,
    EvaluationUnit,
    MetricScope,
    MetricStatus,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class BinaryMetricBlockRequest:  # noqa: DOC601 DOC603
    """Input parameters for one binary metric block.

    Attributes:
        metric_scope (MetricScope): Metric block being built.
        prediction_unit (EvaluationUnit): Unit used for detector predictions.
        label_unit (EvaluationUnit): Unit used for ground-truth labels.
        tp (int): True-positive count.
        fp (int): False-positive count.
        tn (int): True-negative count.
        fn (int): False-negative count.
        normal_count (int): Number of normal labels in the evaluation set.
        anomalous_count (int): Number of anomalous labels in the evaluation set.
        evaluation_unit_count (int | None): Optional total evaluation-unit count.
        counted_predictions (int | None): Optional explicit decision count.
        abstained_prediction_count (int | None): Optional abstention count.
        ignored_prediction_count (int | None): Optional ignored-label count.
        aggregation_policy (AggregationPolicy | None): Optional aggregation policy.
        allow_single_class_reporting (bool): Whether to allow single-class blocks.
        diagnostic_only (bool): Whether to mark the block as diagnostic only.
        diagnostics (object | None): Optional diagnostic payload.
    """

    metric_scope: MetricScope
    prediction_unit: EvaluationUnit
    label_unit: EvaluationUnit
    tp: int
    fp: int
    tn: int
    fn: int
    normal_count: int
    anomalous_count: int
    evaluation_unit_count: int | None = None
    counted_predictions: int | None = None
    abstained_prediction_count: int | None = None
    ignored_prediction_count: int | None = None
    aggregation_policy: AggregationPolicy | None = None
    allow_single_class_reporting: bool = False
    diagnostic_only: bool = False
    diagnostics: object | None = None


@dataclass(frozen=True, slots=True)
class DiagnosticMetricBlockRequest:  # noqa: DOC601 DOC603
    """Input parameters for one non-binary metric block.

    Attributes:
        metric_scope (MetricScope): Metric block being built.
        prediction_unit (EvaluationUnit): Unit used for detector predictions.
        label_unit (EvaluationUnit): Unit used for ground-truth labels.
        aggregation_policy (AggregationPolicy | None): Optional aggregation policy.
        status (MetricStatus): Status assigned to the diagnostic block.
        headline_metrics (object | None): Optional headline metrics payload.
        diagnostics (object | None): Optional diagnostic payload.
    """

    metric_scope: MetricScope
    prediction_unit: EvaluationUnit
    label_unit: EvaluationUnit
    aggregation_policy: AggregationPolicy | None = None
    status: MetricStatus = MetricStatus.VALID
    headline_metrics: object | None = None
    diagnostics: object | None = None


class ConfusionMatrix(msgspec.Struct, frozen=True):  # noqa: DOC601 DOC603
    """Binary confusion-matrix counts for one metric block.

    Attributes:
        tp (int): True positives.
        fp (int): False positives.
        tn (int): True negatives.
        fn (int): False negatives.
    """

    tp: int
    fp: int
    tn: int
    fn: int


class MetricBlock(msgspec.Struct, frozen=True):  # noqa: DOC601 DOC603
    """One scoped metric block plus validation state.

    Attributes:
        metric_scope (MetricScope): Scope represented by the block.
        prediction_unit (EvaluationUnit): Unit used for predictions.
        label_unit (EvaluationUnit): Unit used for labels.
        aggregation_policy (AggregationPolicy | None): Optional aggregation policy.
        status (MetricStatus): Validation status for the block.
        invalid_reason (str | None): Reason recorded when the block is invalid.
        evaluation_unit_count (int | None): Optional evaluation-unit total.
        counted_predictions (int | None): Count of predictions included in the block.
        abstained_prediction_count (int | None): Count of abstained predictions.
        ignored_prediction_count (int | None): Count of ignored predictions.
        class_counts (dict[str, int] | None): Normal/anomalous support counts.
        confusion_matrix (ConfusionMatrix | None): Binary confusion counts.
        headline_metrics (Any): Headline metrics payload for valid blocks.
        diagnostics (Any): Additional diagnostics for the block.
    """

    metric_scope: MetricScope
    prediction_unit: EvaluationUnit
    label_unit: EvaluationUnit
    aggregation_policy: AggregationPolicy | None = None
    status: MetricStatus = MetricStatus.VALID
    invalid_reason: str | None = None
    evaluation_unit_count: int | None = None
    counted_predictions: int | None = None
    abstained_prediction_count: int | None = None
    ignored_prediction_count: int | None = None
    class_counts: dict[str, int] | None = None
    confusion_matrix: ConfusionMatrix | None = None
    headline_metrics: Any = msgspec.field(default_factory=dict)
    diagnostics: Any = msgspec.field(default_factory=dict)


_STATUS_PRIORITY: dict[MetricStatus, int] = {
    MetricStatus.VALID: 0,
    MetricStatus.DIAGNOSTIC_ONLY: 1,
    MetricStatus.INVALID: 2,
    MetricStatus.NOT_APPLICABLE: 3,
}


def build_binary_metric_block(
    *,
    request: BinaryMetricBlockRequest,
) -> MetricBlock:
    """Build and validate a binary detection metric block.

    Args:
        request (BinaryMetricBlockRequest): Binary metric inputs to validate.

    Returns:
        MetricBlock: Validated binary metric block with confusion counts and
            headline metrics when the block is valid.
    """
    confusion_matrix = ConfusionMatrix(
        tp=request.tp,
        fp=request.fp,
        tn=request.tn,
        fn=request.fn,
    )
    counted_predictions = (
        request.tp + request.fp + request.tn + request.fn
        if request.counted_predictions is None
        else request.counted_predictions
    )
    status = (
        MetricStatus.DIAGNOSTIC_ONLY if request.diagnostic_only else MetricStatus.VALID
    )
    invalid_reason: str | None = None
    if counted_predictions != request.tp + request.fp + request.tn + request.fn:
        status = MetricStatus.INVALID
        invalid_reason = "confusion_matrix_mismatch"
    elif not request.allow_single_class_reporting and (
        request.normal_count == 0 or request.anomalous_count == 0
    ):
        status = MetricStatus.INVALID
        invalid_reason = "single_class_test_set"
    else:
        total_labels = request.normal_count + request.anomalous_count
        non_counted = 0
        if request.abstained_prediction_count is not None:
            non_counted += request.abstained_prediction_count
        if request.ignored_prediction_count is not None:
            non_counted += request.ignored_prediction_count
        if request.evaluation_unit_count is not None and (
            counted_predictions + non_counted != request.evaluation_unit_count
        ):
            status = MetricStatus.INVALID
            invalid_reason = "evaluation_unit_count_mismatch"
        elif request.label_unit is not request.prediction_unit and (
            request.aggregation_policy is None
        ):
            status = MetricStatus.INVALID
            invalid_reason = "label_unit_prediction_unit_mismatch"
        elif (
            total_labels != request.evaluation_unit_count
            and request.evaluation_unit_count is not None
        ):
            # The total label support should match the relevant evaluation unit
            # count once abstentions and ignored samples are accounted for.
            status = MetricStatus.INVALID
            invalid_reason = "class_count_mismatch"

    precision_denominator = request.tp + request.fp
    recall_denominator = request.tp + request.fn
    precision = request.tp / precision_denominator if precision_denominator else 0.0
    recall = request.tp / recall_denominator if recall_denominator else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    accuracy = (
        (request.tp + request.tn) / counted_predictions if counted_predictions else 0.0
    )
    headline_metrics = (
        {}
        if status is MetricStatus.INVALID
        else {
            "accuracy": round(accuracy, 8),
            "precision": round(precision, 8),
            "recall": round(recall, 8),
            "f1": round(f1, 8),
        }
    )
    diagnostics: Any = (
        request.diagnostics if isinstance(request.diagnostics, dict) else {}
    )
    return MetricBlock(
        metric_scope=request.metric_scope,
        prediction_unit=request.prediction_unit,
        label_unit=request.label_unit,
        aggregation_policy=request.aggregation_policy,
        status=status,
        invalid_reason=invalid_reason,
        evaluation_unit_count=request.evaluation_unit_count,
        counted_predictions=counted_predictions,
        abstained_prediction_count=request.abstained_prediction_count,
        ignored_prediction_count=request.ignored_prediction_count,
        class_counts={
            "normal": request.normal_count,
            "anomalous": request.anomalous_count,
        },
        confusion_matrix=confusion_matrix,
        headline_metrics=headline_metrics,
        diagnostics=diagnostics,
    )


def build_not_applicable_metric_block(
    *,
    metric_scope: MetricScope,
    prediction_unit: EvaluationUnit,
    label_unit: EvaluationUnit,
    diagnostics: Mapping[str, object] | None = None,
) -> MetricBlock:
    """Build a metric block explicitly marked as not applicable.

    Args:
        metric_scope (MetricScope): Metric scope represented by the block.
        prediction_unit (EvaluationUnit): Unit used for predictions.
        label_unit (EvaluationUnit): Unit used for labels.
        diagnostics (Mapping[str, object] | None): Optional diagnostic payload.

    Returns:
        MetricBlock: Block with a not-applicable status and no headline
            metrics.
    """
    return MetricBlock(
        metric_scope=metric_scope,
        prediction_unit=prediction_unit,
        label_unit=label_unit,
        status=MetricStatus.NOT_APPLICABLE,
        diagnostics=dict(diagnostics or {}),
    )


def build_diagnostic_metric_block(
    *,
    request: DiagnosticMetricBlockRequest,
) -> MetricBlock:
    """Build a non-binary diagnostic metric block.

    Args:
        request (DiagnosticMetricBlockRequest): Metric metadata and payload to
            serialise.

    Returns:
        MetricBlock: Diagnostic or headline metric block without confusion
            counts.
    """
    headline_metrics: Any = (
        request.headline_metrics if isinstance(request.headline_metrics, dict) else {}
    )
    diagnostics: Any = (
        request.diagnostics if isinstance(request.diagnostics, dict) else {}
    )
    return MetricBlock(
        metric_scope=request.metric_scope,
        prediction_unit=request.prediction_unit,
        label_unit=request.label_unit,
        aggregation_policy=request.aggregation_policy,
        status=request.status,
        headline_metrics=headline_metrics,
        diagnostics=diagnostics,
    )


def select_primary_metric_scope(
    blocks: Mapping[MetricScope, MetricBlock],
    *,
    requested_primary_scope: MetricScope | None,
    evaluation_unit: EvaluationUnit | None,
) -> MetricScope | None:
    """Select the primary metric scope for a run.

    Args:
        blocks (Mapping[MetricScope, MetricBlock]): Available metric blocks.
        requested_primary_scope (MetricScope | None): Configured preferred
            primary scope.
        evaluation_unit (EvaluationUnit | None): Dataset evaluation unit used
            to rank fallbacks.

    Returns:
        MetricScope | None: Best available primary metric scope for the run,
            or ``None`` when no metric blocks exist.
    """
    if requested_primary_scope is not None:
        block = blocks.get(requested_primary_scope)
        if block is not None and block.status not in {
            MetricStatus.INVALID,
            MetricStatus.NOT_APPLICABLE,
        }:
            return requested_primary_scope

    preference = _metric_scope_preference(evaluation_unit)
    best_scope: MetricScope | None = None
    best_rank = len(_STATUS_PRIORITY)
    for scope in preference:
        block = blocks.get(scope)
        if block is None:
            continue
        rank = block_status_rank(block.status)
        if rank < best_rank:
            best_scope = scope
            best_rank = rank
            if rank == _STATUS_PRIORITY[MetricStatus.VALID]:
                break
    return best_scope if best_scope is not None else next(iter(blocks), None)


def _metric_scope_preference(
    evaluation_unit: EvaluationUnit | None,
) -> tuple[MetricScope, ...]:
    if evaluation_unit in {
        EvaluationUnit.CHRONOLOGICAL_EVENT_STREAM,
        EvaluationUnit.CONTINUOUS_EVENT_STREAM,
        EvaluationUnit.STREAM,
    }:
        return (
            MetricScope.EVENT_LEVEL_DETECTION,
            MetricScope.NEXT_EVENT_PREDICTION,
            MetricScope.SEQUENCE_LEVEL_DETECTION,
            MetricScope.WINDOW_LEVEL_DETECTION,
            MetricScope.CLUSTER_LEVEL_TRIAGE,
            MetricScope.MANUAL_WORKLOAD_REDUCTION,
            MetricScope.SEMI_AUTOMATIC_WORKLOAD_REDUCTION,
        )
    if evaluation_unit in {
        EvaluationUnit.SEQUENCE,
        EvaluationUnit.WINDOW,
        EvaluationUnit.CLUSTER,
    }:
        return (
            MetricScope.SEQUENCE_LEVEL_DETECTION,
            MetricScope.EVENT_LEVEL_DETECTION,
            MetricScope.NEXT_EVENT_PREDICTION,
            MetricScope.WINDOW_LEVEL_DETECTION,
            MetricScope.CLUSTER_LEVEL_TRIAGE,
            MetricScope.MANUAL_WORKLOAD_REDUCTION,
            MetricScope.SEMI_AUTOMATIC_WORKLOAD_REDUCTION,
        )
    if evaluation_unit is EvaluationUnit.NEXT_EVENT:
        return (
            MetricScope.NEXT_EVENT_PREDICTION,
            MetricScope.EVENT_LEVEL_DETECTION,
            MetricScope.SEQUENCE_LEVEL_DETECTION,
            MetricScope.WINDOW_LEVEL_DETECTION,
        )
    return (
        MetricScope.SEQUENCE_LEVEL_DETECTION,
        MetricScope.EVENT_LEVEL_DETECTION,
        MetricScope.NEXT_EVENT_PREDICTION,
        MetricScope.WINDOW_LEVEL_DETECTION,
        MetricScope.CLUSTER_LEVEL_TRIAGE,
        MetricScope.MANUAL_WORKLOAD_REDUCTION,
        MetricScope.SEMI_AUTOMATIC_WORKLOAD_REDUCTION,
    )


def block_status_rank(status: MetricStatus) -> int:
    """Return a stable ordering for block statuses.

    Args:
        status (MetricStatus): Block status being ranked.

    Returns:
        int: Lower values indicate a more preferred status.
    """
    return _STATUS_PRIORITY[status]
