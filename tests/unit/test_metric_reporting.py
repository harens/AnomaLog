"""Tests for scoped metric block validation and selection."""

from __future__ import annotations

import pytest

from experiments.models.metric_reporting import (
    BinaryMetricBlockRequest,
    DiagnosticMetricBlockRequest,
    build_binary_metric_block,
    build_diagnostic_metric_block,
    build_not_applicable_metric_block,
    select_primary_metric_scope,
)
from experiments.models.metric_schema import (
    EvaluationUnit,
    MetricScope,
    MetricStatus,
)

EXPECTED_COUNTED_PREDICTIONS = 4
EXPECTED_TEST_COUNT = 5


def test_sequence_level_binary_metrics_are_valid_for_mixed_labels() -> None:
    """Mixed labels should produce a valid sequence-level detection block."""
    block = build_binary_metric_block(
        request=BinaryMetricBlockRequest(
            metric_scope=MetricScope.SEQUENCE_LEVEL_DETECTION,
            prediction_unit=EvaluationUnit.SEQUENCE,
            label_unit=EvaluationUnit.SEQUENCE,
            tp=6,
            fp=2,
            tn=10,
            fn=1,
            normal_count=12,
            anomalous_count=7,
            evaluation_unit_count=19,
            counted_predictions=19,
            abstained_prediction_count=0,
            ignored_prediction_count=0,
        ),
    )

    assert block.status is MetricStatus.VALID
    assert block.invalid_reason is None
    assert block.class_counts == {"normal": 12, "anomalous": 7}
    assert not hasattr(block, "aggregation_policy")
    assert block.confusion_matrix is not None
    assert block.headline_metrics["precision"] == pytest.approx(6 / 8)
    assert block.headline_metrics["recall"] == pytest.approx(6 / 7)


def test_single_class_sequence_level_metrics_are_invalid() -> None:
    """Single-class binary detection blocks should be rejected as invalid."""
    block = build_binary_metric_block(
        request=BinaryMetricBlockRequest(
            metric_scope=MetricScope.SEQUENCE_LEVEL_DETECTION,
            prediction_unit=EvaluationUnit.SEQUENCE,
            label_unit=EvaluationUnit.SEQUENCE,
            tp=48,
            fp=0,
            tn=0,
            fn=0,
            normal_count=0,
            anomalous_count=48,
            evaluation_unit_count=48,
            counted_predictions=48,
            abstained_prediction_count=0,
            ignored_prediction_count=0,
        ),
    )

    assert block.status is MetricStatus.INVALID
    assert block.invalid_reason == "single_class_test_set"
    assert block.headline_metrics == {}
    assert block.class_counts == {"normal": 0, "anomalous": 48}
    assert block.confusion_matrix is not None


def test_event_level_streaming_metrics_can_remain_diagnostic_only() -> None:
    """Secondary event-level blocks should stay diagnostic-only on streams."""
    event_block = build_binary_metric_block(
        request=BinaryMetricBlockRequest(
            metric_scope=MetricScope.EVENT_LEVEL_DETECTION,
            prediction_unit=EvaluationUnit.EVENT,
            label_unit=EvaluationUnit.EVENT,
            tp=9,
            fp=1,
            tn=7,
            fn=2,
            normal_count=8,
            anomalous_count=11,
            evaluation_unit_count=19,
            counted_predictions=19,
            abstained_prediction_count=0,
            ignored_prediction_count=0,
            diagnostic_only=True,
        ),
    )
    sequence_block = build_not_applicable_metric_block(
        metric_scope=MetricScope.SEQUENCE_LEVEL_DETECTION,
        prediction_unit=EvaluationUnit.SEQUENCE,
        label_unit=EvaluationUnit.SEQUENCE,
    )

    assert event_block.status is MetricStatus.DIAGNOSTIC_ONLY
    assert sequence_block.status is MetricStatus.NOT_APPLICABLE
    assert (
        select_primary_metric_scope(
            {
                MetricScope.EVENT_LEVEL_DETECTION: event_block,
                MetricScope.SEQUENCE_LEVEL_DETECTION: sequence_block,
            },
            requested_primary_scope=None,
            evaluation_unit=EvaluationUnit.CONTINUOUS_EVENT_STREAM,
        )
        == MetricScope.EVENT_LEVEL_DETECTION
    )


def test_next_event_only_metrics_leave_anomaly_blocks_not_applicable() -> None:
    """Next-event-only runs should not fabricate binary anomaly metrics."""
    next_event_block = build_diagnostic_metric_block(
        request=DiagnosticMetricBlockRequest(
            metric_scope=MetricScope.NEXT_EVENT_PREDICTION,
            prediction_unit=EvaluationUnit.NEXT_EVENT,
            label_unit=EvaluationUnit.NEXT_EVENT,
            headline_metrics={"coverage": 0.75},
            diagnostics={"task": "next_event_prediction"},
        ),
    )
    event_block = build_not_applicable_metric_block(
        metric_scope=MetricScope.EVENT_LEVEL_DETECTION,
        prediction_unit=EvaluationUnit.EVENT,
        label_unit=EvaluationUnit.EVENT,
    )
    sequence_block = build_not_applicable_metric_block(
        metric_scope=MetricScope.SEQUENCE_LEVEL_DETECTION,
        prediction_unit=EvaluationUnit.SEQUENCE,
        label_unit=EvaluationUnit.SEQUENCE,
    )

    assert next_event_block.status is MetricStatus.VALID
    assert event_block.status is MetricStatus.NOT_APPLICABLE
    assert sequence_block.status is MetricStatus.NOT_APPLICABLE
    assert (
        select_primary_metric_scope(
            {
                MetricScope.NEXT_EVENT_PREDICTION: next_event_block,
                MetricScope.EVENT_LEVEL_DETECTION: event_block,
                MetricScope.SEQUENCE_LEVEL_DETECTION: sequence_block,
            },
            requested_primary_scope=MetricScope.NEXT_EVENT_PREDICTION,
            evaluation_unit=EvaluationUnit.NEXT_EVENT,
        )
        == MetricScope.NEXT_EVENT_PREDICTION
    )


def test_deepcase_style_abstention_keeps_coverage_separate() -> None:
    """Abstained predictions should reconcile without breaking auto metrics."""
    block = build_binary_metric_block(
        request=BinaryMetricBlockRequest(
            metric_scope=MetricScope.SEQUENCE_LEVEL_DETECTION,
            prediction_unit=EvaluationUnit.SEQUENCE,
            label_unit=EvaluationUnit.SEQUENCE,
            tp=2,
            fp=1,
            tn=1,
            fn=0,
            normal_count=2,
            anomalous_count=3,
            evaluation_unit_count=5,
            counted_predictions=4,
            abstained_prediction_count=1,
            ignored_prediction_count=0,
        ),
    )

    assert block.status is MetricStatus.VALID
    assert block.counted_predictions == EXPECTED_COUNTED_PREDICTIONS
    assert block.abstained_prediction_count == 1
    assert block.counted_predictions + block.abstained_prediction_count == (
        EXPECTED_TEST_COUNT
    )
    assert block.headline_metrics["accuracy"] == pytest.approx(3 / 4)


def test_binary_metric_blocks_require_matching_units() -> None:
    """Binary blocks should still reject mismatched prediction and label units."""
    block = build_binary_metric_block(
        request=BinaryMetricBlockRequest(
            metric_scope=MetricScope.EVENT_LEVEL_DETECTION,
            prediction_unit=EvaluationUnit.EVENT,
            label_unit=EvaluationUnit.SEQUENCE,
            tp=1,
            fp=0,
            tn=1,
            fn=0,
            normal_count=1,
            anomalous_count=1,
            evaluation_unit_count=2,
            counted_predictions=2,
            abstained_prediction_count=0,
            ignored_prediction_count=0,
        ),
    )

    assert block.status is MetricStatus.INVALID
    assert block.invalid_reason == "label_unit_prediction_unit_mismatch"
