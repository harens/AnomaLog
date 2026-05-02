"""Tests for shared experiment evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from anomalog.sequences import SplitLabel, TemplateSequence
from experiments.models.base import PredictionOutcome
from experiments.models.evaluate import RunMetrics


@dataclass(frozen=True, slots=True)
class _TestPredictionOutcome(PredictionOutcome):
    abstained: bool = False

    def is_abstained(self) -> bool:
        return self.abstained


def _sequence(*, label: int) -> TemplateSequence:
    return TemplateSequence(
        events=[("A", [], None)],
        label=label,
        entity_ids=["entity-1"],
        window_id=0,
        split_label=SplitLabel.TEST,
    )


def test_run_metrics_counts_abstentions_separately() -> None:
    """Run metrics should ignore abstentions in the confusion matrix."""
    metrics = RunMetrics()
    metrics.record_test(
        _sequence(label=1),
        _TestPredictionOutcome(predicted_label=1, score=0.25),
        abstained=False,
    )
    metrics.record_test(
        _sequence(label=0),
        _TestPredictionOutcome(predicted_label=0, score=0.5),
        abstained=False,
    )
    metrics.record_test(
        _sequence(label=1),
        _TestPredictionOutcome(predicted_label=0, score=0.75, abstained=True),
        abstained=True,
    )

    summary = metrics.metrics()

    assert metrics.tp == 1
    assert metrics.tn == 1
    assert metrics.fp == 0
    assert metrics.fn == 0
    assert not hasattr(metrics, "auto_decision_count")
    assert not hasattr(metrics, "abstained_prediction_count")
    assert summary["accuracy"] == pytest.approx(1.0)
    assert summary["precision"] == pytest.approx(1.0)
    assert summary["recall"] == pytest.approx(1.0)
    assert summary["f1"] == pytest.approx(1.0)
    assert summary["mean_test_score"] == pytest.approx(0.5)
