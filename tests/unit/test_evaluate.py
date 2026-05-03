"""Tests for shared experiment evaluation metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from anomalog.sequences import SplitLabel, TemplateSequence
from experiments.models.base import ModelManifest, PredictionOutcome, SequenceSummary
from experiments.models.deepcase.detector import DeepCasePredictionOutcome
from experiments.models.deepcase.shared import DeepCaseSequenceDecision
from experiments.models.evaluate import (
    PredictionOutputConfig,
    RunMetrics,
    stream_predictions,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from rich.progress import Progress


@dataclass(frozen=True, slots=True)
class _TestPredictionOutcome(PredictionOutcome):
    abstained: bool = False

    @property
    def is_abstained(self) -> bool:
        return self.abstained


@dataclass(slots=True)
class _StubDetector:
    outcomes: dict[int, DeepCasePredictionOutcome]
    detector_name: str = "deepcase"

    @staticmethod
    def fit(
        train_sequences: Iterable[TemplateSequence],
        *,
        progress: Progress,
        logger: logging.Logger | None = None,
    ) -> None:
        del train_sequences, progress, logger

    def predict(self, sequence: TemplateSequence) -> DeepCasePredictionOutcome:
        return self.outcomes[sequence.window_id]

    def model_manifest(self, *, sequence_summary: SequenceSummary) -> ModelManifest:
        del sequence_summary
        return ModelManifest(
            detector=self.detector_name,
            train_sequence_count=0,
            test_sequence_count=0,
            train_label_counts={},
            test_label_counts={},
        )

    @staticmethod
    def run_metrics(
        *,
        run_metrics: dict[str, int | float | dict[int, int]],
    ) -> None:
        del run_metrics


def _sequence(*, label: int, window_id: int = 0) -> TemplateSequence:
    return TemplateSequence(
        events=[("A", [], None)],
        label=label,
        entity_ids=["entity-1"],
        window_id=window_id,
        split_label=SplitLabel.TEST,
    )


def test_run_metrics_counts_abstentions_separately() -> None:
    """Run metrics should ignore abstentions in the confusion matrix."""
    expected_counted_predictions = 2
    expected_abstained_predictions = 1
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
    assert metrics.abstained_prediction_count == 1
    assert summary["accuracy"] == pytest.approx(1.0)
    assert summary["precision"] == pytest.approx(1.0)
    assert summary["recall"] == pytest.approx(1.0)
    assert summary["f1"] == pytest.approx(1.0)
    assert summary["mean_test_score"] == pytest.approx(0.5)
    assert summary["counted_predictions"] == expected_counted_predictions
    assert summary["abstained_prediction_count"] == expected_abstained_predictions


@pytest.mark.parametrize(
    ("sequence_label", "predicted_label", "expected_key"),
    [
        (1, 1, "tp"),
        (0, 1, "fp"),
        (0, 0, "tn"),
        (1, 0, "fn"),
    ],
)
def test_run_metrics_counts_automatic_predictions_by_label(
    sequence_label: int,
    predicted_label: int,
    expected_key: str,
) -> None:
    """Automatic predictions should feed the matching confusion-matrix cell.

    Args:
        sequence_label (int): Ground-truth sequence label under test.
        predicted_label (int): Automatic detector label to record.
        expected_key (str): Confusion-matrix bucket that should increment.
    """
    metrics = RunMetrics()
    metrics.record_test(
        _sequence(label=sequence_label),
        _TestPredictionOutcome(predicted_label=predicted_label, score=0.25),
        abstained=False,
    )

    summary = metrics.metrics()

    assert getattr(metrics, expected_key) == 1
    assert metrics.tp + metrics.tn + metrics.fp + metrics.fn == 1
    assert summary["counted_predictions"] == 1
    assert summary["abstained_prediction_count"] == 0


def test_stream_predictions_counts_deepcase_automatic_decisions(
    tmp_path: Path,
) -> None:
    """DeepCASE outcomes should retain automatic decisions in shared metrics.

    Args:
        tmp_path (Path): Temporary directory for the streamed predictions
            output path.
    """
    expected_tp = 1
    expected_fp = 1
    expected_tn = 1
    expected_fn = 1
    expected_abstained_predictions = 1
    expected_counted_predictions = 4
    expected_test_sequence_count = 5
    sequences = [
        _sequence(label=1, window_id=0),
        _sequence(label=0, window_id=1),
        _sequence(label=0, window_id=2),
        _sequence(label=1, window_id=3),
        _sequence(label=1, window_id=4),
    ]

    detector = _StubDetector(
        outcomes={
            0: DeepCasePredictionOutcome(
                predicted_label=1,
                score=1.0,
                findings=[],
                sequence_decision=DeepCaseSequenceDecision.CONFIDENT_ANOMALY,
                confident_event_count=1,
                abstained_event_count=0,
                confident_anomaly_event_count=1,
            ),
            1: DeepCasePredictionOutcome(
                predicted_label=1,
                score=1.0,
                findings=[],
                sequence_decision=DeepCaseSequenceDecision.CONFIDENT_ANOMALY,
                confident_event_count=1,
                abstained_event_count=0,
                confident_anomaly_event_count=1,
            ),
            2: DeepCasePredictionOutcome(
                predicted_label=0,
                score=0.0,
                findings=[],
                sequence_decision=DeepCaseSequenceDecision.CONFIDENT_NORMAL,
                confident_event_count=1,
                abstained_event_count=0,
                confident_anomaly_event_count=0,
            ),
            3: DeepCasePredictionOutcome(
                predicted_label=0,
                score=0.0,
                findings=[],
                sequence_decision=DeepCaseSequenceDecision.CONFIDENT_NORMAL,
                confident_event_count=1,
                abstained_event_count=0,
                confident_anomaly_event_count=0,
            ),
            4: DeepCasePredictionOutcome(
                predicted_label=0,
                score=0.0,
                findings=[],
                sequence_decision=DeepCaseSequenceDecision.ABSTAINED,
                confident_event_count=0,
                abstained_event_count=1,
                confident_anomaly_event_count=0,
            ),
        },
    )

    accumulator = stream_predictions(
        detector=detector,
        sequence_factory=lambda: iter(sequences),
        prediction_output=PredictionOutputConfig(
            predictions_path=tmp_path / "unused-predictions.jsonl",
            write_predictions=False,
        ),
        logger=logging.getLogger("tests.evaluate"),
    )
    summary = accumulator.metrics()

    assert summary["tp"] == expected_tp
    assert summary["fp"] == expected_fp
    assert summary["tn"] == expected_tn
    assert summary["fn"] == expected_fn
    assert summary["abstained_prediction_count"] == expected_abstained_predictions
    assert summary["counted_predictions"] == expected_counted_predictions
    assert accumulator.test_sequence_count == expected_test_sequence_count
    assert accumulator.abstained_prediction_count == expected_abstained_predictions
