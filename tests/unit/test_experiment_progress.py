"""Tests for shared experiment progress helpers and runtime behaviour."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from rich.progress import Progress
from typing_extensions import override

from anomalog.sequences import SplitLabel, TemplateSequence
from experiments.models.base import (
    ExperimentDetector,
    ModelManifest,
    PredictionOutcome,
    SequenceSummary,
    decode_experiment_model_config,
)
from experiments.models.evaluate import (
    PredictionOutputConfig,
    SequenceFactory,
    iter_train_sequences,
    run_model,
    stream_predictions,
)
from experiments.models.progress import ProgressHint
from experiments.models.template_frequency import TemplateFrequencyModelConfig

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    import pytest

EXPECTED_TEST_SEQUENCE_COUNT = 2


def _sequence(
    window_id: int,
    *,
    templates: list[str],
    label: int,
    split_label: SplitLabel,
) -> TemplateSequence:
    return TemplateSequence(
        events=[(template, [], None) for template in templates],
        label=label,
        entity_ids=[f"entity-{window_id}"],
        window_id=window_id,
        split_label=split_label,
    )


def _interleaved_sequence_stream() -> list[TemplateSequence]:
    """Return a non-prefix-ordered stream representative of interleaved splits.

    Returns:
        list[TemplateSequence]: Stream with a later train sequence after the
            first test sequence, matching the unsafe before-grouping case.
    """
    return [
        _sequence(1, templates=["train-a"], label=0, split_label=SplitLabel.TRAIN),
        _sequence(2, templates=["test-a"], label=0, split_label=SplitLabel.TEST),
        _sequence(3, templates=["train-b"], label=1, split_label=SplitLabel.TRAIN),
    ]


def test_stream_predictions_uses_known_test_total_when_supplied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scoring progress should use the exact test-only total when available.

    Args:
        tmp_path (Path): Temporary filesystem root for the prediction stream.
        monkeypatch (pytest.MonkeyPatch): Replaces the progress factory so the
            test can inspect the created task.
    """

    @dataclass(slots=True)
    class _RecordingDetector(ExperimentDetector):
        detector_name: ClassVar[str] = "recording"

        @override
        def fit(
            self,
            train_sequences: Iterable[TemplateSequence],
            *,
            progress: Progress,
            logger: logging.Logger | None = None,
        ) -> None:
            del train_sequences, progress, logger

        @override
        def predict(self, sequence: TemplateSequence) -> PredictionOutcome:
            del sequence
            return PredictionOutcome(predicted_label=0, score=0.0)

        @override
        def model_manifest(self, *, sequence_summary: SequenceSummary) -> ModelManifest:
            del sequence_summary
            return ModelManifest(
                detector=self.detector_name,
                train_sequence_count=0,
                test_sequence_count=0,
                train_label_counts={},
                test_label_counts={},
            )

    progress = Progress(disable=True)

    def _progress_factory(unit: str | None = None) -> Progress:
        """Return the disabled progress instance injected by the test.

        Args:
            unit (str | None): Optional unit label requested by the caller.

        Returns:
            Progress: Test-owned progress instance.
        """
        del unit
        return progress

    monkeypatch.setattr(
        "experiments.models.evaluate.make_count_progress",
        _progress_factory,
    )
    sequences = [
        _sequence(1, templates=["train-a"], label=0, split_label=SplitLabel.TRAIN),
        _sequence(2, templates=["test-a"], label=0, split_label=SplitLabel.TEST),
        _sequence(3, templates=["test-b"], label=1, split_label=SplitLabel.TEST),
    ]

    stream_predictions(
        detector=_RecordingDetector(),
        sequence_factory=lambda: iter(sequences),
        prediction_output=PredictionOutputConfig(
            predictions_path=tmp_path / "predictions.jsonl",
            write_predictions=True,
        ),
        logger=logging.getLogger("tests.stream_predictions.progress"),
        score_progress_hint=ProgressHint(total=EXPECTED_TEST_SEQUENCE_COUNT),
    )

    assert len(progress.tasks) == 1
    task = progress.tasks[0]
    assert task.total == EXPECTED_TEST_SEQUENCE_COUNT
    assert task.completed == EXPECTED_TEST_SEQUENCE_COUNT
    assert task.description == "Scoring recording test sequences"


def test_iter_train_sequences_keeps_late_train_sequences_by_default() -> None:
    """The shared iterator should remain non-destructive without the flag."""
    sequences = list(
        iter_train_sequences(
            lambda: iter(_interleaved_sequence_stream()),
        ),
    )

    assert [sequence.window_id for sequence in sequences] == [1, 3]


def test_run_model_uses_bounded_train_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model fitting should use the bounded train replay when provided."""

    @dataclass(slots=True)
    class _RecordingDetector(ExperimentDetector):
        detector_name: ClassVar[str] = "recording"

        @override
        def fit(
            self,
            train_sequences: Iterable[TemplateSequence],
            *,
            progress: Progress,
            logger: logging.Logger | None = None,
        ) -> None:
            del progress, logger
            sequences = list(train_sequences)
            assert [sequence.split_label for sequence in sequences] == [
                SplitLabel.TRAIN,
            ]

        @override
        def predict(self, sequence: TemplateSequence) -> PredictionOutcome:
            del sequence
            return PredictionOutcome(predicted_label=0, score=0.0)

        @override
        def model_manifest(
            self,
            *,
            sequence_summary: SequenceSummary,
        ) -> ModelManifest:
            del sequence_summary
            return ModelManifest(
                detector=self.detector_name,
                train_sequence_count=0,
                test_sequence_count=0,
                train_label_counts={},
                test_label_counts={},
            )

    sequences = [
        _sequence(1, templates=["train-a"], label=0, split_label=SplitLabel.TRAIN),
        _sequence(2, templates=["test-a"], label=0, split_label=SplitLabel.TEST),
        _sequence(3, templates=["test-b"], label=1, split_label=SplitLabel.TEST),
    ]
    train_sequences = [sequences[0]]

    config = decode_experiment_model_config(
        {"name": "template_frequency"},
        config_type=TemplateFrequencyModelConfig,
    )
    monkeypatch.setattr(
        TemplateFrequencyModelConfig,
        "build_detector",
        lambda _self: _RecordingDetector(),
    )

    summary = run_model(
        sequence_factory=SequenceFactory(
            factory=lambda: iter(sequences),
            train_factory=lambda: iter(train_sequences),
        ),
        config=config,
        prediction_output=PredictionOutputConfig(
            predictions_path=tmp_path / "predictions.jsonl",
            write_predictions=False,
        ),
        logger=logging.getLogger("tests.run_model.train_factory"),
    )

    assert summary.sequence_summary.train_sequence_count == 1
    expected_test_sequence_count = 2
    assert summary.sequence_summary.test_sequence_count == expected_test_sequence_count
