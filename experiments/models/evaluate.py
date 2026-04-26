"""Shared streaming evaluation runtime for experiment detectors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import msgspec

from anomalog.io_utils import make_count_progress
from anomalog.sequences import SplitLabel
from experiments.models.base import (
    ExperimentDetector,
    ExperimentModelConfig,
    ModelRunSummary,
    SequencePrediction,
    SequenceSummary,
)
from experiments.models.progress import (
    ProgressHint,
    RunProgressPlan,
    score_stage_description,
    with_known_total,
)

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

    from anomalog.sequences import TemplateSequence

_PROGRESS_EVERY = 10_000
TrainProgressHint = ProgressHint


@dataclass(slots=True)
class RunMetrics:
    """Accumulate split counts and classification metrics while streaming.

    Attributes:
        sequence_count (int): Total scored sequences across all splits.
        train_sequence_count (int): Number of train-split sequences seen.
        test_sequence_count (int): Number of test-split sequences seen.
        tp (int): True positives on the test split.
        tn (int): True negatives on the test split.
        fp (int): False positives on the test split.
        fn (int): False negatives on the test split.
        test_score_sum (float): Running sum of test-split anomaly scores.
        train_label_counts (Counter[int]): Train label histogram.
        test_label_counts (Counter[int]): Test label histogram.
    """

    sequence_count: int = 0
    train_sequence_count: int = 0
    test_sequence_count: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    test_score_sum: float = 0.0
    train_label_counts: Counter[int] = field(default_factory=Counter)
    test_label_counts: Counter[int] = field(default_factory=Counter)

    def record_train(
        self,
        sequence: TemplateSequence,
    ) -> None:
        """Record one train-split sequence without scoring it.

        Args:
            sequence (TemplateSequence): Train-split sequence seen while
                streaming the dataset.
        """
        self.sequence_count += 1
        self.train_sequence_count += 1
        self.train_label_counts[sequence.label] += 1

    def record_test(
        self,
        sequence: TemplateSequence,
        prediction: SequencePrediction,
    ) -> None:
        """Update metrics from one test-split prediction.

        Args:
            sequence (TemplateSequence): Test-split sequence that produced
                the prediction.
            prediction (SequencePrediction): Serialised prediction record for
                the sequence.
        """
        self.sequence_count += 1
        self.test_sequence_count += 1
        self.test_label_counts[sequence.label] += 1
        self.test_score_sum += prediction.score
        if prediction.label == 1 and prediction.predicted_label == 1:
            self.tp += 1
        elif prediction.label == 0 and prediction.predicted_label == 0:
            self.tn += 1
        elif prediction.label == 0 and prediction.predicted_label == 1:
            self.fp += 1
        else:
            self.fn += 1

    def metrics(self) -> dict[str, int | float]:
        """Return finalised run metrics.

        Returns:
            dict[str, int | float]: Aggregate classification and split metrics.
        """
        test_count = self.test_sequence_count
        accuracy = (self.tp + self.tn) / test_count if test_count else 0.0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        mean_test_score = self.test_score_sum / test_count if test_count else 0.0
        return {
            "sequence_count": self.sequence_count,
            "train_sequence_count": self.train_sequence_count,
            "test_sequence_count": self.test_sequence_count,
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "accuracy": round(accuracy, 8),
            "precision": round(precision, 8),
            "recall": round(recall, 8),
            "f1": round(f1, 8),
            "mean_test_score": round(mean_test_score, 8),
        }

    def summary(self) -> SequenceSummary:
        """Return finalised sequence summary.

        Returns:
            SequenceSummary: Aggregate split and label counts.
        """
        return SequenceSummary(
            sequence_count=self.sequence_count,
            train_sequence_count=self.train_sequence_count,
            test_sequence_count=self.test_sequence_count,
            train_label_counts=dict(self.train_label_counts),
            test_label_counts=dict(self.test_label_counts),
        )


def run_model(
    *,
    sequence_factory: Callable[[], Iterator[TemplateSequence]],
    config: ExperimentModelConfig,
    predictions_path: Path,
    logger: logging.Logger,
    progress_plan: RunProgressPlan | None = None,
) -> ModelRunSummary:
    """Fit the configured detector and stream predictions to disk.

    Args:
        sequence_factory (Callable[[], Iterator[TemplateSequence]]): Factory
            producing the full sequence stream.
        config (ExperimentModelConfig): Model config used to build the detector.
        predictions_path (Path): Output path for streamed JSONL predictions.
        logger (logging.Logger): Logger for progress messages.
        progress_plan (RunProgressPlan | None): Exact bounded fit/scoring
            metadata when the caller can provide it cheaply.

    Returns:
        ModelRunSummary: Metrics, manifest, and sequence summary for the run.
    """
    detector = config.build_detector()
    fit_detector(
        detector=detector,
        train_sequences=iter_train_sequences(sequence_factory),
        logger=logger,
        train_progress_hint=None if progress_plan is None else progress_plan.train,
    )
    accumulator = stream_predictions(
        detector=detector,
        sequence_factory=sequence_factory,
        predictions_path=predictions_path,
        logger=logger,
        score_progress_hint=None if progress_plan is None else progress_plan.score,
    )
    sequence_summary = accumulator.summary()
    return ModelRunSummary(
        metrics=accumulator.metrics(),
        model_manifest=detector.model_manifest(sequence_summary=sequence_summary),
        sequence_summary=sequence_summary,
    )


def iter_train_sequences(
    sequence_factory: Callable[[], Iterator[TemplateSequence]],
) -> Iterator[TemplateSequence]:
    """Yield only training sequences from a sequence factory.

    Args:
        sequence_factory (Callable[[], Iterator[TemplateSequence]]): Factory
            producing the full sequence stream.

    Yields:
        TemplateSequence: Sequences belonging to the training split.
    """
    for sequence in sequence_factory():
        if sequence.split_label is SplitLabel.TRAIN:
            yield sequence


def fit_detector(
    *,
    detector: ExperimentDetector,
    train_sequences: Iterable[TemplateSequence],
    logger: logging.Logger,
    train_progress_hint: ProgressHint | None = None,
) -> None:
    """Fit a detector on the training split.

    Args:
        detector (ExperimentDetector): Detector to fit.
        train_sequences (Iterable[TemplateSequence]): Training sequences to replay.
        logger (logging.Logger): Logger used for progress messages.
        train_progress_hint (ProgressHint | None): Exact bounded train-fit
            progress metadata when known cheaply by the caller.
    """
    logger.info("Fitting %s detector on training split", detector.detector_name)
    train_sequences = with_known_total(train_sequences, hint=train_progress_hint)
    with make_count_progress(
        unit=None if train_progress_hint is None else train_progress_hint.unit,
    ) as progress:
        detector.fit(train_sequences, progress=progress, logger=logger)
    logger.info("Finished fitting %s detector", detector.detector_name)


def stream_predictions(
    *,
    detector: ExperimentDetector,
    sequence_factory: Callable[[], Iterator[TemplateSequence]],
    predictions_path: Path,
    logger: logging.Logger,
    score_progress_hint: ProgressHint | None = None,
) -> RunMetrics:
    """Write test predictions incrementally while accumulating metrics.

    Args:
        detector (ExperimentDetector): Fitted detector to evaluate.
        sequence_factory (Callable[[], Iterator[TemplateSequence]]): Factory
            producing the full sequence stream.
        predictions_path (Path): Output path for streamed JSONL predictions.
        logger (logging.Logger): Logger for progress messages.
        score_progress_hint (ProgressHint | None): Exact bounded test-scoring
            progress metadata when known cheaply by the caller.

    Returns:
        RunMetrics: Accumulated metrics for the streamed run.
    """
    accumulator = RunMetrics()
    with (
        predictions_path.open("w", encoding="utf-8") as file_obj,
        make_count_progress(
            unit=None if score_progress_hint is None else score_progress_hint.unit,
        ) as progress,
    ):
        score_task = progress.add_task(
            score_stage_description(detector.detector_name),
            total=None if score_progress_hint is None else score_progress_hint.total,
        )
        for sequence in sequence_factory():
            if sequence.split_label is SplitLabel.TRAIN:
                accumulator.record_train(sequence)
                continue

            outcome = detector.predict(sequence)
            prediction = outcome.to_prediction_record(sequence)
            file_obj.write(msgspec.json.encode(prediction.to_dict()).decode("utf-8"))
            file_obj.write("\n")
            accumulator.record_test(sequence, prediction)
            progress.advance(score_task)
            if accumulator.test_sequence_count % _PROGRESS_EVERY == 0:
                logger.info(
                    "Processed %s test sequences for %s detector",
                    accumulator.test_sequence_count,
                    detector.detector_name,
                )
    return accumulator
