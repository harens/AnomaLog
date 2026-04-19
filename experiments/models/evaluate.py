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

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

    from anomalog.sequences import TemplateSequence

_PROGRESS_EVERY = 10_000


@dataclass(slots=True)
class RunMetrics:
    """Accumulate split counts and classification metrics while streaming."""

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

    def update(
        self,
        sequence: TemplateSequence,
        prediction: SequencePrediction,
    ) -> None:
        """Update metrics from one streamed prediction."""
        self.sequence_count += 1
        if sequence.split_label is SplitLabel.TRAIN:
            self.train_sequence_count += 1
            self.train_label_counts[sequence.label] += 1
            return

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
        """Return finalized run metrics.

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
        """Return finalized sequence summary.

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
) -> ModelRunSummary:
    """Fit the configured detector and stream predictions to disk.

    Args:
        sequence_factory (Callable[[], Iterator[TemplateSequence]]): Factory
            producing the full sequence stream.
        config (ExperimentModelConfig): Model config used to build the detector.
        predictions_path (Path): Output path for streamed JSONL predictions.
        logger (logging.Logger): Logger for progress messages.

    Returns:
        ModelRunSummary: Metrics, manifest, and sequence summary for the run.
    """
    detector = config.build_detector()
    fit_detector(
        detector=detector,
        train_sequences=iter_train_sequences(sequence_factory),
        logger=logger,
    )
    accumulator = stream_predictions(
        detector=detector,
        sequence_factory=sequence_factory,
        predictions_path=predictions_path,
        logger=logger,
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
) -> None:
    """Fit a detector on the training split."""
    logger.info("Fitting %s detector on training split", detector.detector_name)
    with make_count_progress() as progress:
        detector.fit(train_sequences, progress=progress, logger=logger)
    logger.info("Finished fitting %s detector", detector.detector_name)


def stream_predictions(
    *,
    detector: ExperimentDetector,
    sequence_factory: Callable[[], Iterator[TemplateSequence]],
    predictions_path: Path,
    logger: logging.Logger,
) -> RunMetrics:
    """Write predictions incrementally while accumulating metrics.

    Args:
        detector (ExperimentDetector): Fitted detector to evaluate.
        sequence_factory (Callable[[], Iterator[TemplateSequence]]): Factory
            producing the full sequence stream.
        predictions_path (Path): Output path for streamed JSONL predictions.
        logger (logging.Logger): Logger for progress messages.

    Returns:
        RunMetrics: Accumulated metrics for the streamed predictions.
    """
    accumulator = RunMetrics()
    with (
        predictions_path.open("w", encoding="utf-8") as file_obj,
        make_count_progress() as progress,
    ):
        for sequence in progress.track(
            sequence_factory(),
            description=f"Scoring {detector.detector_name} sequences",
        ):
            outcome = detector.predict(sequence)
            prediction = outcome.to_prediction_record(sequence)
            file_obj.write(msgspec.json.encode(prediction).decode("utf-8"))
            file_obj.write("\n")
            accumulator.update(sequence, prediction)
            if accumulator.sequence_count % _PROGRESS_EVERY == 0:
                logger.info(
                    "Processed %s sequences for %s detector",
                    accumulator.sequence_count,
                    detector.detector_name,
                )
    return accumulator
