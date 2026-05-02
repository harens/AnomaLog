"""Shared streaming evaluation runtime for experiment detectors."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import msgspec

from anomalog.io_utils import make_count_progress
from anomalog.parsers.structured.contracts import is_anomalous_label
from anomalog.sequences import SplitLabel
from experiments.models.base import (
    AbstainAwarePredictionOutcome,
    BatchExperimentDetector,
    ExperimentDetector,
    ExperimentModelConfig,
    ModelRunSummary,
    PredictionOutcome,
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


@dataclass(frozen=True, slots=True)
class PredictionOutputConfig:
    """Settings for writing streamed test predictions.

    Attributes:
        predictions_path (Path): Output path for streamed JSONL predictions.
        write_predictions (bool): Whether to persist `predictions.jsonl`.
    """

    predictions_path: Path
    write_predictions: bool = False


@dataclass(slots=True)
class RunMetrics:
    """Accumulate split counts and classification metrics while streaming.

    Attributes:
        sequence_count (int): Total processed sequences across train, ignored,
            and test splits.
        train_sequence_count (int): Number of train-split sequences seen.
        test_sequence_count (int): Number of test-split sequences seen.
        ignored_sequence_count (int): Number of sequences withheld from the
            current training prefix.
        tp (int): True positives on the test split.
        tn (int): True negatives on the test split.
        fp (int): False positives on the test split.
        fn (int): False negatives on the test split.
        test_score_sum (float): Running sum of test-split anomaly scores.
        train_label_counts (dict[int, int]): Train label histogram.
        test_label_counts (dict[int, int]): Test label histogram.
        ignored_label_counts (dict[int, int]): Label histogram for withheld
            sequences.
    """

    sequence_count: int = 0
    train_sequence_count: int = 0
    test_sequence_count: int = 0
    ignored_sequence_count: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    test_score_sum: float = 0.0
    train_label_counts: dict[int, int] = field(default_factory=dict)
    test_label_counts: dict[int, int] = field(default_factory=dict)
    ignored_label_counts: dict[int, int] = field(default_factory=dict)

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
        self.train_label_counts[sequence.label] = (
            self.train_label_counts.get(sequence.label, 0) + 1
        )

    def record_ignored(self, sequence: TemplateSequence) -> None:
        """Record one ignored sequence that is withheld from both stages.

        Args:
            sequence (TemplateSequence): Sequence that remains outside the
                current train prefix and fixed test suffix.
        """
        self.sequence_count += 1
        self.ignored_sequence_count += 1
        self.ignored_label_counts[sequence.label] = (
            self.ignored_label_counts.get(sequence.label, 0) + 1
        )

    def record_test(
        self,
        sequence: TemplateSequence,
        prediction: PredictionOutcome,
        *,
        abstained: bool = False,
    ) -> None:
        """Update metrics from one test-split prediction.

        Args:
            sequence (TemplateSequence): Test-split sequence that produced
                the prediction.
            prediction (PredictionOutcome): Detector output for the sequence.
            abstained (bool): Whether the detector deferred the sequence for
                manual review.
        """
        self.sequence_count += 1
        self.test_sequence_count += 1
        self.test_label_counts[sequence.label] = (
            self.test_label_counts.get(sequence.label, 0) + 1
        )
        self.test_score_sum += prediction.score
        if abstained:
            # Abstained predictions still count towards test coverage and mean
            # score, but they must not affect the automatic confusion matrix.
            return

        label_is_anomalous = is_anomalous_label(sequence.label)
        if label_is_anomalous and prediction.predicted_label == 1:
            self.tp += 1
        elif not label_is_anomalous and prediction.predicted_label == 0:
            self.tn += 1
        elif not label_is_anomalous and prediction.predicted_label == 1:
            self.fp += 1
        else:
            self.fn += 1

    def metrics(self) -> dict[str, int | float | dict[int, int]]:
        """Return finalised run metrics.

        Returns:
            dict[str, int | float | dict[int, int]]: Aggregate classification
                and split metrics.
        """
        decision_count = self.tp + self.tn + self.fp + self.fn
        test_count = self.test_sequence_count
        accuracy = (self.tp + self.tn) / decision_count if decision_count else 0.0
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        mean_test_score = self.test_score_sum / test_count if test_count else 0.0
        metrics: dict[str, int | float | dict[int, int]] = {
            "sequence_count": self.sequence_count,
            "train_sequence_count": self.train_sequence_count,
            "test_sequence_count": self.test_sequence_count,
            "ignored_sequence_count": self.ignored_sequence_count,
            "ignored_label_counts": dict(self.ignored_label_counts),
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
        return metrics

    def summary(self) -> SequenceSummary:
        """Return finalised sequence summary.

        Returns:
            SequenceSummary: Aggregate split and label counts.
        """
        return SequenceSummary(
            sequence_count=self.sequence_count,
            train_sequence_count=self.train_sequence_count,
            test_sequence_count=self.test_sequence_count,
            ignored_sequence_count=self.ignored_sequence_count,
            train_label_counts=dict(self.train_label_counts),
            test_label_counts=dict(self.test_label_counts),
            ignored_label_counts=dict(self.ignored_label_counts),
        )


def run_model(
    *,
    sequence_factory: Callable[[], Iterator[TemplateSequence]],
    config: ExperimentModelConfig,
    prediction_output: PredictionOutputConfig,
    logger: logging.Logger,
    progress_plan: RunProgressPlan | None = None,
) -> ModelRunSummary:
    """Fit the configured detector and stream predictions to disk.

    Args:
        sequence_factory (Callable[[], Iterator[TemplateSequence]]): Factory
            producing the full sequence stream.
        config (ExperimentModelConfig): Model config used to build the detector.
        prediction_output (PredictionOutputConfig): Prediction stream settings.
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
        prediction_output=prediction_output,
        logger=logger,
        score_progress_hint=None if progress_plan is None else progress_plan.score,
    )
    sequence_summary = accumulator.summary()
    metrics = accumulator.metrics()
    extra_metrics = detector.run_metrics(run_metrics=metrics)
    if extra_metrics is not None:
        metrics = {**metrics, **msgspec.to_builtins(extra_metrics)}
    return ModelRunSummary(
        metrics=metrics,
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
    prediction_output: PredictionOutputConfig,
    logger: logging.Logger,
    score_progress_hint: ProgressHint | None = None,
) -> RunMetrics:
    """Write test predictions incrementally while accumulating metrics.

    Args:
        detector (ExperimentDetector): Fitted detector to evaluate.
        sequence_factory (Callable[[], Iterator[TemplateSequence]]): Factory
            producing the full sequence stream.
        prediction_output (PredictionOutputConfig): Prediction stream settings.
        logger (logging.Logger): Logger for progress messages.
        score_progress_hint (ProgressHint | None): Exact bounded test-scoring
            progress metadata when known cheaply by the caller.

    Returns:
        RunMetrics: Accumulated metrics for the streamed run.
    """
    accumulator = RunMetrics()
    file_context = (
        prediction_output.predictions_path.open("w", encoding="utf-8")
        if prediction_output.write_predictions
        else nullcontext(None)
    )
    with (
        file_context as file_obj,
        make_count_progress(
            unit=None if score_progress_hint is None else score_progress_hint.unit,
        ) as progress,
    ):
        score_task = progress.add_task(
            score_stage_description(detector.detector_name),
            total=None if score_progress_hint is None else score_progress_hint.total,
        )
        for sequence, outcome in _iter_prediction_inputs(
            detector=detector,
            sequence_factory=sequence_factory,
            accumulator=accumulator,
        ):
            prediction = outcome.to_prediction_record(sequence)
            abstained = _prediction_is_abstained(outcome)
            if file_obj is not None:
                file_obj.write(
                    msgspec.json.encode(prediction.to_dict()).decode("utf-8"),
                )
                file_obj.write("\n")
            accumulator.record_test(sequence, outcome, abstained=abstained)
            progress.advance(score_task)
            if accumulator.test_sequence_count % _PROGRESS_EVERY == 0:
                logger.info(
                    "Processed %s test sequences for %s detector",
                    accumulator.test_sequence_count,
                    detector.detector_name,
                )
    return accumulator


def _iter_prediction_inputs(
    *,
    detector: ExperimentDetector,
    sequence_factory: Callable[[], Iterator[TemplateSequence]],
    accumulator: RunMetrics,
) -> Iterator[tuple[TemplateSequence, PredictionOutcome]]:
    """Yield scored test sequences while accounting for unscored train items.

    Args:
        detector (ExperimentDetector): Detector being evaluated.
        sequence_factory (Callable[[], Iterator[TemplateSequence]]): Factory
            producing the full sequence stream.
        accumulator (RunMetrics): Metrics accumulator updated for train items.

    Yields:
        (TemplateSequence, PredictionOutcome): Test sequences and their
        prediction outcomes, in dataset order.
    """
    test_sequences = _iter_test_sequences(
        sequence_factory=sequence_factory,
        accumulator=accumulator,
    )
    if isinstance(detector, BatchExperimentDetector):
        yield from detector.predict_all(test_sequences)
        return
    for sequence in test_sequences:
        yield sequence, detector.predict(sequence)


def _iter_test_sequences(
    *,
    sequence_factory: Callable[[], Iterator[TemplateSequence]],
    accumulator: RunMetrics,
) -> Iterator[TemplateSequence]:
    """Yield test sequences while accounting for train sequences inline.

    Args:
        sequence_factory (Callable[[], Iterator[TemplateSequence]]): Factory
            producing the full sequence stream.
        accumulator (RunMetrics): Metrics accumulator updated for skipped train
            sequences.

    Yields:
        TemplateSequence: Test-split sequences in dataset order.
    """
    for sequence in sequence_factory():
        if sequence.split_label is SplitLabel.TRAIN:
            accumulator.record_train(sequence)
            continue
        if sequence.split_label is SplitLabel.IGNORED:
            accumulator.record_ignored(sequence)
            continue
        yield sequence


def _prediction_is_abstained(prediction: PredictionOutcome) -> bool:
    """Return whether a detector-specific prediction outcome abstains.

    Args:
        prediction (PredictionOutcome): Detector output being inspected.

    Returns:
        bool: True when the prediction outcome exposes an abstain hook and it
            reports manual review.
    """
    return isinstance(prediction, AbstainAwarePredictionOutcome) and (
        prediction.is_abstained
    )
