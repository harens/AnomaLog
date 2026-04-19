"""Template-frequency baseline detector."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from experiments import ConfigError
from experiments.models.base import (
    ExperimentDetector,
    ExperimentModelConfig,
    ModelManifest,
    PredictionOutcome,
    SequenceSummary,
)

if TYPE_CHECKING:
    import logging
    from collections.abc import Iterable

    from rich.progress import Progress

    from anomalog.sequences import TemplateSequence


class TemplateFrequencyModelConfig(
    ExperimentModelConfig,
    tag="template_frequency",
    frozen=True,
):
    """Baseline detector using template frequencies from train sequences."""

    score_threshold: float | None = None
    calibration_quantile: float = 0.95
    smoothing: float = 1.0

    def __post_init__(self) -> None:
        """Validate detector-specific model settings.

        Raises:
            ConfigError: If detector-specific settings are invalid.
        """
        if self.score_threshold is not None and self.score_threshold < 0:
            msg = "model.score_threshold must be non-negative."
            raise ConfigError(msg)
        if not 0.0 <= self.calibration_quantile <= 1.0:
            msg = "model.calibration_quantile must be between 0.0 and 1.0."
            raise ConfigError(msg)
        if self.smoothing <= 0:
            msg = "model.smoothing must be positive."
            raise ConfigError(msg)

    def build_detector(self) -> TemplateFrequencyDetector:
        """Construct the configured template-frequency detector.

        Returns:
            TemplateFrequencyDetector: Configured detector instance.
        """
        return TemplateFrequencyDetector(
            configured_score_threshold=self.score_threshold,
            calibration_quantile=self.calibration_quantile,
            smoothing=self.smoothing,
        )


@dataclass(slots=True)
class TemplateFrequencyDetector(ExperimentDetector):
    """Baseline detector scoring sequences by train-set template frequencies."""

    detector_name: ClassVar[str] = "template_frequency"
    configured_score_threshold: float | None
    calibration_quantile: float
    smoothing: float
    template_counts: Counter[str] = field(default_factory=Counter)
    total_events: int = 0
    score_threshold: float = 0.0
    threshold_source: str = "configured"

    def fit(
        self,
        train_sequences: Iterable[TemplateSequence],
        *,
        progress: Progress,
        logger: logging.Logger | None = None,
    ) -> None:
        """Fit template counts from train sequences.

        Args:
            train_sequences (Iterable[TemplateSequence]): Training split
                sequences.
            progress (Progress): Progress reporter.
            logger (logging.Logger | None): Optional logger for fit diagnostics.

        Raises:
            ValueError: If the training split contains zero events.
        """
        counts: Counter[str] = Counter()
        total_events = 0
        del logger
        calibration_sequences: list[TemplateSequence] = []
        all_sequences: list[TemplateSequence] | None = (
            [] if self.configured_score_threshold is None else None
        )
        for sequence in progress.track(
            train_sequences,
            description="Fitting template_frequency sequences",
        ):
            counts.update(sequence.templates)
            total_events += len(sequence.templates)
            if all_sequences is not None:
                all_sequences.append(sequence)
                if sequence.label == 0:
                    calibration_sequences.append(sequence)
        if total_events == 0:
            msg = "Cannot fit template_frequency detector with zero train events."
            raise ValueError(msg)
        self.template_counts = counts
        self.total_events = total_events
        if self.configured_score_threshold is not None:
            self.score_threshold = self.configured_score_threshold
            self.threshold_source = "configured"
            return

        if not calibration_sequences:
            if all_sequences is None:
                msg = "template_frequency calibration requires replayable train data."
                raise ValueError(msg)
            calibration_sequences = all_sequences
        calibration_scores = sorted(
            self.score(sequence) for sequence in calibration_sequences
        )
        self.score_threshold = _quantile(
            calibration_scores,
            self.calibration_quantile,
        )
        self.threshold_source = "train_score_quantile"

    def predict(self, sequence: TemplateSequence) -> PredictionOutcome:
        """Return a prediction record for a sequence."""
        score = self.score(sequence)
        predicted_label = int(score > self.score_threshold)
        return PredictionOutcome(
            predicted_label=predicted_label,
            score=score,
        )

    def model_manifest(self, *, sequence_summary: SequenceSummary) -> ModelManifest:
        """Return serializable detector metadata."""
        return TemplateFrequencyManifest.from_sequence_summary(
            detector=self.detector_name,
            sequence_summary=sequence_summary,
            score_threshold=self.score_threshold,
            threshold_source=self.threshold_source,
            calibration_quantile=self.calibration_quantile,
            smoothing=self.smoothing,
            train_event_count=self.total_events,
            train_template_vocabulary=len(self.template_counts),
        )

    def score(self, sequence: TemplateSequence) -> float:
        """Return the mean negative log-probability for a sequence."""
        if not sequence.templates:
            return 0.0
        vocab_size = max(len(self.template_counts), 1)
        denominator = self.total_events + (self.smoothing * vocab_size)
        loss_sum = 0.0
        for template in sequence.templates:
            numerator = self.template_counts.get(template, 0) + self.smoothing
            probability = numerator / denominator
            loss_sum += -math.log(probability)
        return loss_sum / len(sequence.templates)


class TemplateFrequencyManifest(ModelManifest, frozen=True):
    """Serializable template-frequency detector metadata."""

    score_threshold: float
    threshold_source: str
    calibration_quantile: float
    smoothing: float
    train_event_count: int
    train_template_vocabulary: int


def _quantile(sorted_values: list[float], q: float) -> float:
    """Return the inclusive quantile from a pre-sorted value list.

    Args:
        sorted_values (list[float]): Sorted values to sample from.
        q (float): Inclusive quantile in the range `[0.0, 1.0]`.

    Returns:
        float: Quantile value from the sorted input list.

    Raises:
        ValueError: If `sorted_values` is empty.
    """
    if not sorted_values:
        msg = "Cannot compute a quantile from an empty score list."
        raise ValueError(msg)
    index = min(math.ceil(q * len(sorted_values)) - 1, len(sorted_values) - 1)
    return sorted_values[max(index, 0)]
