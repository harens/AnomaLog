"""Unsupervised template-frequency sanity baseline."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, ClassVar

import msgspec

from anomalog.parsers.structured.contracts import is_anomalous_label
from anomalog.sequences import SplitLabel
from experiments.models.base import (
    ExperimentDetector,
    ExperimentModelConfig,
    ModelManifest,
    NonNegativeFloat,
    PositiveFloat,
    PredictionOutcome,
    Probability,
    SequenceSummary,
    SingleFitMixin,
)
from experiments.models.event_level_detection import (
    EventLevelDetectionDiagnostics,
    EventLevelDetectionState,
)
from experiments.models.progress import fit_stage_description

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
    """Unsupervised template-frequency sanity baseline.

    The detector counts templates in the train prefix and turns unexpectedly
    rare sequences into high anomaly scores. When the threshold is not fixed by
    config, calibration uses the same eligible training targets as the DeepLog
    event masks, rather than whole-sequence labels. When event labels are
    available, the detector can also report event-level anomaly metrics. If the
    training split does not carry any per-event training signal, the event-level
    threshold reuses the calibrated sequence threshold instead of paying for a
    redundant event-wise calibration pass.
    """  # noqa: DOC601, DOC603

    score_threshold: Annotated[
        NonNegativeFloat | None,
        msgspec.Meta(
            description=(
                "Optional fixed anomaly score threshold. When omitted, the "
                "detector calibrates from eligible training scores only."
            ),
        ),
    ] = None
    calibration_quantile: Annotated[
        Probability,
        msgspec.Meta(
            description=(
                "Training-score quantile used to choose a threshold when "
                "score_threshold is omitted."
            ),
        ),
    ] = 0.95
    smoothing: Annotated[
        PositiveFloat,
        msgspec.Meta(description="Additive smoothing value for template frequencies."),
    ] = 1.0

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
class TemplateFrequencyDetector(SingleFitMixin, ExperimentDetector):
    """Unsupervised detector scoring sequences by train-set template frequency.

    It is a lightweight sanity check for whether simple template statistics
    already separate the labels, not a paper-faithful DeepLog/DeepCASE
    reproduction.

    Attributes:
        detector_name (ClassVar[str]): Stable detector name for manifests/logging.
        configured_score_threshold (float | None): User-provided threshold, if any.
        calibration_quantile (float): Quantile used when calibrating a threshold.
        smoothing (float): Additive smoothing applied to template counts.
        template_counts (Counter[str]): Learned train-set template counts.
        total_events (int): Total number of training events seen.
        score_threshold (float): Effective anomaly threshold after fitting.
        threshold_source (str): Whether the threshold was configured or calibrated.
        event_score_threshold (float): Effective event-level anomaly threshold.
        event_threshold_source (str): Whether the event threshold was configured
            or calibrated.
    """

    detector_name: ClassVar[str] = "template_frequency"
    configured_score_threshold: float | None
    calibration_quantile: float
    smoothing: float
    template_counts: Counter[str] = field(default_factory=Counter)
    total_events: int = 0
    score_threshold: float = 0.0
    threshold_source: str = "configured"
    event_score_threshold: float = 0.0
    event_threshold_source: str = "configured"
    _template_losses: dict[str, float] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _unknown_template_loss: float = field(default=0.0, init=False, repr=False)
    _event_level_state: EventLevelDetectionState = field(
        default_factory=EventLevelDetectionState,
        init=False,
        repr=False,
    )

    def fit(
        self,
        train_sequences: Iterable[TemplateSequence],
        *,
        progress: Progress,
        logger: logging.Logger | None = None,
    ) -> None:
        """Fit template counts from train sequences and calibrate a threshold.

        Args:
            train_sequences (Iterable[TemplateSequence]): Training split
                sequences. Threshold calibration needs at least one sequence
                with eligible training targets when the threshold is not
                configured.
            progress (Progress): Progress reporter.
            logger (logging.Logger | None): Optional logger for fit diagnostics.

        Raises:
            ValueError: If the training split contains zero events.
        """
        self._ensure_unfit(detector_name=self.detector_name)
        counts: Counter[str] = Counter()
        total_events = 0
        del logger
        calibration_templates: list[list[str]] = []
        has_event_level_training_signal = False
        self._event_level_state.reset()
        for sequence in progress.track(
            train_sequences,
            description=fit_stage_description(self.detector_name),
        ):
            eligible_training_templates = _eligible_training_templates(sequence)
            if not eligible_training_templates:
                continue
            counts.update(eligible_training_templates)
            total_events += len(eligible_training_templates)
            if self.configured_score_threshold is None:
                calibration_templates.append(eligible_training_templates)
                if not has_event_level_training_signal and (
                    sequence.training_event_mask is not None
                    or sequence.event_labels is not None
                ):
                    has_event_level_training_signal = True
        if total_events == 0:
            msg = (
                "Cannot fit template_frequency detector with zero eligible "
                "training events."
            )
            raise ValueError(msg)
        self.template_counts = counts
        self.total_events = total_events
        vocab_size = max(len(self.template_counts), 1)
        denominator = self.total_events + (self.smoothing * vocab_size)
        self._template_losses = {
            template: -math.log((count + self.smoothing) / denominator)
            for template, count in self.template_counts.items()
        }
        self._unknown_template_loss = -math.log(self.smoothing / denominator)
        if self.configured_score_threshold is not None:
            self.score_threshold = self.configured_score_threshold
            self.threshold_source = "configured"
            self.event_score_threshold = self.configured_score_threshold
            self.event_threshold_source = "configured"
            self._mark_fit_complete()
            return

        if not calibration_templates:
            msg = (
                "template_frequency calibration requires at least one eligible "
                "training sequence."
            )
            raise ValueError(msg)
        self._calibrate_thresholds(
            calibration_templates=calibration_templates,
            has_event_level_training_signal=has_event_level_training_signal,
        )

    def predict(self, sequence: TemplateSequence) -> PredictionOutcome:
        """Return a prediction record for a sequence.

        Args:
            sequence (TemplateSequence): Sequence to score.

        Returns:
            PredictionOutcome: Predicted label and anomaly score for the sequence.
        """
        pure_test_stream = (
            sequence.split_label is SplitLabel.TEST
            and sequence.evaluation_event_mask is None
        )
        if sequence.event_labels is None:
            if pure_test_stream:
                score = self._score_sequence_templates(sequence.events)
            else:
                score = self.score(sequence)
        elif pure_test_stream:
            score = self._score_sequence_and_event_metrics_from_events(
                sequence.events,
                sequence.event_labels,
            )
        else:
            score = self._score_sequence_and_event_metrics(sequence)
        predicted_label = int(score > self.score_threshold)
        return PredictionOutcome(
            predicted_label=predicted_label,
            score=score,
        )

    def model_manifest(
        self,
        *,
        sequence_summary: SequenceSummary,
    ) -> TemplateFrequencyManifest:
        """Return serialisable detector metadata.

        Args:
            sequence_summary (SequenceSummary): Aggregate split and label counts
                for the run.

        Returns:
            TemplateFrequencyManifest: Serialisable template-frequency manifest
                for the run.
        """
        return TemplateFrequencyManifest.from_sequence_summary(
            detector=self.detector_name,
            sequence_summary=sequence_summary,
            score_threshold=self.score_threshold,
            threshold_source=self.threshold_source,
            event_score_threshold=self.event_score_threshold,
            event_threshold_source=self.event_threshold_source,
            calibration_quantile=self.calibration_quantile,
            smoothing=self.smoothing,
            train_event_count=self.total_events,
            train_template_vocabulary=len(self.template_counts),
        )

    def score(self, sequence: TemplateSequence) -> float:
        """Return the mean negative log-probability for a sequence.

        Args:
            sequence (TemplateSequence): Sequence to score.

        Returns:
            float: Mean negative log-probability under the learned template model.
        """
        if (
            sequence.split_label is SplitLabel.TEST
            and sequence.evaluation_event_mask is None
        ):
            return self._score_sequence_templates(sequence.events)
        templates = _eligible_evaluation_templates(sequence)
        return self._score_templates(templates)

    def _score_template(self, template: str) -> float:
        """Return the negative log-probability for one template.

        Args:
            template (str): Template to score.

        Returns:
            float: Negative log-probability under the learned template model.
        """
        return self._template_losses.get(template, self._unknown_template_loss)

    def _score_templates(self, templates: list[str]) -> float:
        """Return the mean negative log-probability for a template list.

        Args:
            templates (list[str]): Templates to score.

        Returns:
            float: Mean negative log-probability across the templates.
        """
        if not templates:
            return 0.0
        loss_sum = 0.0
        for template in templates:
            loss_sum += self._score_template(template)
        return loss_sum / len(templates)

    def _score_sequence_templates(
        self,
        events: list[tuple[str, list[str], int | None]],
    ) -> float:
        """Return the mean loss for a pure test sequence without mask setup.

        Args:
            events (list[tuple[str, list[str], int | None]]): Sequence events
                already filtered to the pure test path.

        Returns:
            float: Mean negative log-probability across the sequence events.
        """
        if not events:
            return 0.0
        loss_sum = 0.0
        for template, _, _ in events:
            loss_sum += self._score_template(template)
        return loss_sum / len(events)

    def _score_sequence_and_event_metrics_from_events(
        self,
        events: list[tuple[str, list[str], int | None]],
        event_labels: tuple[int | None, ...],
    ) -> float:
        """Return a sequence score while updating event-level metrics.

        Args:
            events (list[tuple[str, list[str], int | None]]): Sequence events
                to score.
            event_labels (tuple[int | None, ...]): Ground-truth event labels
                aligned with ``events``.

        Returns:
            float: Mean negative log-probability across the sequence events.
        """
        if not events:
            return 0.0
        loss_sum = 0.0
        for (template, _, _), actual_label in zip(events, event_labels, strict=True):
            loss = self._score_template(template)
            loss_sum += loss
            if actual_label is None:
                continue
            predicted_label = int(loss > self.event_score_threshold)
            self._event_level_state.record(
                actual_label=actual_label,
                predicted_label=predicted_label,
            )
        return loss_sum / len(events)

    def _score_sequence_and_event_metrics(
        self,
        sequence: TemplateSequence,
    ) -> float:
        """Return a sequence score while updating event-level metrics.

        Args:
            sequence (TemplateSequence): Sequence to score.

        Returns:
            float: Mean sequence score for the sequence.
        """
        event_labels = sequence.event_labels
        if event_labels is None:
            return self.score(sequence)
        if (
            sequence.split_label is SplitLabel.TEST
            and sequence.evaluation_event_mask is None
        ):
            return self._score_sequence_and_event_metrics_from_events(
                sequence.events,
                event_labels,
            )
        eligible_event_mask = _evaluation_event_mask(sequence)
        event_count = len(sequence.events)
        if event_count == 0:
            return 0.0
        loss_sum = 0.0
        for (template, _, _), actual_label, is_eligible in zip(
            sequence.events,
            event_labels,
            eligible_event_mask,
            strict=False,
        ):
            loss = self._score_template(template)
            loss_sum += loss
            if not is_eligible or actual_label is None:
                continue
            predicted_label = int(loss > self.event_score_threshold)
            self._event_level_state.record(
                actual_label=actual_label,
                predicted_label=predicted_label,
            )
        return loss_sum / event_count

    def _calibrate_thresholds(
        self,
        *,
        calibration_templates: list[list[str]],
        has_event_level_training_signal: bool,
    ) -> None:
        """Calibrate sequence and event thresholds from cached train scores.

        Args:
            calibration_templates (list[list[str]]): Eligible training
                templates grouped by sequence.
            has_event_level_training_signal (bool): Whether the training split
                carries explicit per-event supervision or masks.
        """
        self.score_threshold = _quantile(
            sorted(
                self._score_templates(templates) for templates in calibration_templates
            ),
            self.calibration_quantile,
        )
        self.threshold_source = "train_score_quantile"
        if not has_event_level_training_signal:
            self.event_score_threshold = self.score_threshold
            self.event_threshold_source = self.threshold_source
            self._mark_fit_complete()
            return

        calibration_event_scores = self._calibration_event_scores(
            calibration_templates,
        )
        self.event_score_threshold = _quantile(
            calibration_event_scores,
            self.calibration_quantile,
        )
        self.event_threshold_source = "train_event_score_quantile"
        self._mark_fit_complete()

    def _calibration_event_scores(
        self,
        calibration_templates: list[list[str]],
    ) -> list[float]:
        """Return sorted per-event losses for calibrated event thresholds.

        Args:
            calibration_templates (list[list[str]]): Eligible training
                templates grouped by sequence.

        Returns:
            list[float]: Sorted per-event losses used for event calibration.
        """
        calibration_event_scores: list[float] = []
        for templates in calibration_templates:
            calibration_event_scores.extend(
                self._score_template(template) for template in templates
            )
        calibration_event_scores.sort()
        return calibration_event_scores

    def _event_level_state_snapshot(self) -> EventLevelDetectionDiagnostics | None:
        """Return the latest event-level detection diagnostics.

        Returns:
            EventLevelDetectionDiagnostics | None: Latest event-level
                diagnostics, or `None` when no events have been scored.
        """
        return self._event_level_state.snapshot(task="event_level_detection")

    def run_metrics(
        self,
        *,
        run_metrics: dict[str, int | float | dict[int, int]],
    ) -> object | None:
        """Return event-level diagnostics for the latest scored run.

        Args:
            run_metrics (dict[str, int | float | dict[int, int]]): Metric values
                collected during the run.

        Returns:
            object | None: Event-level diagnostics wrapper, or `None` when no
                event-level state exists.
        """
        del run_metrics
        snapshot = self._event_level_state_snapshot()
        return None if snapshot is None else {"event_level_detection": snapshot}


class TemplateFrequencyManifest(ModelManifest, frozen=True):
    """Serialisable template-frequency detector metadata.

    Attributes:
        score_threshold (float): Effective anomaly threshold after fitting.
        threshold_source (str): Whether the threshold was configured or calibrated.
        event_score_threshold (float): Effective event-level anomaly threshold.
        event_threshold_source (str): Whether the event threshold was configured
            or calibrated.
        calibration_quantile (float): Quantile used during calibration.
        smoothing (float): Additive smoothing applied to template counts.
        train_event_count (int): Total number of training events seen.
        train_template_vocabulary (int): Learned template vocabulary size.
    """

    score_threshold: float
    threshold_source: str
    event_score_threshold: float
    event_threshold_source: str
    calibration_quantile: float
    smoothing: float
    train_event_count: int
    train_template_vocabulary: int


def _eligible_training_templates(sequence: TemplateSequence) -> list[str]:
    """Return the templates eligible for template-frequency fitting.

    Args:
        sequence (TemplateSequence): Training sequence to inspect.

    Returns:
        list[str]: Templates eligible for counting and calibration.
    """
    if sequence.training_event_mask is None:
        if is_anomalous_label(sequence.label):
            return []
        return [template for template, _, _ in sequence.events]
    return [
        template
        for (template, _, _), is_eligible in zip(
            sequence.events,
            sequence.training_event_mask,
            strict=True,
        )
        if is_eligible
    ]


def _eligible_evaluation_templates(sequence: TemplateSequence) -> list[str]:
    """Return the templates eligible for template-frequency scoring.

    Args:
        sequence (TemplateSequence): Sequence to score.

    Returns:
        list[str]: Templates eligible for anomaly scoring.
    """
    if sequence.evaluation_event_mask is None:
        if sequence.split_label is not SplitLabel.TEST:
            return []
        return [template for template, _, _ in sequence.events]
    return [
        template
        for (template, _, _), is_eligible in zip(
            sequence.events,
            sequence.evaluation_event_mask,
            strict=True,
        )
        if is_eligible
    ]


def _evaluation_event_mask(sequence: TemplateSequence) -> tuple[bool, ...]:
    """Return the evaluation mask, preserving the no-mask fast path.

    Args:
        sequence (TemplateSequence): Sequence whose scoring mask is needed.

    Returns:
        tuple[bool, ...]: Explicit or derived per-event evaluation mask.
    """
    if sequence.evaluation_event_mask is not None:
        return sequence.evaluation_event_mask
    if sequence.split_label is not SplitLabel.TEST:
        return ()
    return tuple(True for _ in sequence.events)


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
