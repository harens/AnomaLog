"""Normal-only Markov n-gram sanity baseline over template transitions."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, ClassVar

import msgspec

from anomalog.parsers.structured.contracts import is_anomalous_label
from experiments.models.base import (
    ExperimentDetector,
    ExperimentModelConfig,
    ModelManifest,
    NonNegativeFloat,
    PositiveFloat,
    PositiveInt,
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
from experiments.models.sequence_masks import (
    evaluation_event_index_mask,
    training_event_index_mask,
)

if TYPE_CHECKING:
    import logging
    from collections.abc import Iterable, Iterator

    from rich.progress import Progress

    from anomalog.sequences import TemplateSequence


class MarkovModelConfig(
    ExperimentModelConfig,
    tag="markov",
    frozen=True,
):
    """Template transition baseline that respects DeepLog-style eligibility masks.

    The detector still reports the original sequence-level score for legacy
    comparisons, but it can also surface event-level detection when the dataset
    carries per-event labels.
    """  # noqa: DOC601, DOC603

    order: Annotated[
        PositiveInt,
        msgspec.Meta(
            description=(
                "Number of prior templates used as the Markov context. "
                "An order of 1 yields a standard first-order transition model."
            ),
        ),
    ] = 1
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
        msgspec.Meta(
            description="Additive smoothing value for transition probabilities.",
        ),
    ] = 1.0

    def build_detector(self) -> MarkovDetector:
        """Construct the configured Markov detector.

        Returns:
            MarkovDetector: Configured detector instance.
        """
        return MarkovDetector(
            configured_score_threshold=self.score_threshold,
            calibration_quantile=self.calibration_quantile,
            smoothing=self.smoothing,
            order=self.order,
        )


@dataclass(slots=True)
class MarkovDetector(SingleFitMixin, ExperimentDetector):
    """Score template sequences by masked transition likelihood.

    The detector learns an n-gram transition model from training sequences that
    still carry eligible targets under the DeepLog masking contract. That makes
    it a deliberately simple sequence-order comparator for DeepLog-style
    experiments rather than a direct competitor to DeepCASE's
    workload-reduction metrics.

    Attributes:
        detector_name (ClassVar[str]): Stable detector name for manifests/logging.
        configured_score_threshold (float | None): User-provided threshold, if any.
        calibration_quantile (float): Quantile used when calibrating a threshold.
        smoothing (float): Additive smoothing applied to transition counts.
        order (int): Number of prior templates used as context.
        transition_counts_by_context (dict[tuple[str, ...], Counter[str]]):
            Learned next-template counts per context.
        context_counts (Counter[tuple[str, ...]]): Learned context counts.
        normal_sequence_count (int): Number of normal train sequences used for fitting.
        normal_transition_count (int): Number of scored normal transitions
            seen during fit.
        score_threshold (float): Effective anomaly threshold after fitting.
        threshold_source (str): Whether the threshold was configured or calibrated.
        event_score_threshold (float): Effective event-level anomaly threshold.
        event_threshold_source (str): Whether the event threshold was configured
            or calibrated.
        normal_template_vocabulary (set[str]): Templates observed in normal training.
    """

    detector_name: ClassVar[str] = "markov"
    configured_score_threshold: float | None
    calibration_quantile: float
    smoothing: float
    order: int
    transition_counts_by_context: dict[tuple[str, ...], Counter[str]] = field(
        default_factory=dict,
    )
    context_counts: Counter[tuple[str, ...]] = field(default_factory=Counter)
    normal_sequence_count: int = 0
    normal_transition_count: int = 0
    score_threshold: float = 0.0
    threshold_source: str = "configured"
    event_score_threshold: float = 0.0
    event_threshold_source: str = "configured"
    normal_template_vocabulary: set[str] = field(default_factory=set)
    _stream_context_templates: list[str] = field(
        default_factory=list,
        init=False,
        repr=False,
    )
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
        """Fit the transition model from eligible training sequences.

        Args:
            train_sequences (Iterable[TemplateSequence]): Training split
                sequences. Only sequences with eligible training targets are
                used for fitting.
            progress (Progress): Progress reporter.
            logger (logging.Logger | None): Optional logger for fit diagnostics.

        Raises:
            ValueError: If the training split contains no eligible sequences.
        """
        del logger
        self._ensure_unfit(detector_name=self.detector_name)
        calibration_sequences: list[_PreparedMarkovCalibrationSequence] = []
        stream_context_templates: list[str] = []
        self._event_level_state.reset()
        self._stream_context_templates = []
        for sequence in progress.track(
            train_sequences,
            description=fit_stage_description(self.detector_name),
        ):
            prepared_sequence, stream_context_templates = (
                self._prepare_training_sequence(
                    sequence=sequence,
                    stream_context_templates=stream_context_templates,
                )
            )
            if prepared_sequence is None:
                continue
            self.normal_sequence_count += 1
            self.normal_template_vocabulary.update(sequence.templates)
            self.normal_transition_count += self._learn_sequence(
                sequence,
                eligible_target_indexes=prepared_sequence.eligible_target_indexes,
                prefix_templates=prepared_sequence.prefix_templates,
            )
            calibration_sequences.append(prepared_sequence)

        if not calibration_sequences:
            msg = "markov detector requires at least one normal training sequence."
            raise ValueError(msg)

        if self.configured_score_threshold is not None:
            self.score_threshold = self.configured_score_threshold
            self.threshold_source = "configured"
            self.event_score_threshold = self.configured_score_threshold
            self.event_threshold_source = "configured"
            self._stream_context_templates = stream_context_templates
            self._mark_fit_complete()
            return

        calibration_scores = []
        calibration_event_scores = []
        for prepared_sequence in calibration_sequences:
            calibration_scores.append(
                self._score_sequence(
                    prepared_sequence.sequence,
                    eligible_target_indexes=prepared_sequence.eligible_target_indexes,
                    prefix_templates=prepared_sequence.prefix_templates,
                ),
            )
            for event_index, context, next_template in _sequence_transitions(
                prepared_sequence.sequence.templates,
                self.order,
                prefix_templates=prepared_sequence.prefix_templates,
            ):
                if event_index not in prepared_sequence.eligible_target_indexes:
                    continue
                calibration_event_scores.append(
                    self._transition_score(context, next_template),
                )
        self.score_threshold = _quantile(
            sorted(calibration_scores),
            self.calibration_quantile,
        )
        self.threshold_source = "train_score_quantile"
        self.event_score_threshold = _quantile(
            sorted(calibration_event_scores),
            self.calibration_quantile,
        )
        self.event_threshold_source = "train_event_score_quantile"
        self._stream_context_templates = stream_context_templates
        self._mark_fit_complete()

    def predict(self, sequence: TemplateSequence) -> PredictionOutcome:
        """Return the predicted label and anomaly score for one sequence.

        Args:
            sequence (TemplateSequence): Sequence to score.

        Returns:
            PredictionOutcome: Predicted label and anomaly score.
        """
        prefix_templates = (
            self._stream_context_templates if sequence.continuous_context else []
        )
        score = self._score_sequence(
            sequence,
            eligible_target_indexes=set(evaluation_event_index_mask(sequence)),
            prefix_templates=prefix_templates,
        )
        self._record_event_level_predictions(
            sequence,
            prefix_templates=prefix_templates,
        )
        self._advance_stream_context(
            sequence=sequence,
            prefix_templates=prefix_templates,
        )
        predicted_label = int(score > self.score_threshold)
        return PredictionOutcome(predicted_label=predicted_label, score=score)

    def model_manifest(self, *, sequence_summary: SequenceSummary) -> MarkovManifest:
        """Return serialisable detector metadata.

        Args:
            sequence_summary (SequenceSummary): Aggregate split and label counts
                for the run.

        Returns:
            MarkovManifest: Serialisable Markov manifest for the run.
        """
        return MarkovManifest.from_sequence_summary(
            detector=self.detector_name,
            sequence_summary=sequence_summary,
            order=self.order,
            score_threshold=self.score_threshold,
            threshold_source=self.threshold_source,
            event_score_threshold=self.event_score_threshold,
            event_threshold_source=self.event_threshold_source,
            calibration_quantile=self.calibration_quantile,
            smoothing=self.smoothing,
            normal_sequence_count=self.normal_sequence_count,
            normal_transition_count=self.normal_transition_count,
            normal_context_vocabulary=len(self.context_counts),
            normal_template_vocabulary=len(self.normal_template_vocabulary),
        )

    def score(self, sequence: TemplateSequence) -> float:
        """Return the mean negative log transition probability for a sequence.

        Args:
            sequence (TemplateSequence): Sequence to score.

        Returns:
            float: Mean negative log transition probability under the learned
            Markov model.
        """
        return self._score_sequence(
            sequence,
            eligible_target_indexes=set(evaluation_event_index_mask(sequence)),
            prefix_templates=(
                self._stream_context_templates if sequence.continuous_context else []
            ),
        )

    def _score_sequence(
        self,
        sequence: TemplateSequence,
        *,
        eligible_target_indexes: set[int] | None,
        prefix_templates: list[str] | None,
    ) -> float:
        """Return the masked mean negative log transition probability.

        Args:
            sequence (TemplateSequence): Sequence to score.
            eligible_target_indexes (set[int] | None): Optional target indexes
                allowed to contribute to the score.
            prefix_templates (list[str] | None): Optional trailing context from
                the previous chronological chunk.

        Returns:
            float: Mean negative log transition probability over the eligible
            transitions in the sequence.
        """
        transitions = [
            (context, next_template)
            for target_index, context, next_template in _sequence_transitions(
                sequence.templates,
                self.order,
                prefix_templates=prefix_templates,
            )
            if eligible_target_indexes is None
            or target_index in eligible_target_indexes
        ]
        if not transitions:
            return 0.0
        vocabulary_size = max(len(self.normal_template_vocabulary), 1)
        loss_sum = 0.0
        for context, next_template in transitions:
            numerator = (
                self.transition_counts_by_context.get(context, Counter()).get(
                    next_template,
                    0,
                )
                + self.smoothing
            )
            denominator = self.context_counts.get(context, 0) + (
                self.smoothing * vocabulary_size
            )
            loss_sum += -math.log(numerator / denominator)
        return loss_sum / len(transitions)

    def _learn_sequence(
        self,
        sequence: TemplateSequence,
        *,
        eligible_target_indexes: set[int],
        prefix_templates: list[str] | None,
    ) -> int:
        """Update transition counts from one normal training sequence.

        Args:
            sequence (TemplateSequence): Training sequence to learn from.
            eligible_target_indexes (set[int]): Target indexes allowed to
                contribute transition counts.
            prefix_templates (list[str] | None): Trailing template history from
                the previous chronological chunk, when one is available.

        Returns:
            int: Number of learned transitions contributed by the sequence.
        """
        transition_count = 0
        for target_index, context, next_template in _sequence_transitions(
            sequence.templates,
            self.order,
            prefix_templates=prefix_templates,
        ):
            if target_index not in eligible_target_indexes:
                continue
            context_counts = self.transition_counts_by_context.setdefault(
                context,
                Counter(),
            )
            context_counts[next_template] += 1
            self.context_counts[context] += 1
            transition_count += 1
        return transition_count

    def _prepare_training_sequence(
        self,
        *,
        sequence: TemplateSequence,
        stream_context_templates: list[str],
    ) -> tuple[_PreparedMarkovCalibrationSequence | None, list[str]]:
        """Return the training payload for one sequence and the next stream tail."""
        eligible_target_indexes = set(training_event_index_mask(sequence))
        prefix_templates = (
            stream_context_templates if sequence.continuous_context else []
        )
        if is_anomalous_label(sequence.label):
            return None, []
        if not eligible_target_indexes:
            if sequence.continuous_context:
                return (
                    None,
                    _sequence_tail(
                        sequence.templates,
                        self.order,
                        prefix_templates=prefix_templates,
                    ),
                )
            return None, []
        prepared_sequence = _PreparedMarkovCalibrationSequence(
            sequence=sequence,
            prefix_templates=list(prefix_templates),
            eligible_target_indexes=eligible_target_indexes,
        )
        if sequence.continuous_context:
            return (
                prepared_sequence,
                _sequence_tail(
                    sequence.templates,
                    self.order,
                    prefix_templates=prefix_templates,
                ),
            )
        return prepared_sequence, []

    def _record_event_level_predictions(
        self,
        sequence: TemplateSequence,
        *,
        prefix_templates: list[str] | None,
    ) -> None:
        """Accumulate event-level confusion counts for one scored sequence.

        Args:
            sequence (TemplateSequence): Sequence whose labelled events should
                be scored.
            prefix_templates (list[str] | None): Optional trailing history from
                the previous chronological chunk.
        """
        if sequence.event_labels is None:
            return
        eligible_target_indexes = set(evaluation_event_index_mask(sequence))
        if not eligible_target_indexes:
            return
        for target_index, context, next_template in _sequence_transitions(
            sequence.templates,
            self.order,
            prefix_templates=prefix_templates,
        ):
            if target_index not in eligible_target_indexes:
                continue
            if target_index >= len(sequence.event_labels):
                continue
            actual_label = sequence.event_labels[target_index]
            if actual_label is None:
                continue
            predicted_label = int(
                self._transition_score(context, next_template)
                > self.event_score_threshold,
            )
            self._event_level_state.record(
                actual_label=actual_label,
                predicted_label=predicted_label,
            )

    def _transition_score(self, context: tuple[str, ...], next_template: str) -> float:
        """Return the negative log transition probability for one event.

        Args:
            context (tuple[str, ...]): Observed Markov context.
            next_template (str): Observed next template.

        Returns:
            float: Negative log transition probability under the fitted model.
        """
        vocabulary_size = max(len(self.normal_template_vocabulary), 1)
        numerator = (
            self.transition_counts_by_context.get(context, Counter()).get(
                next_template,
                0,
            )
            + self.smoothing
        )
        denominator = self.context_counts.get(context, 0) + (
            self.smoothing * vocabulary_size
        )
        return -math.log(numerator / denominator)

    def _advance_stream_context(
        self,
        *,
        sequence: TemplateSequence,
        prefix_templates: list[str] | None,
    ) -> None:
        """Carry forward the trailing history for chronological chunking.

        Args:
            sequence (TemplateSequence): Sequence that has just been scored.
            prefix_templates (list[str] | None): Optional trailing history from
                the previous chronological chunk.
        """
        if sequence.continuous_context:
            self._stream_context_templates = _sequence_tail(
                sequence.templates,
                self.order,
                prefix_templates=prefix_templates,
            )
            return
        self._stream_context_templates = []

    def _event_level_state_snapshot(self) -> EventLevelDetectionDiagnostics | None:
        """Return the latest event-level detection diagnostics.

        Returns:
            EventLevelDetectionDiagnostics | None: Latest accumulated event
            metrics, or `None` when no labelled events have been scored yet.
        """
        return self._event_level_state.snapshot(task="event_level_detection")

    def run_metrics(
        self,
        *,
        run_metrics: dict[str, int | float | dict[int, int]],
    ) -> object | None:
        """Return event-level diagnostics for the latest scored run.

        Args:
            run_metrics (dict[str, int | float | dict[int, int]]): Shared run
                metrics payload, ignored by the Markov detector.

        Returns:
            object | None: Event-level diagnostics payload, or `None` when no
            labelled events were scored.
        """
        del run_metrics
        snapshot = self._event_level_state_snapshot()
        return None if snapshot is None else {"event_level_detection": snapshot}


class MarkovManifest(ModelManifest, frozen=True):
    """Serialisable Markov detector metadata.

    Attributes:
        order (int): Number of prior templates used as context.
        score_threshold (float): Effective anomaly threshold after fitting.
        threshold_source (str): Whether the threshold was configured or calibrated.
        event_score_threshold (float): Effective event-level anomaly threshold.
        event_threshold_source (str): Whether the event threshold was configured
            or calibrated.
        calibration_quantile (float): Quantile used during calibration.
        smoothing (float): Additive smoothing applied to transition counts.
        normal_sequence_count (int): Number of training sequences that
            contributed eligible targets during fit.
        normal_transition_count (int): Number of eligible transitions seen
            during fit.
        normal_context_vocabulary (int): Number of learned transition contexts.
        normal_template_vocabulary (int): Number of unique templates seen in
            normal training.
    """

    order: int
    score_threshold: float
    threshold_source: str
    event_score_threshold: float
    event_threshold_source: str
    calibration_quantile: float
    smoothing: float
    normal_sequence_count: int
    normal_transition_count: int
    normal_context_vocabulary: int
    normal_template_vocabulary: int


@dataclass(slots=True)
class _PreparedMarkovCalibrationSequence:
    """Cached training payload for one normal Markov calibration sequence."""

    sequence: TemplateSequence
    prefix_templates: list[str] | None
    eligible_target_indexes: set[int]


def _sequence_transitions(
    templates: list[str],
    order: int,
    *,
    prefix_templates: list[str] | None = None,
) -> Iterator[tuple[int, tuple[str, ...], str]]:
    """Yield fixed-width transition contexts and next templates.

    Args:
        templates (list[str]): Template ids from one sequence.
        order (int): Number of prior templates to use as context.
        prefix_templates (list[str] | None): Optional trailing history carried
            across a chronological chunk boundary.

    Yields:
        (int, tuple[str, ...], str): Target index within the current sequence,
        context tuple, and next template.
    """
    combined_templates = [] if prefix_templates is None else list(prefix_templates)
    combined_templates.extend(templates)
    prefix_length = 0 if prefix_templates is None else len(prefix_templates)
    if len(combined_templates) <= order:
        return
    for index in range(len(templates)):
        target_index = prefix_length + index
        if target_index < order:
            continue
        start_index = target_index - order
        yield (
            index,
            tuple(combined_templates[start_index:target_index]),
            templates[index],
        )


def _sequence_tail(
    templates: list[str],
    order: int,
    *,
    prefix_templates: list[str] | None = None,
) -> list[str]:
    """Return the trailing context needed for the next chronological chunk.

    Args:
        templates (list[str]): Sequence templates from the completed chunk.
        order (int): Markov order used to determine the trailing history.
        prefix_templates (list[str] | None): Optional prior trailing history
            that should be preserved when a stream is chunked.

    Returns:
        list[str]: Trailing templates to carry into the next chunk.
    """
    combined_templates = [] if prefix_templates is None else list(prefix_templates)
    combined_templates.extend(templates)
    if order <= 0:
        return []
    return combined_templates[-order:]


def _quantile(sorted_values: list[float], q: float) -> float:
    """Return the inclusive quantile from a pre-sorted value list.

    Args:
        sorted_values (list[float]): Values sorted in ascending order.
        q (float): Desired quantile in the inclusive range `[0, 1]`.

    Returns:
        float: Inclusive quantile value.

    Raises:
        ValueError: If `sorted_values` is empty.
    """
    if not sorted_values:
        msg = "Cannot compute a quantile from an empty score list."
        raise ValueError(msg)
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    position = q * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    lower_weight = upper_index - position
    upper_weight = position - lower_index
    return (
        sorted_values[lower_index] * lower_weight
        + sorted_values[upper_index] * upper_weight
    )
