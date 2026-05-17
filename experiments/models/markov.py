"""Normal-only Markov n-gram sanity baseline over template transitions."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, ClassVar

import msgspec

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
    """Template transition baseline that respects DeepLog-style eligibility masks."""  # noqa: DOC601, DOC603

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
    normal_template_vocabulary: set[str] = field(default_factory=set)

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
        calibration_sequences: list[TemplateSequence] = []
        for sequence in progress.track(
            train_sequences,
            description=fit_stage_description(self.detector_name),
        ):
            eligible_target_indexes = training_event_index_mask(sequence)
            if not eligible_target_indexes:
                continue
            calibration_sequences.append(sequence)
            self.normal_sequence_count += 1
            self.normal_template_vocabulary.update(sequence.templates)
            self.normal_transition_count += self._learn_sequence(
                sequence,
                eligible_target_indexes=eligible_target_indexes,
            )

        if not calibration_sequences:
            msg = "markov detector requires at least one normal training sequence."
            raise ValueError(msg)

        if self.configured_score_threshold is not None:
            self.score_threshold = self.configured_score_threshold
            self.threshold_source = "configured"
            self._mark_fit_complete()
            return

        calibration_scores = sorted(
            self._score_sequence(
                sequence,
                eligible_target_indexes=training_event_index_mask(sequence),
            )
            for sequence in calibration_sequences
        )
        self.score_threshold = _quantile(calibration_scores, self.calibration_quantile)
        self.threshold_source = "train_score_quantile"
        self._mark_fit_complete()

    def predict(self, sequence: TemplateSequence) -> PredictionOutcome:
        """Return the predicted label and anomaly score for one sequence.

        Args:
            sequence (TemplateSequence): Sequence to score.

        Returns:
            PredictionOutcome: Predicted label and anomaly score.
        """
        score = self.score(sequence)
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
            eligible_target_indexes=evaluation_event_index_mask(sequence),
        )

    def _score_sequence(
        self,
        sequence: TemplateSequence,
        *,
        eligible_target_indexes: list[int] | None,
    ) -> float:
        """Return the masked mean negative log transition probability."""
        eligible_indexes = (
            set(eligible_target_indexes)
            if eligible_target_indexes is not None
            else None
        )
        transitions = [
            (context, next_template)
            for target_index, context, next_template in _sequence_transitions(
                sequence.templates,
                self.order,
            )
            if eligible_indexes is None or target_index in eligible_indexes
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
        eligible_target_indexes: list[int],
    ) -> int:
        """Update transition counts from one normal training sequence.

        Args:
            sequence (TemplateSequence): Training sequence to learn from.
            eligible_target_indexes (list[int]): Target indexes allowed to
                contribute transition counts.

        Returns:
            int: Number of learned transitions contributed by the sequence.
        """
        eligible_indexes = set(eligible_target_indexes)
        transition_count = 0
        for target_index, context, next_template in _sequence_transitions(
            sequence.templates,
            self.order,
        ):
            if target_index not in eligible_indexes:
                continue
            context_counts = self.transition_counts_by_context.setdefault(
                context,
                Counter(),
            )
            context_counts[next_template] += 1
            self.context_counts[context] += 1
            transition_count += 1
        return transition_count


class MarkovManifest(ModelManifest, frozen=True):
    """Serialisable Markov detector metadata.

    Attributes:
        order (int): Number of prior templates used as context.
        score_threshold (float): Effective anomaly threshold after fitting.
        threshold_source (str): Whether the threshold was configured or calibrated.
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
    calibration_quantile: float
    smoothing: float
    normal_sequence_count: int
    normal_transition_count: int
    normal_context_vocabulary: int
    normal_template_vocabulary: int


def _sequence_transitions(
    templates: list[str],
    order: int,
) -> Iterator[tuple[int, tuple[str, ...], str]]:
    """Yield fixed-width transition contexts and next templates.

    Args:
        templates (list[str]): Template ids from one sequence.
        order (int): Number of prior templates to use as context.

    Yields:
        (tuple[str, ...], str): Context tuple and next template.
    """
    if len(templates) <= order:
        return
    for index in range(order, len(templates)):
        yield index, tuple(templates[index - order : index]), templates[index]


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
