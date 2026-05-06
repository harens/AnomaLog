"""Shared next-event prediction diagnostics for experiment detectors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    from collections.abc import Sequence


class VocabularyPolicy(str, Enum):
    """Vocabulary policy labels for next-event diagnostics.

    Attributes:
        TRAIN_ONLY: Use the fitted training vocabulary only.
        FULL_DATASET: Use the complete dataset vocabulary observed in the run.
    """

    TRAIN_ONLY = "train_only"
    FULL_DATASET = "full_dataset"


class NextEventPredictionExclusionReason(str, Enum):
    """Stable exclusion reasons for next-event diagnostics.

    Attributes:
        UNKNOWN_HISTORY: The context contained a template outside the fitted
            vocabulary.
        UNKNOWN_TARGET: The target template was outside the fitted vocabulary.
        INSUFFICIENT_HISTORY: The event did not have enough prior context.
    """

    UNKNOWN_HISTORY = "unknown_history"
    UNKNOWN_TARGET = "unknown_target"
    INSUFFICIENT_HISTORY = "insufficient_history"


class NextEventPredictionTotals(msgspec.Struct, frozen=True):
    """Aggregate counts for next-event prediction diagnostics.

    Attributes:
        events_seen (int): Total next-event samples encountered.
        events_eligible (int): Samples that were eligible for scoring.
        coverage (float): Eligible samples divided by samples seen.
    """

    events_seen: int
    events_eligible: int
    coverage: float


class NextEventPredictionTopK(msgspec.Struct, frozen=True):
    """Top-k next-event hit counts and accuracies.

    Attributes:
        k_values (list[int]): Reporting cut-offs used for top-k metrics.
        hit_count (dict[str, int]): Number of eligible samples whose target
            label appeared within each top-k candidate set.
        accuracy (dict[str, float]): Top-k hit rate for each reporting cut-off.
    """

    k_values: list[int]
    hit_count: dict[str, int]
    accuracy: dict[str, float]


class NextEventPredictionWeightedMetrics(msgspec.Struct, frozen=True):
    """Weighted top-1 classification metrics for next-event prediction.

    Attributes:
        precision (float): Support-weighted top-1 precision.
        recall (float): Support-weighted top-1 recall.
        f1 (float): Support-weighted top-1 F1 score.
        accuracy (float): Top-1 accuracy over eligible samples.
    """

    precision: float
    recall: float
    f1: float
    accuracy: float


class NextEventPredictionMacroMetrics(msgspec.Struct, frozen=True):
    """Macro-averaged top-1 classification metrics for next-event prediction.

    Attributes:
        precision (float): Unweighted top-1 precision across labels.
        recall (float): Unweighted top-1 recall across labels.
        f1 (float): Unweighted top-1 F1 score across labels.
        accuracy (float): Overall top-1 accuracy over eligible samples.
    """

    precision: float
    recall: float
    f1: float
    accuracy: float


class NextEventPredictionExclusions(msgspec.Struct, frozen=True):
    """Counts of excluded next-event samples by reason.

    Attributes:
        unknown_history (int): Samples excluded because their history
            contained an unseen template.
        unknown_target (int): Samples excluded because the target template was
            unseen during training.
        insufficient_history (int): Samples excluded because they did not
            have enough prior events for a full history window.
    """

    unknown_history: int
    unknown_target: int
    insufficient_history: int


class NextEventPredictionDiagnostics(msgspec.Struct, frozen=True):
    """Serialisable next-event prediction diagnostics for model manifests.

    Attributes:
        task (str): Stable task label for downstream consumers.
        totals (NextEventPredictionTotals): Aggregate next-event coverage
            counts.
        top_k (NextEventPredictionTopK): Top-k hit counts and accuracies.
        classification_top1_macro (NextEventPredictionMacroMetrics): Macro-
            averaged top-1 classification metrics.
        classification_top1_weighted (NextEventPredictionWeightedMetrics):
            Weighted top-1 classification metrics.
        table_iv_prediction_metrics (NextEventPredictionWeightedMetrics):
            Explicit Table IV comparison block. This mirrors the weighted
            top-1 metrics because the paper reports weighted multi-class
            next-event scores over a 20/80 chronological split.
        exclusions (NextEventPredictionExclusions): Exclusion counts by
            reason.
        vocabulary_policy (VocabularyPolicy): Stable vocabulary policy label
            for the diagnostic run.
    """

    task: str
    totals: NextEventPredictionTotals
    top_k: NextEventPredictionTopK
    classification_top1_macro: NextEventPredictionMacroMetrics
    classification_top1_weighted: NextEventPredictionWeightedMetrics
    table_iv_prediction_metrics: NextEventPredictionWeightedMetrics
    exclusions: NextEventPredictionExclusions
    vocabulary_policy: VocabularyPolicy


@dataclass(slots=True)
class NextEventPredictionState:
    """Accumulate next-event prediction metrics while scoring a run.

    Attributes:
        k_values (tuple[int, ...]): Top-k reporting cut-offs to track.
        vocabulary_policy (VocabularyPolicy): Vocabulary policy used to
            decide whether unseen history/target templates are excluded.
        events_seen (int): Total samples encountered, including exclusions.
        events_eligible (int): Samples eligible for scoring.
        top_k_hit_counts (dict[int, int]): Raw top-k hit counters.
        actual_counts (Counter[str]): Label support counts for eligible samples.
        predicted_counts (Counter[str]): Top-1 prediction counts.
        correct_counts (Counter[str]): Top-1 true-positive counts by label.
        exclusions (Counter[NextEventPredictionExclusionReason]): Excluded-
            sample counts by reason.
    """

    k_values: tuple[int, ...]
    vocabulary_policy: VocabularyPolicy
    events_seen: int = 0
    events_eligible: int = 0
    top_k_hit_counts: dict[int, int] = field(default_factory=dict)
    actual_counts: Counter[str] = field(default_factory=Counter)
    predicted_counts: Counter[str] = field(default_factory=Counter)
    correct_counts: Counter[str] = field(default_factory=Counter)
    exclusions: Counter[NextEventPredictionExclusionReason] = field(
        default_factory=Counter,
    )

    @classmethod
    def create(
        cls,
        *,
        k_values: Sequence[int],
        vocabulary_policy: VocabularyPolicy,
    ) -> NextEventPredictionState:
        """Build a normalised accumulator for one scoring run.

        Args:
            k_values (Sequence[int]): Top-k reporting cut-offs to track.
            vocabulary_policy (VocabularyPolicy): Vocabulary policy used to
                decide whether unseen history/target templates are excluded.

        Returns:
            NextEventPredictionState: Reset accumulator for one scoring run.
        """
        return cls(
            k_values=tuple(k_values),
            vocabulary_policy=vocabulary_policy,
        )

    def record_exclusion(self, reason: NextEventPredictionExclusionReason) -> None:
        """Record one excluded event sample.

        Args:
            reason (NextEventPredictionExclusionReason): Stable exclusion
                reason label.
        """
        self.events_seen += 1
        self.exclusions[reason] += 1

    def record_prediction(
        self,
        *,
        actual_label: str,
        predicted_labels: Sequence[str],
    ) -> None:
        """Record one eligible next-event prediction.

        Args:
            actual_label (str): True next-event label.
            predicted_labels (Sequence[str]): Ranked predicted labels, with
                the best candidate first.
        """
        self.events_seen += 1
        self._record_prediction(
            actual_label=actual_label,
            predicted_labels=predicted_labels,
        )

    def record_observation(
        self,
        *,
        actual_label: str,
        predicted_labels: Sequence[str],
        target_is_known: bool = True,
        history_is_known: bool = True,
    ) -> None:
        """Record or exclude one next-event sample under the active policy.

        Args:
            actual_label (str): True next-event label.
            predicted_labels (Sequence[str]): Ranked predicted labels, with
                the best candidate first.
            target_is_known (bool): Whether the observed target is present in
                the configured vocabulary.
            history_is_known (bool): Whether the observed history is present
                in the configured vocabulary.
        """
        if self.vocabulary_policy is VocabularyPolicy.TRAIN_ONLY:
            if not target_is_known:
                self.exclusions[NextEventPredictionExclusionReason.UNKNOWN_TARGET] += 1
                self.events_seen += 1
                return
            if not history_is_known:
                self.exclusions[NextEventPredictionExclusionReason.UNKNOWN_HISTORY] += 1
                self.events_seen += 1
                return
        self.events_seen += 1
        self._record_prediction(
            actual_label=actual_label,
            predicted_labels=predicted_labels,
        )

    def _record_prediction(
        self,
        *,
        actual_label: str,
        predicted_labels: Sequence[str],
    ) -> None:
        self.events_eligible += 1
        self.actual_counts[actual_label] += 1
        if not predicted_labels:
            return

        top1_label = predicted_labels[0]
        self.predicted_counts[top1_label] += 1
        if top1_label == actual_label:
            self.correct_counts[actual_label] += 1
        for k_value in self.k_values:
            if actual_label in predicted_labels[:k_value]:
                self.top_k_hit_counts[k_value] = (
                    self.top_k_hit_counts.get(k_value, 0) + 1
                )

    def snapshot(self) -> NextEventPredictionDiagnostics | None:
        """Return serialisable diagnostics for the accumulated run.

        Returns:
            NextEventPredictionDiagnostics | None: Aggregated next-event
                diagnostics, or `None` if no samples were seen.
        """
        if self.events_seen == 0:
            return None

        total_eligible = self.events_eligible
        total_seen = self.events_seen
        coverage = total_eligible / total_seen if total_seen else 0.0
        total_support = sum(self.actual_counts.values())
        total_correct = sum(self.correct_counts.values())
        top_k_accuracy = {
            str(k_value): _fraction(
                self.top_k_hit_counts.get(k_value, 0),
                total_eligible,
            )
            for k_value in self.k_values
        }
        macro_precision, macro_recall, macro_f1 = _macro_top1_metrics(
            actual_counts=self.actual_counts,
            predicted_counts=self.predicted_counts,
            correct_counts=self.correct_counts,
        )
        weighted_precision, weighted_recall, weighted_f1 = _weighted_top1_metrics(
            actual_counts=self.actual_counts,
            predicted_counts=self.predicted_counts,
            correct_counts=self.correct_counts,
        )
        return NextEventPredictionDiagnostics(
            task="next_event_prediction",
            totals=NextEventPredictionTotals(
                events_seen=total_seen,
                events_eligible=total_eligible,
                coverage=coverage,
            ),
            top_k=NextEventPredictionTopK(
                k_values=list(self.k_values),
                hit_count={
                    str(k_value): self.top_k_hit_counts.get(k_value, 0)
                    for k_value in self.k_values
                },
                accuracy=top_k_accuracy,
            ),
            classification_top1_macro=NextEventPredictionMacroMetrics(
                precision=macro_precision,
                recall=macro_recall,
                f1=macro_f1,
                accuracy=_fraction(total_correct, total_support),
            ),
            classification_top1_weighted=NextEventPredictionWeightedMetrics(
                precision=weighted_precision,
                recall=weighted_recall,
                f1=weighted_f1,
                accuracy=_fraction(total_correct, total_support),
            ),
            table_iv_prediction_metrics=NextEventPredictionWeightedMetrics(
                precision=weighted_precision,
                recall=weighted_recall,
                f1=weighted_f1,
                accuracy=_fraction(total_correct, total_support),
            ),
            exclusions=NextEventPredictionExclusions(
                unknown_history=self.exclusions.get(
                    NextEventPredictionExclusionReason.UNKNOWN_HISTORY,
                    0,
                ),
                unknown_target=self.exclusions.get(
                    NextEventPredictionExclusionReason.UNKNOWN_TARGET,
                    0,
                ),
                insufficient_history=self.exclusions.get(
                    NextEventPredictionExclusionReason.INSUFFICIENT_HISTORY,
                    0,
                ),
            ),
            vocabulary_policy=self.vocabulary_policy,
        )


def _weighted_top1_metrics(
    *,
    actual_counts: Counter[str],
    predicted_counts: Counter[str],
    correct_counts: Counter[str],
) -> tuple[float, float, float]:
    """Return weighted precision, recall, and F1 for top-1 predictions.

    Args:
        actual_counts (Counter[str]): Support counts per true label.
        predicted_counts (Counter[str]): Top-1 prediction counts per label.
        correct_counts (Counter[str]): True-positive counts per label.

    Returns:
        tuple[float, float, float]: Weighted precision, recall, and F1.
    """
    total_support = sum(actual_counts.values())
    if total_support == 0:
        return 0.0, 0.0, 0.0

    weighted_precision = 0.0
    weighted_recall = 0.0
    weighted_f1 = 0.0
    for label, support in actual_counts.items():
        true_positive = correct_counts.get(label, 0)
        predicted_positive = predicted_counts.get(label, 0)
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        weighted_precision += precision * support
        weighted_recall += recall * support
        weighted_f1 += f1 * support

    return (
        weighted_precision / total_support,
        weighted_recall / total_support,
        weighted_f1 / total_support,
    )


def _macro_top1_metrics(
    *,
    actual_counts: Counter[str],
    predicted_counts: Counter[str],
    correct_counts: Counter[str],
) -> tuple[float, float, float]:
    """Return macro precision, recall, and F1 for top-1 predictions.

    Args:
        actual_counts (Counter[str]): Support counts per true label.
        predicted_counts (Counter[str]): Top-1 prediction counts per label.
        correct_counts (Counter[str]): True-positive counts per label.

    Returns:
        tuple[float, float, float]: Macro precision, recall, and F1.
    """
    if not actual_counts:
        return 0.0, 0.0, 0.0

    label_count = len(actual_counts)
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    for label, support in actual_counts.items():
        true_positive = correct_counts.get(label, 0)
        predicted_positive = predicted_counts.get(label, 0)
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

    return (
        precision_sum / label_count,
        recall_sum / label_count,
        f1_sum / label_count,
    )


def _fraction(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
