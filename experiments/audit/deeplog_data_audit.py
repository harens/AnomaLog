"""Dataset-level auditing helpers for DeepLog reproducibility checks."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import ceil, floor, isclose
from statistics import mean
from typing import TYPE_CHECKING

from anomalog.parsers.structured.contracts import ANOMALOUS_FIELD, is_anomalous_label
from anomalog.sequences import SplitApplicationOrder, SplitLabel, StraddlingGroupPolicy
from experiments.config import (
    ChronologicalStreamSequenceConfig,
    DatasetVariantConfig,
    EntitySequenceConfig,
    RawEntryPrefixCountSplitConfig,
    RawEntryPrefixFractionSplitConfig,
    RawEntryPrefixNormalFractionSplitConfig,
    serialise_config,
)
from experiments.config_loader import _decode_dataset_config
from experiments.datasets import build_dataset_spec
from experiments.models.deeplog.shared import (
    evaluation_event_mask_for_sequence,
    training_event_mask_for_sequence,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from anomalog.sequences import TemplateSequence
    from experiments.models.base import ExperimentModelConfig


_BGL_PAPER_CHUNK_SIZE = 100_000
_BGL_PAPER_TRAIN_FRACTION_1PCT = 0.01
_BGL_PAPER_TRAIN_FRACTION_10PCT = 0.10
_BGL_PAPER_HISTORY_SIZE = 3
_BGL_PAPER_TOP_G = 6
_BGL_PAPER_NUM_LAYERS = 1
_BGL_PAPER_HIDDEN_SIZE = 256
_HDFS_PAPER_TRAIN_ENTRY_COUNT = 100_000
_HDFS_PAPER_TRAIN_FRACTION = 0.01
_HDFS_PAPER_TEST_FRACTION = 0.99
_HDFS_PAPER_HISTORY_SIZE = 10
_HDFS_PAPER_TOP_G = 9
_HDFS_PAPER_NUM_LAYERS = 2
_HDFS_PAPER_HIDDEN_SIZE = 64
_FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class WarmupAccounting:
    """DeepLog history warm-up accounting totals.

    Attributes:
        events_seen (int): Total observed events considered in accounting.
        insufficient_history (int): Events excluded for lacking full history.
        events_eligible (int): Events with enough history for prediction.
        insufficient_history_rate (float): Fraction of `events_seen` excluded
            for insufficient history.
    """

    events_seen: int
    insufficient_history: int
    events_eligible: int
    insufficient_history_rate: float


@dataclass(frozen=True, slots=True)
class SequenceLengthSummary:
    """Summary statistics over sequence lengths.

    Attributes:
        min (int): Minimum sequence length.
        p25 (float): 25th percentile sequence length.
        median (float): 50th percentile sequence length.
        p75 (float): 75th percentile sequence length.
        max (int): Maximum sequence length.
        mean (float): Mean sequence length.
        count_lte_history_size (int): Sequence count with length `<= h`.
        count_gt_history_size (int): Sequence count with length `> h`.
    """

    min: int
    p25: float
    median: float
    p75: float
    max: int
    mean: float
    count_lte_history_size: int
    count_gt_history_size: int


@dataclass(frozen=True, slots=True)
class SplitAuditSummary:
    """Split-scoped sequence and event counts.

    Attributes:
        sequence_count (int): Number of sequences in the split.
        event_count (int): Number of events in the split.
        normal_sequence_count (int): Non-anomalous sequence count in split.
        anomalous_sequence_count (int): Anomalous sequence count in split.
        warmup (WarmupAccounting): DeepLog warm-up accounting for the split.
    """

    sequence_count: int
    event_count: int
    normal_sequence_count: int
    anomalous_sequence_count: int
    warmup: WarmupAccounting


@dataclass(frozen=True, slots=True)
class NoEligibleSummary:
    """Counts for sequences with no DeepLog-eligible prediction events.

    Attributes:
        sequence_count (int): Number of sequences with zero eligible events.
        label_counts (dict[int, int]): Label counts among those sequences.
    """

    sequence_count: int
    label_counts: dict[int, int]


@dataclass(frozen=True, slots=True)
class TrainingTargetSummary:
    """Training-target eligibility counts for the DeepLog train split.

    Attributes:
        eligible_normal_event_count (int): Normal targets that can train the
            DeepLog models.
        excluded_anomalous_event_count (int): Anomalous targets excluded from
            training.
        excluded_context_event_count (int): Non-anomalous train-split events
            excluded because they occur after the raw-entry cutoff.
        will_train (bool): Whether the train split yields any eligible targets.
    """

    eligible_normal_event_count: int
    excluded_anomalous_event_count: int
    excluded_context_event_count: int
    will_train: bool


@dataclass(frozen=True, slots=True)
class BGLChunkSensitivitySummary:
    """Chunk-size sensitivity summary for BGL chronological-stream audits.

    Attributes:
        chunk_size (int): Chronological chunk size under test.
        sequence_count (int): Number of emitted chunks.
        eligible_training_targets (int): Training-target count eligible under the
            DeepLog mask.
        evaluation_event_count (int): Evaluation events eligible for scoring.
        anomalous_evaluation_targets (int): Evaluation targets labelled anomalous.
        normal_evaluation_targets (int): Evaluation targets labelled normal.
        insufficient_history (int): Events excluded for lacking a full history.
        warmup_loss (int): Extra insufficient-history loss relative to a single
            unchunked stream.
        post_cutoff_events_excluded (int): Post-cutoff events lost to chunk
            handling rather than the evaluation mask.
    """

    chunk_size: int
    sequence_count: int
    eligible_training_targets: int
    evaluation_event_count: int
    anomalous_evaluation_targets: int
    normal_evaluation_targets: int
    insufficient_history: int
    warmup_loss: int
    post_cutoff_events_excluded: int


@dataclass(frozen=True, slots=True)
class EvaluationWarmupSummary:
    """Event-level evaluation accounting for one chronological stream.

    Attributes:
        events_eligible (int): Evaluation targets with enough history.
        insufficient_history (int): Evaluation targets excluded for lacking a
            full history window.
        anomalous_events (int): Eligible evaluation targets labelled anomalous.
        normal_events (int): Eligible evaluation targets labelled normal.
        post_cutoff_events_excluded (int): Evaluation-target events lost to
            chunk/context handling rather than the evaluation mask.
    """

    events_eligible: int
    insufficient_history: int
    anomalous_events: int
    normal_events: int
    post_cutoff_events_excluded: int


@dataclass(frozen=True, slots=True)
class HDFSSessionObservation:
    """One HDFS session observed in raw-entry order.

    Attributes:
        entity_id (str): Session/block identifier.
        first_line_order (int): First raw-entry index for the session.
        last_line_order (int): Last raw-entry index for the session.
        label (int | None): Session label under the dataset's anomaly semantics.
        event_count (int): Number of raw entries in the session.
        pre_cutoff_event_count (int): Number of raw entries before the cutoff.
        post_cutoff_event_count (int): Number of raw entries at or after the cutoff.
    """

    entity_id: str
    first_line_order: int
    last_line_order: int
    label: int | None
    event_count: int
    pre_cutoff_event_count: int
    post_cutoff_event_count: int


@dataclass(slots=True)
class _HDFSSessionAccumulator:
    first_line_order: int
    last_line_order: int
    event_count: int = 0
    pre_cutoff_event_count: int = 0
    post_cutoff_event_count: int = 0


@dataclass(frozen=True, slots=True)
class HDFSFirst100kPolicySummary:
    """Counts for one candidate interpretation of the HDFS first-100k split.

    Attributes:
        policy_name (str): Stable policy label.
        train_normal_sessions (int): Normal sessions assigned to train.
        train_anomalous_sessions (int): Anomalous sessions assigned to train.
        ignored_sessions (int): Sessions withheld from both train and test.
        test_normal_sessions (int): Normal sessions assigned to test.
        test_anomalous_sessions (int): Anomalous sessions assigned to test.
        total_sessions (int): Total observed source sessions.
        emitted_segment_count (int): Number of policy-emitted segments.
        template_count (int): Number of templates/log keys.
        no_eligible_sessions (int): Sessions with no DeepLog-eligible targets.
        train_normal_delta (int): Difference from the paper target train-normal count.
        test_normal_delta (int): Difference from the paper target test-normal count.
        test_anomalous_delta (int): Difference from the paper target
            test-anomalous count.
    """

    policy_name: str
    train_normal_sessions: int
    train_anomalous_sessions: int
    ignored_sessions: int
    test_normal_sessions: int
    test_anomalous_sessions: int
    total_sessions: int
    emitted_segment_count: int
    template_count: int
    no_eligible_sessions: int
    train_normal_delta: int
    test_normal_delta: int
    test_anomalous_delta: int

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-friendly representation.

        Returns:
            dict[str, int | str]: Serialised policy summary.
        """
        return {
            "policy_name": self.policy_name,
            "train_normal_sessions": self.train_normal_sessions,
            "train_anomalous_sessions": self.train_anomalous_sessions,
            "ignored_sessions": self.ignored_sessions,
            "test_normal_sessions": self.test_normal_sessions,
            "test_anomalous_sessions": self.test_anomalous_sessions,
            "total_sessions": self.total_sessions,
            "emitted_segment_count": self.emitted_segment_count,
            "template_count": self.template_count,
            "no_eligible_sessions": self.no_eligible_sessions,
            "train_normal_delta": self.train_normal_delta,
            "test_normal_delta": self.test_normal_delta,
            "test_anomalous_delta": self.test_anomalous_delta,
        }


@dataclass(frozen=True, slots=True)
class DeepLogDatasetAudit:
    """DeepLog-focused audit report for one dataset configuration.

    Attributes:
        dataset_variant (str): Dataset variant config name.
        dataset_name (str): Stable dataset namespace.
        raw_log_entry_count (int): Raw log line count.
        parsed_event_count (int): Parsed structured event count.
        parsed_template_count (int): Number of parsed templates/log keys.
        event_label_distribution (dict[str, int]): Event-level label histogram.
        sequence_label_distribution (dict[str, int]): Sequence-level label
            histogram.
        grouping_key (str): Sequence grouping mode label.
        split_strategy (dict[str, object]): Requested split policy and grouping
            options.
        raw_entry_split_summary (dict[str, int | str | None] | None): Optional
            diagnostics for raw-entry split modes.
        sequence_count (int): Total sequence count.
        train_sequence_count (int): Train split sequence count.
        train_event_count (int): Train split event count.
        train_normal_sequence_count (int): Train split normal sequence count.
        train_anomalous_sequence_count (int): Train split anomalous sequence count.
        test_sequence_count (int): Test split sequence count.
        test_event_count (int): Test split event count.
        test_normal_sequence_count (int): Test split normal sequence count.
        test_anomalous_sequence_count (int): Test split anomalous sequence count.
        ignored_sequence_count (int): Ignored split sequence count.
        ignored_event_count (int): Ignored split event count.
        ignored_normal_sequence_count (int): Ignored split normal sequence count.
        ignored_anomalous_sequence_count (int): Ignored split anomalous sequence count.
        sequence_length_summary (SequenceLengthSummary): Length summary over all
            sequences.
        warmup_overall (WarmupAccounting): Overall warm-up accounting.
        warmup_by_split (dict[str, WarmupAccounting]): Warm-up accounting per split.
        no_eligible_predictions (NoEligibleSummary): Overall no-eligible summary.
        no_eligible_predictions_by_split (dict[str, NoEligibleSummary]):
            No-eligible summary per split.
        training_target_summary (TrainingTargetSummary): Train-split event-level
            eligibility summary for the DeepLog models.
        split_summaries (dict[str, SplitAuditSummary]): Split-level sequence/event
            summaries.
    """

    dataset_variant: str
    dataset_name: str
    raw_log_entry_count: int
    parsed_event_count: int
    parsed_template_count: int
    event_label_distribution: dict[str, int]
    sequence_label_distribution: dict[str, int]
    grouping_key: str
    split_strategy: dict[str, object]
    raw_entry_split_summary: dict[str, int | str | None] | None
    sequence_count: int
    train_sequence_count: int
    train_event_count: int
    train_normal_sequence_count: int
    train_anomalous_sequence_count: int
    test_sequence_count: int
    test_event_count: int
    test_normal_sequence_count: int
    test_anomalous_sequence_count: int
    ignored_sequence_count: int
    ignored_event_count: int
    ignored_normal_sequence_count: int
    ignored_anomalous_sequence_count: int
    sequence_length_summary: SequenceLengthSummary
    warmup_overall: WarmupAccounting
    warmup_by_split: dict[str, WarmupAccounting]
    no_eligible_predictions: NoEligibleSummary
    no_eligible_predictions_by_split: dict[str, NoEligibleSummary]
    training_target_summary: TrainingTargetSummary
    split_summaries: dict[str, SplitAuditSummary]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation.

        Returns:
            dict[str, object]: Audit report converted to built-in JSON types.
        """
        return {
            "dataset_variant": self.dataset_variant,
            "dataset_name": self.dataset_name,
            "raw_log_entry_count": self.raw_log_entry_count,
            "parsed_event_count": self.parsed_event_count,
            "parsed_template_count": self.parsed_template_count,
            "event_label_distribution": dict(self.event_label_distribution),
            "sequence_label_distribution": dict(self.sequence_label_distribution),
            "grouping_key": self.grouping_key,
            "split_strategy": dict(self.split_strategy),
            "raw_entry_split_summary": self.raw_entry_split_summary,
            "sequence_count": self.sequence_count,
            "train_sequence_count": self.train_sequence_count,
            "train_event_count": self.train_event_count,
            "train_normal_sequence_count": self.train_normal_sequence_count,
            "train_anomalous_sequence_count": self.train_anomalous_sequence_count,
            "test_sequence_count": self.test_sequence_count,
            "test_event_count": self.test_event_count,
            "test_normal_sequence_count": self.test_normal_sequence_count,
            "test_anomalous_sequence_count": self.test_anomalous_sequence_count,
            "ignored_sequence_count": self.ignored_sequence_count,
            "ignored_event_count": self.ignored_event_count,
            "ignored_normal_sequence_count": self.ignored_normal_sequence_count,
            "ignored_anomalous_sequence_count": self.ignored_anomalous_sequence_count,
            "sequence_length_summary": _sequence_length_summary_to_dict(
                self.sequence_length_summary,
            ),
            "warmup_overall": _warmup_accounting_to_dict(self.warmup_overall),
            "warmup_by_split": {
                split: _warmup_accounting_to_dict(stats)
                for split, stats in self.warmup_by_split.items()
            },
            "no_eligible_predictions": _no_eligible_summary_to_dict(
                self.no_eligible_predictions,
            ),
            "no_eligible_predictions_by_split": {
                split: _no_eligible_summary_to_dict(stats)
                for split, stats in self.no_eligible_predictions_by_split.items()
            },
            "training_target_summary": _training_target_summary_to_dict(
                self.training_target_summary,
            ),
            "split_summaries": {
                split: _split_audit_summary_to_dict(stats)
                for split, stats in self.split_summaries.items()
            },
        }


@dataclass(slots=True)
class _SplitAccumulator:
    sequence_count: int = 0
    event_count: int = 0
    normal_sequence_count: int = 0
    anomalous_sequence_count: int = 0
    sequence_lengths: list[int] = field(default_factory=list)
    no_eligible_sequence_count: int = 0
    no_eligible_label_counts: Counter[int] = field(default_factory=Counter)
    eligible_training_target_count: int = 0
    excluded_anomalous_training_target_count: int = 0
    excluded_context_training_target_count: int = 0

    def add(self, *, sequence: TemplateSequence, history_size: int) -> None:
        length = len(sequence.events)
        self.sequence_count += 1
        self.event_count += length
        self.sequence_lengths.append(length)
        if sequence.split_label is SplitLabel.TRAIN:
            event_mask = training_event_mask_for_sequence(sequence)
            for event_index, is_eligible in enumerate(event_mask):
                if is_eligible:
                    self.eligible_training_target_count += 1
                    continue
                raw_label = (
                    sequence.event_labels[event_index]
                    if sequence.event_labels is not None
                    else sequence.label
                )
                if is_anomalous_label(raw_label):
                    self.excluded_anomalous_training_target_count += 1
                else:
                    self.excluded_context_training_target_count += 1
        if is_anomalous_label(sequence.label):
            self.anomalous_sequence_count += 1
        else:
            self.normal_sequence_count += 1
        if max(0, length - history_size) == 0:
            self.no_eligible_sequence_count += 1
            self.no_eligible_label_counts[sequence.label] += 1

    def warmup(self, *, history_size: int) -> WarmupAccounting:
        return aggregate_warmup_accounting(
            sequence_lengths=self.sequence_lengths,
            history_size=history_size,
        )

    def no_eligible(self) -> NoEligibleSummary:
        return NoEligibleSummary(
            sequence_count=self.no_eligible_sequence_count,
            label_counts=dict(self.no_eligible_label_counts),
        )

    def training_targets(self) -> TrainingTargetSummary:
        return TrainingTargetSummary(
            eligible_normal_event_count=self.eligible_training_target_count,
            excluded_anomalous_event_count=(
                self.excluded_anomalous_training_target_count
            ),
            excluded_context_event_count=self.excluded_context_training_target_count,
            will_train=self.eligible_training_target_count > 0,
        )

    def as_summary(self, *, history_size: int) -> SplitAuditSummary:
        return SplitAuditSummary(
            sequence_count=self.sequence_count,
            event_count=self.event_count,
            normal_sequence_count=self.normal_sequence_count,
            anomalous_sequence_count=self.anomalous_sequence_count,
            warmup=self.warmup(history_size=history_size),
        )


def warmup_counts_for_sequence_length(
    *,
    sequence_length: int,
    history_size: int,
) -> tuple[int, int]:
    """Return `(insufficient_history, eligible_events)` for one sequence.

    Args:
        sequence_length (int): Number of events in the sequence.
        history_size (int): DeepLog history window size.

    Returns:
        tuple[int, int]: Warm-up and eligible event counts for the sequence.

    Raises:
        ValueError: If `sequence_length` or `history_size` is negative.
    """
    if sequence_length < 0:
        msg = "sequence_length must be non-negative."
        raise ValueError(msg)
    if history_size < 0:
        msg = "history_size must be non-negative."
        raise ValueError(msg)
    insufficient_history = min(sequence_length, history_size)
    events_eligible = max(0, sequence_length - history_size)
    return insufficient_history, events_eligible


def aggregate_warmup_accounting(
    *,
    sequence_lengths: Iterable[int],
    history_size: int,
    additional_excluded_events: int = 0,
) -> WarmupAccounting:
    """Aggregate DeepLog warm-up accounting across sequence lengths.

    Args:
        sequence_lengths (Iterable[int]): Sequence lengths to aggregate.
        history_size (int): DeepLog history window size.
        additional_excluded_events (int): Extra excluded event count to include
            in `events_seen` totals.

    Returns:
        WarmupAccounting: Aggregate warm-up accounting summary.

    Raises:
        ValueError: If `history_size` or `additional_excluded_events` is
            negative.
    """
    if history_size < 0:
        msg = "history_size must be non-negative."
        raise ValueError(msg)
    if additional_excluded_events < 0:
        msg = "additional_excluded_events must be non-negative."
        raise ValueError(msg)

    insufficient_history = 0
    events_eligible = 0
    for sequence_length in sequence_lengths:
        insufficient, eligible = warmup_counts_for_sequence_length(
            sequence_length=sequence_length,
            history_size=history_size,
        )
        insufficient_history += insufficient
        events_eligible += eligible
    events_seen = insufficient_history + events_eligible + additional_excluded_events
    insufficient_history_rate = (
        insufficient_history / events_seen if events_seen else 0.0
    )
    return WarmupAccounting(
        events_seen=events_seen,
        insufficient_history=insufficient_history,
        events_eligible=events_eligible,
        insufficient_history_rate=insufficient_history_rate,
    )


def audit_dataset_for_deeplog(
    *,
    config: DatasetVariantConfig,
    repo_root: Path,
    history_size: int,
    validate_paper_config: bool = True,
) -> DeepLogDatasetAudit:
    """Build and audit one dataset variant for DeepLog reproducibility work.

    Args:
        config (DatasetVariantConfig): Dataset config to audit.
        repo_root (Path): Repository root used to resolve config paths.
        history_size (int): DeepLog history window used for warm-up accounting.
        validate_paper_config (bool): Whether to enforce the DeepLog paper
            protocol before building the dataset.

    Returns:
        DeepLogDatasetAudit: Audit report for the requested dataset config.

    Raises:
        ValueError: If `history_size` is negative.
    """
    if history_size < 0:
        msg = "history_size must be non-negative."
        raise ValueError(msg)

    if validate_paper_config:
        validate_deeplog_paper_config(dataset_config=config)
    dataset_spec = build_dataset_spec(config, repo_root=repo_root)
    templated = dataset_spec.build()
    raw_log_entry_count = _count_lines(templated.sink.raw_dataset_path.resolve())
    event_label_distribution = _event_label_distribution(templated.sink)

    sequences = config.sequence.apply(templated)
    (
        all_split_accumulator,
        split_accumulators,
        template_set,
        sequence_label_counter,
    ) = _collect_sequence_stats(
        sequences=sequences,
        history_size=history_size,
    )
    grouping_value, split_strategy = _build_split_strategy(config=config)
    raw_entry_split_summary = sequences.build_raw_entry_split_summary()

    return DeepLogDatasetAudit(
        dataset_variant=config.name,
        dataset_name=config.dataset_name,
        raw_log_entry_count=raw_log_entry_count,
        parsed_event_count=templated.sink.count_rows(),
        parsed_template_count=len(template_set),
        event_label_distribution=event_label_distribution,
        sequence_label_distribution={
            str(key): value for key, value in sequence_label_counter.items()
        },
        grouping_key=grouping_value,
        split_strategy=split_strategy,
        raw_entry_split_summary=(
            None
            if raw_entry_split_summary is None
            else raw_entry_split_summary.as_dict()
        ),
        sequence_count=all_split_accumulator.sequence_count,
        train_sequence_count=split_accumulators[SplitLabel.TRAIN].sequence_count,
        train_event_count=split_accumulators[SplitLabel.TRAIN].event_count,
        train_normal_sequence_count=split_accumulators[
            SplitLabel.TRAIN
        ].normal_sequence_count,
        train_anomalous_sequence_count=split_accumulators[
            SplitLabel.TRAIN
        ].anomalous_sequence_count,
        test_sequence_count=split_accumulators[SplitLabel.TEST].sequence_count,
        test_event_count=split_accumulators[SplitLabel.TEST].event_count,
        test_normal_sequence_count=split_accumulators[
            SplitLabel.TEST
        ].normal_sequence_count,
        test_anomalous_sequence_count=split_accumulators[
            SplitLabel.TEST
        ].anomalous_sequence_count,
        ignored_sequence_count=split_accumulators[SplitLabel.IGNORED].sequence_count,
        ignored_event_count=split_accumulators[SplitLabel.IGNORED].event_count,
        ignored_normal_sequence_count=split_accumulators[
            SplitLabel.IGNORED
        ].normal_sequence_count,
        ignored_anomalous_sequence_count=split_accumulators[
            SplitLabel.IGNORED
        ].anomalous_sequence_count,
        sequence_length_summary=_sequence_length_summary(
            sequence_lengths=all_split_accumulator.sequence_lengths,
            history_size=history_size,
        ),
        warmup_overall=all_split_accumulator.warmup(history_size=history_size),
        warmup_by_split={
            split_label.value: split_accumulator.warmup(history_size=history_size)
            for split_label, split_accumulator in split_accumulators.items()
        },
        no_eligible_predictions=all_split_accumulator.no_eligible(),
        no_eligible_predictions_by_split={
            split_label.value: split_accumulator.no_eligible()
            for split_label, split_accumulator in split_accumulators.items()
        },
        training_target_summary=split_accumulators[SplitLabel.TRAIN].training_targets(),
        split_summaries={
            split_label.value: split_accumulator.as_summary(history_size=history_size)
            for split_label, split_accumulator in split_accumulators.items()
        },
    )


def _require_equal(value: object, expected: object, message: str) -> None:
    if value != expected:
        raise ValueError(message)


def _require_close(value: float, expected: float, message: str) -> None:
    if not isclose(value, expected, rel_tol=0.0, abs_tol=_FLOAT_TOLERANCE):
        raise ValueError(message)


def _model_config_value(model_config: object, attribute: str) -> object:
    return getattr(model_config, attribute)


def _structured_line_order(row: object) -> int:
    return int(getattr(row, "line_order", 0))


def validate_deeplog_paper_config(
    *,
    dataset_config: DatasetVariantConfig,
    model_config: ExperimentModelConfig | None = None,
) -> None:
    """Fail fast when a named DeepLog paper config drifts from its protocol.

    Args:
        dataset_config (DatasetVariantConfig): Dataset config to validate.
        model_config (ExperimentModelConfig | None): Optional decoded model
            config to validate.
    """
    if dataset_config.name.startswith("bgl_deeplog_paper_"):
        _validate_bgl_deeplog_paper_config(
            dataset_config=dataset_config,
            model_config=model_config,
        )
        return

    if dataset_config.name.startswith("hdfs_v1_deeplog_paper_"):
        _validate_hdfs_deeplog_paper_config(
            dataset_config=dataset_config,
            model_config=model_config,
        )


def _validate_bgl_deeplog_paper_config(
    *,
    dataset_config: DatasetVariantConfig,
    model_config: ExperimentModelConfig | None,
) -> None:
    sequence = dataset_config.sequence
    if not isinstance(sequence, ChronologicalStreamSequenceConfig):
        msg = "BGL DeepLog paper configs must use chronological_stream grouping."
        raise TypeError(msg)
    _require_equal(
        sequence.chunk_size,
        _BGL_PAPER_CHUNK_SIZE,
        "BGL DeepLog paper configs must use chunk_size = 100000.",
    )
    split = sequence.split
    if split is None:
        msg = "BGL DeepLog paper configs must define a raw-entry split."
        raise ValueError(msg)
    _require_equal(
        split.application_order,
        SplitApplicationOrder.BEFORE_GROUPING,
        "BGL DeepLog paper configs must split before grouping.",
    )
    if isinstance(split, RawEntryPrefixFractionSplitConfig):
        _require_close(
            split.train_entry_fraction,
            _BGL_PAPER_TRAIN_FRACTION_10PCT,
            "BGL 10% paper configs must use train_entry_fraction = 0.10.",
        )
        _require_close(
            sequence.train_fraction,
            _BGL_PAPER_TRAIN_FRACTION_10PCT,
            "BGL 10% paper configs must use train_fraction = 0.10.",
        )
        _require_close(
            sequence.test_fraction,
            1.0 - _BGL_PAPER_TRAIN_FRACTION_10PCT,
            "BGL 10% paper configs must use test_fraction = 0.90.",
        )
    elif isinstance(split, RawEntryPrefixNormalFractionSplitConfig):
        _require_close(
            split.train_normal_entry_fraction,
            _BGL_PAPER_TRAIN_FRACTION_1PCT,
            "BGL 1% paper configs must use train_normal_entry_fraction = 0.01.",
        )
        _require_close(
            sequence.train_fraction,
            _BGL_PAPER_TRAIN_FRACTION_1PCT,
            "BGL 1% paper configs must use train_fraction = 0.01.",
        )
        _require_close(
            sequence.test_fraction,
            1.0 - _BGL_PAPER_TRAIN_FRACTION_1PCT,
            "BGL 1% paper configs must use test_fraction = 0.99.",
        )
    else:
        msg = "BGL DeepLog paper configs must use raw-entry prefix split modes."
        raise TypeError(msg)
    if model_config is not None:
        _require_equal(
            _model_config_value(model_config, "history_size"),
            _BGL_PAPER_HISTORY_SIZE,
            "BGL DeepLog paper configs must use history_size = 3.",
        )
        _require_equal(
            _model_config_value(model_config, "top_g"),
            _BGL_PAPER_TOP_G,
            "BGL DeepLog paper configs must use top_g = 6.",
        )
        _require_equal(
            _model_config_value(model_config, "num_layers"),
            _BGL_PAPER_NUM_LAYERS,
            "BGL DeepLog paper configs must use num_layers = 1.",
        )
        _require_equal(
            _model_config_value(model_config, "hidden_size"),
            _BGL_PAPER_HIDDEN_SIZE,
            "BGL DeepLog paper configs must use hidden_size = 256.",
        )


def _validate_hdfs_deeplog_paper_config(
    *,
    dataset_config: DatasetVariantConfig,
    model_config: ExperimentModelConfig | None,
) -> None:
    sequence = dataset_config.sequence
    if not isinstance(sequence, EntitySequenceConfig):
        msg = "HDFS DeepLog paper configs must use entity grouping."
        raise TypeError(msg)
    if not sequence.train_on_normal_entities_only:
        msg = "HDFS DeepLog paper configs must train on normal entities only."
        raise ValueError(msg)
    split = sequence.split
    if not isinstance(split, RawEntryPrefixCountSplitConfig):
        msg = "HDFS DeepLog paper configs must use raw_entry_prefix_count."
        raise TypeError(msg)
    _require_equal(
        split.application_order,
        SplitApplicationOrder.BEFORE_GROUPING,
        "HDFS DeepLog paper configs must split before grouping.",
    )
    _require_equal(
        split.train_entry_count,
        _HDFS_PAPER_TRAIN_ENTRY_COUNT,
        "HDFS DeepLog paper configs must use train_entry_count = 100000.",
    )
    _require_close(
        sequence.train_fraction,
        _HDFS_PAPER_TRAIN_FRACTION,
        "HDFS DeepLog paper configs must use train_fraction = 0.01.",
    )
    _require_close(
        sequence.test_fraction,
        _HDFS_PAPER_TEST_FRACTION,
        "HDFS DeepLog paper configs must use test_fraction = 0.99.",
    )
    if dataset_config.name.endswith("split_partial"):
        _require_equal(
            split.straddling_group_policy,
            StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES,
            "HDFS split_partial configs must use split_partial_sequences.",
        )
    if dataset_config.name.endswith("assign_first"):
        _require_equal(
            split.straddling_group_policy,
            StraddlingGroupPolicy.ASSIGN_BY_FIRST_EVENT,
            "HDFS assign_first configs must use assign_by_first_event.",
        )
    if model_config is not None:
        _require_equal(
            _model_config_value(model_config, "history_size"),
            _HDFS_PAPER_HISTORY_SIZE,
            "HDFS DeepLog paper configs must use history_size = 10.",
        )
        _require_equal(
            _model_config_value(model_config, "top_g"),
            _HDFS_PAPER_TOP_G,
            "HDFS DeepLog paper configs must use top_g = 9.",
        )
        _require_equal(
            _model_config_value(model_config, "num_layers"),
            _HDFS_PAPER_NUM_LAYERS,
            "HDFS DeepLog paper configs must use num_layers = 2.",
        )
        _require_equal(
            _model_config_value(model_config, "hidden_size"),
            _HDFS_PAPER_HIDDEN_SIZE,
            "HDFS DeepLog paper configs must use hidden_size = 64.",
        )


def audit_bgl_chunk_size_sensitivity(
    *,
    config: DatasetVariantConfig,
    repo_root: Path,
    history_size: int,
    chunk_sizes: Iterable[int],
) -> list[BGLChunkSensitivitySummary]:
    """Measure how BGL chronological chunking affects warm-up accounting.

    Args:
        config (DatasetVariantConfig): BGL paper dataset config to audit.
        repo_root (Path): Repository root used to resolve the config.
        history_size (int): DeepLog history window size.
        chunk_sizes (Iterable[int]): Chunk sizes to compare.

    Returns:
        list[BGLChunkSensitivitySummary]: One summary per requested chunk size.
    """
    validate_deeplog_paper_config(dataset_config=config)
    summaries: list[BGLChunkSensitivitySummary] = []
    for chunk_size in chunk_sizes:
        chunk_config = _dataset_config_with_chunk_size(
            config,
            chunk_size=chunk_size,
        )
        report = audit_dataset_for_deeplog(
            config=chunk_config,
            repo_root=repo_root,
            history_size=history_size,
            validate_paper_config=False,
        )
        dataset_spec = build_dataset_spec(chunk_config, repo_root=repo_root)
        templated = dataset_spec.build()
        evaluation_summary = _evaluation_warmup_from_sequences(
            sequences=chunk_config.sequence.apply(templated),
            history_size=history_size,
        )
        summaries.append(
            BGLChunkSensitivitySummary(
                chunk_size=chunk_size,
                sequence_count=report.sequence_count,
                eligible_training_targets=(
                    report.training_target_summary.eligible_normal_event_count
                ),
                evaluation_event_count=evaluation_summary.events_eligible,
                anomalous_evaluation_targets=evaluation_summary.anomalous_events,
                normal_evaluation_targets=evaluation_summary.normal_events,
                insufficient_history=evaluation_summary.insufficient_history,
                warmup_loss=evaluation_summary.insufficient_history,
                post_cutoff_events_excluded=(
                    evaluation_summary.post_cutoff_events_excluded
                ),
            ),
        )
    return summaries


def audit_hdfs_first_100k_policies(
    *,
    config: DatasetVariantConfig,
    repo_root: Path,
    history_size: int,
) -> list[HDFSFirst100kPolicySummary]:
    """Compare plausible HDFS first-100k split interpretations.

    Args:
        config (DatasetVariantConfig): HDFS paper dataset config to audit.
        repo_root (Path): Repository root used to resolve the config.
        history_size (int): DeepLog history window size.

    Returns:
        list[HDFSFirst100kPolicySummary]: Candidate policy summaries ordered by
            policy name.
    """
    validate_deeplog_paper_config(dataset_config=config)
    dataset_spec = build_dataset_spec(config, repo_root=repo_root)
    templated = dataset_spec.build()
    parsed_template_count = len(
        {
            template
            for sequence in config.sequence.apply(templated)
            for template in sequence.templates
        },
    )
    sessions = _collect_hdfs_session_observations(
        rows=templated.sink.iter_structured_lines_in_source_order()(),
        label_for_group=templated.anomaly_labels.label_for_group,
        cutoff=_HDFS_PAPER_TRAIN_ENTRY_COUNT,
    )
    return _summarise_hdfs_first_100k_policies(
        sessions=sessions,
        cutoff=_HDFS_PAPER_TRAIN_ENTRY_COUNT,
        history_size=history_size,
        template_count=parsed_template_count,
    )


def _event_label_distribution(sink: object) -> dict[str, int]:
    iter_structured_lines = getattr(sink, "iter_structured_lines", None)
    if not callable(iter_structured_lines):
        return {}
    labels: Counter[str] = Counter()
    for row in iter_structured_lines(columns=[ANOMALOUS_FIELD])():
        raw_label = row.anomalous
        if raw_label is None:
            labels["none"] += 1
        else:
            labels[str(int(raw_label))] += 1
    return dict(labels)


def _count_lines(path: Path) -> int:
    line_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as file_obj:
        for _ in file_obj:
            line_count += 1
    return line_count


def _collect_sequence_stats(
    *,
    sequences: Iterable[TemplateSequence],
    history_size: int,
) -> tuple[
    _SplitAccumulator,
    dict[SplitLabel, _SplitAccumulator],
    set[str],
    Counter[int],
]:
    all_split_accumulator = _SplitAccumulator()
    split_accumulators = {
        SplitLabel.TRAIN: _SplitAccumulator(),
        SplitLabel.TEST: _SplitAccumulator(),
        SplitLabel.IGNORED: _SplitAccumulator(),
    }
    template_set: set[str] = set()
    sequence_label_counter: Counter[int] = Counter()

    for sequence in sequences:
        all_split_accumulator.add(sequence=sequence, history_size=history_size)
        split_accumulators[sequence.split_label].add(
            sequence=sequence,
            history_size=history_size,
        )
        sequence_label_counter[sequence.label] += 1
        template_set.update(sequence.templates)

    return (
        all_split_accumulator,
        split_accumulators,
        template_set,
        sequence_label_counter,
    )


def _build_split_strategy(
    *,
    config: DatasetVariantConfig,
) -> tuple[str, dict[str, object]]:
    sequence_config = serialise_config(config.sequence)
    grouping_value = sequence_config.get("grouping")
    if not isinstance(grouping_value, str):
        msg = "sequence config must define grouping."
        raise TypeError(msg)

    split_strategy: dict[str, object] = dict(sequence_config)
    split_strategy["grouping"] = grouping_value
    return grouping_value, split_strategy


def _training_target_summary_to_dict(
    summary: TrainingTargetSummary,
) -> dict[str, int | bool]:
    return {
        "eligible_normal_event_count": summary.eligible_normal_event_count,
        "excluded_anomalous_event_count": summary.excluded_anomalous_event_count,
        "excluded_context_event_count": summary.excluded_context_event_count,
        "will_train": summary.will_train,
    }


def _sequence_length_summary(
    *,
    sequence_lengths: list[int],
    history_size: int,
) -> SequenceLengthSummary:
    if not sequence_lengths:
        return SequenceLengthSummary(
            min=0,
            p25=0.0,
            median=0.0,
            p75=0.0,
            max=0,
            mean=0.0,
            count_lte_history_size=0,
            count_gt_history_size=0,
        )

    sorted_lengths = sorted(sequence_lengths)
    return SequenceLengthSummary(
        min=sorted_lengths[0],
        p25=_percentile(sorted_lengths, fraction=0.25),
        median=_percentile(sorted_lengths, fraction=0.5),
        p75=_percentile(sorted_lengths, fraction=0.75),
        max=sorted_lengths[-1],
        mean=mean(sequence_lengths),
        count_lte_history_size=sum(
            1 for length in sequence_lengths if length <= history_size
        ),
        count_gt_history_size=sum(
            1 for length in sequence_lengths if length > history_size
        ),
    )


def _percentile(values: list[int], *, fraction: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    index = (len(values) - 1) * fraction
    lower_index = floor(index)
    upper_index = ceil(index)
    if lower_index == upper_index:
        return float(values[lower_index])
    lower_value = values[lower_index]
    upper_value = values[upper_index]
    weight = index - lower_index
    return float(lower_value + (upper_value - lower_value) * weight)


def _warmup_accounting_to_dict(accounting: WarmupAccounting) -> dict[str, int | float]:
    return {
        "events_seen": accounting.events_seen,
        "insufficient_history": accounting.insufficient_history,
        "events_eligible": accounting.events_eligible,
        "insufficient_history_rate": accounting.insufficient_history_rate,
    }


def _sequence_length_summary_to_dict(
    summary: SequenceLengthSummary,
) -> dict[str, int | float]:
    return {
        "min": summary.min,
        "p25": summary.p25,
        "median": summary.median,
        "p75": summary.p75,
        "max": summary.max,
        "mean": summary.mean,
        "count_lte_history_size": summary.count_lte_history_size,
        "count_gt_history_size": summary.count_gt_history_size,
    }


def _no_eligible_summary_to_dict(
    summary: NoEligibleSummary,
) -> dict[str, int | dict[int, int]]:
    return {
        "sequence_count": summary.sequence_count,
        "label_counts": dict(summary.label_counts),
    }


def _split_audit_summary_to_dict(
    summary: SplitAuditSummary,
) -> dict[str, int | dict[str, int | float]]:
    return {
        "sequence_count": summary.sequence_count,
        "event_count": summary.event_count,
        "normal_sequence_count": summary.normal_sequence_count,
        "anomalous_sequence_count": summary.anomalous_sequence_count,
        "warmup": _warmup_accounting_to_dict(summary.warmup),
    }


def _evaluation_warmup_from_sequences(
    *,
    sequences: Iterable[TemplateSequence],
    history_size: int,
) -> EvaluationWarmupSummary:
    """Count event-level evaluation membership across preserved chunks.

    Args:
        sequences (Iterable[TemplateSequence]): Chronological sequences to
            inspect.
        history_size (int): DeepLog history length used for warm-up counting.

    Returns:
        EvaluationWarmupSummary: Stable evaluation counts for the stream.
    """
    events_eligible = 0
    insufficient_history = 0
    anomalous_events = 0
    normal_events = 0
    for sequence in sequences:
        evaluation_mask = evaluation_event_mask_for_sequence(sequence)
        for event_index, is_evaluation_target in enumerate(evaluation_mask):
            if not is_evaluation_target:
                continue
            raw_label = (
                sequence.event_labels[event_index]
                if sequence.event_labels is not None
                else sequence.label
            )
            if event_index < history_size:
                insufficient_history += 1
                continue
            events_eligible += 1
            if is_anomalous_label(raw_label):
                anomalous_events += 1
            else:
                normal_events += 1
    return EvaluationWarmupSummary(
        events_eligible=events_eligible,
        insufficient_history=insufficient_history,
        anomalous_events=anomalous_events,
        normal_events=normal_events,
        post_cutoff_events_excluded=0,
    )


def _dataset_config_with_chunk_size(
    config: DatasetVariantConfig,
    *,
    chunk_size: int,
) -> DatasetVariantConfig:
    payload = serialise_config(config)
    sequence_payload = payload.get("sequence")
    if not isinstance(sequence_payload, dict):
        msg = "dataset config sequence payload must be a table."
        raise TypeError(msg)
    sequence_payload = {str(key): value for key, value in sequence_payload.items()}
    sequence_payload["chunk_size"] = chunk_size
    payload["sequence"] = sequence_payload
    return _decode_dataset_config(payload)


def _collect_hdfs_session_observations(
    *,
    rows: Iterable[object],
    label_for_group: Callable[[str], int | None],
    cutoff: int,
) -> list[HDFSSessionObservation]:
    sessions: dict[str, _HDFSSessionAccumulator] = {}
    for row in rows:
        entity_id = getattr(row, "entity_id", None)
        if entity_id is None:
            continue
        line_order = _structured_line_order(row)
        label = sessions.setdefault(
            str(entity_id),
            _HDFSSessionAccumulator(
                first_line_order=line_order,
                last_line_order=line_order,
            ),
        )
        label.first_line_order = min(label.first_line_order, line_order)
        label.last_line_order = max(label.last_line_order, line_order)
        label.event_count += 1
        if line_order < cutoff:
            label.pre_cutoff_event_count += 1
        else:
            label.post_cutoff_event_count += 1

    observations: list[HDFSSessionObservation] = []
    for entity_id, payload in sessions.items():
        observations.append(
            HDFSSessionObservation(
                entity_id=entity_id,
                first_line_order=payload.first_line_order,
                last_line_order=payload.last_line_order,
                label=label_for_group(entity_id),
                event_count=payload.event_count,
                pre_cutoff_event_count=payload.pre_cutoff_event_count,
                post_cutoff_event_count=payload.post_cutoff_event_count,
            ),
        )
    observations.sort(key=lambda item: item.first_line_order)
    return observations


def _hdfs_segments_for_policy(  # noqa: C901, PLR0911, PLR0912
    *,
    policy_name: str,
    session: HDFSSessionObservation,
    cutoff: int,
) -> list[tuple[SplitLabel, int, int | None]]:
    is_anomalous = is_anomalous_label(session.label)
    if policy_name == "split_partial_sequences":
        if session.post_cutoff_event_count == 0:
            return [
                (
                    SplitLabel.TRAIN if not is_anomalous else SplitLabel.IGNORED,
                    session.event_count,
                    session.label,
                ),
            ]
        if session.pre_cutoff_event_count == 0:
            return [(SplitLabel.TEST, session.event_count, session.label)]
        train_label = SplitLabel.TRAIN if not is_anomalous else SplitLabel.IGNORED
        segments: list[tuple[SplitLabel, int, int | None]] = []
        if session.pre_cutoff_event_count > 0:
            segments.append(
                (train_label, session.pre_cutoff_event_count, session.label),
            )
        if session.post_cutoff_event_count > 0:
            segments.append(
                (SplitLabel.TEST, session.post_cutoff_event_count, session.label),
            )
        return segments
    if policy_name == "assign_by_first_event":
        split_label = (
            SplitLabel.TRAIN
            if session.first_line_order < cutoff and not is_anomalous
            else (
                SplitLabel.IGNORED
                if session.first_line_order < cutoff
                else SplitLabel.TEST
            )
        )
        return [(split_label, session.event_count, session.label)]
    if policy_name == "assign_by_last_event":
        split_label = (
            SplitLabel.TRAIN
            if session.last_line_order < cutoff and not is_anomalous
            else (
                SplitLabel.IGNORED
                if session.last_line_order < cutoff
                else SplitLabel.TEST
            )
        )
        return [(split_label, session.event_count, session.label)]
    if policy_name == "first_100k_block_ids":
        if session.first_line_order < cutoff and not is_anomalous:
            split_label = SplitLabel.TRAIN
        elif session.first_line_order < cutoff:
            split_label = SplitLabel.IGNORED
        else:
            split_label = SplitLabel.TEST
        return [(split_label, session.event_count, session.label)]
    if policy_name == "normal_complete_sessions":
        if (
            not is_anomalous
            and session.first_line_order < cutoff
            and session.last_line_order < cutoff
        ):
            split_label = SplitLabel.TRAIN
        else:
            split_label = SplitLabel.TEST
        return [(split_label, session.event_count, session.label)]
    msg = f"Unsupported HDFS policy: {policy_name}"
    raise ValueError(msg)


def _summarise_hdfs_first_100k_policies(
    *,
    sessions: list[HDFSSessionObservation],
    cutoff: int,
    history_size: int,
    template_count: int,
) -> list[HDFSFirst100kPolicySummary]:
    summaries: list[HDFSFirst100kPolicySummary] = []
    total_sessions = len(sessions)
    paper_targets = {
        "train_normal": 4_855,
        "test_normal": 553_366,
        "test_anomalous": 15_200,
    }
    for policy_name in (
        "split_partial_sequences",
        "assign_by_first_event",
        "assign_by_last_event",
        "first_100k_block_ids",
        "normal_complete_sessions",
    ):
        segments = [
            segment
            for session in sessions
            for segment in _hdfs_segments_for_policy(
                policy_name=policy_name,
                session=session,
                cutoff=cutoff,
            )
        ]

        train_normal = sum(
            1
            for split_label, _, label in segments
            if split_label is SplitLabel.TRAIN and not is_anomalous_label(label)
        )
        train_anomalous = sum(
            1
            for split_label, _, label in segments
            if split_label is SplitLabel.TRAIN and is_anomalous_label(label)
        )
        ignored = sum(
            1 for split_label, _, _ in segments if split_label is SplitLabel.IGNORED
        )
        test_normal = sum(
            1
            for split_label, _, label in segments
            if split_label is SplitLabel.TEST and not is_anomalous_label(label)
        )
        test_anomalous = sum(
            1
            for split_label, _, label in segments
            if split_label is SplitLabel.TEST and is_anomalous_label(label)
        )
        no_eligible_sessions = sum(
            1 for _, event_count, _ in segments if event_count <= history_size
        )
        summaries.append(
            HDFSFirst100kPolicySummary(
                policy_name=policy_name,
                train_normal_sessions=train_normal,
                train_anomalous_sessions=train_anomalous,
                ignored_sessions=ignored,
                test_normal_sessions=test_normal,
                test_anomalous_sessions=test_anomalous,
                total_sessions=total_sessions,
                emitted_segment_count=len(segments),
                template_count=template_count,
                no_eligible_sessions=no_eligible_sessions,
                train_normal_delta=train_normal - paper_targets["train_normal"],
                test_normal_delta=test_normal - paper_targets["test_normal"],
                test_anomalous_delta=test_anomalous - paper_targets["test_anomalous"],
            ),
        )
    return summaries
