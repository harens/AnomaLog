"""Utilities for building template sequences from structured log lines.

The module groups parsed log lines into windows (entity, fixed-size, or
time-based) and decorates them with inferred templates and anomaly labels.
"""

from __future__ import annotations

import functools
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Iterator
from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import islice
from typing import TYPE_CHECKING, Protocol, TypeGuard, runtime_checkable

from typing_extensions import Self, override

from anomalog._sequence_split_policy import (
    RawEntrySplitRequest,
    build_raw_entry_split_plan,
    count_straddling_groups,
    resolve_straddling_group_label,
    split_rows_by_label,
)
from anomalog.parsers.structured.contracts import (
    StructuredLine,
    is_anomalous_label,
)
from anomalog.parsers.structured.parsers import ThunderbirdParser
from anomalog.representations import (
    SequenceRepresentation,
    SequenceRepresentationView,
    TRepresentation,
)
from anomalog.split_validation import validate_split_fractions

if TYPE_CHECKING:
    from anomalog.parsers.structured.contracts import StructuredSink
    from anomalog.parsers.structured.parquet.writer_worker import (
        EntityChronologyKey,
    )
    from anomalog.parsers.template.dataset import (
        ExtractedParameters,
        LogTemplate,
        TemplatedDataset,
        TemplateParser,
    )
    from experiments.models.base import SequenceSummary

_DENSE_RAW_SESSION_FIELD_COUNT = 3
_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _BeforeGroupingRawReplayState:
    """Mutable cache for the raw-file boundary used by before-grouping splits."""

    test_start_byte_offset: int | None = None
    raw_entry_split_summary: RawEntrySplitSummary | None = None


@dataclass(slots=True)
class _BeforeGroupingRawEntrySummaryCounts:
    """Accumulate raw-entry split diagnostics in one streaming pass."""

    train_normal_entry_count: int = 0
    train_anomalous_entry_count: int = 0
    test_normal_entry_count: int = 0
    test_anomalous_entry_count: int = 0
    straddling_group_count: int = 0

    @property
    def train_raw_entry_count(self) -> int:
        """Return the total number of raw entries assigned to train."""
        return self.train_normal_entry_count + self.train_anomalous_entry_count

    @property
    def test_raw_entry_count(self) -> int:
        """Return the total number of raw entries assigned to test."""
        return self.test_normal_entry_count + self.test_anomalous_entry_count


def _count_before_grouping_raw_group_entries(
    *,
    rows: Collection[StructuredLine],
    cutoff_entry_index: int,
) -> tuple[int, int, int, int, int]:
    """Count the raw entries emitted by one grouped window.

    Returns:
        tuple[int, int, int, int, int]: Train normal, train anomalous, test
            normal, test anomalous, and straddling-group counts for the group.
    """
    train_normal_entry_count = 0
    train_anomalous_entry_count = 0
    test_normal_entry_count = 0
    test_anomalous_entry_count = 0
    has_train_rows = False
    has_test_rows = False
    for row in rows:
        if row.line_order is None:
            continue
        if row.line_order < cutoff_entry_index:
            has_train_rows = True
            if is_anomalous_label(row.anomalous):
                train_anomalous_entry_count += 1
            else:
                train_normal_entry_count += 1
            continue
        has_test_rows = True
        if is_anomalous_label(row.anomalous):
            test_anomalous_entry_count += 1
        else:
            test_normal_entry_count += 1
    return (
        train_normal_entry_count,
        train_anomalous_entry_count,
        test_normal_entry_count,
        test_anomalous_entry_count,
        1 if has_train_rows and has_test_rows else 0,
    )


def _count_before_grouping_raw_entry_summary(
    groups: Iterable[Collection[StructuredLine]],
    *,
    cutoff_entry_index: int,
) -> _BeforeGroupingRawEntrySummaryCounts:
    """Count raw-entry split diagnostics in one streaming pass.

    Returns:
        _BeforeGroupingRawEntrySummaryCounts: Accumulated diagnostics for the
            grouped raw-entry replay path.
    """
    counts = _BeforeGroupingRawEntrySummaryCounts()
    for rows in groups:
        if not rows:
            continue
        (
            group_train_normal_entry_count,
            group_train_anomalous_entry_count,
            group_test_normal_entry_count,
            group_test_anomalous_entry_count,
            group_straddling_group_count,
        ) = _count_before_grouping_raw_group_entries(
            rows=rows,
            cutoff_entry_index=cutoff_entry_index,
        )
        counts.train_normal_entry_count += group_train_normal_entry_count
        counts.train_anomalous_entry_count += group_train_anomalous_entry_count
        counts.test_normal_entry_count += group_test_normal_entry_count
        counts.test_anomalous_entry_count += group_test_anomalous_entry_count
        counts.straddling_group_count += group_straddling_group_count
    return counts


@runtime_checkable
class _SupportsEntityCount(Protocol):
    """Sink capability for providing an exact entity count."""

    def load_entity_count(self) -> int | None:
        """Return a cached entity count when available."""


@runtime_checkable
class _SupportsEntityChronologyIndex(Protocol):
    """Sink capability for exposing entity chronology metadata."""

    def load_entity_chronology_index(self) -> dict[str, EntityChronologyKey]:
        """Return the materialised entity chronology index."""


@runtime_checkable
class _SupportsInlineLabelCache(Protocol):
    """Sink capability for exposing sparse inline anomaly labels."""

    def load_inline_label_cache(self) -> tuple[dict[int, int], dict[str, int]]:
        """Return sparse inline label caches keyed by line and group."""


@runtime_checkable
class _SupportsEntitySequenceSuffixScan(Protocol):
    """Sink capability for suffix-only entity grouping."""

    def iter_entity_sequences_from_line_order(
        self,
        min_line_order: int,
    ) -> Callable[[], Iterator[Collection[StructuredLine]]]:
        """Return a suffix-only entity grouping iterator factory."""


class SplitLabel(str, Enum):
    """Dataset split membership for a sequence.

    Attributes:
        TRAIN: Sequence belongs to the training split.
        TEST: Sequence belongs to the evaluation/test split.
        IGNORED: Sequence belongs to the fixed train pool but is not used for
            the current training prefix.
    """

    TRAIN = "train"
    TEST = "test"
    IGNORED = "ignored"


class SplitApplicationOrder(str, Enum):
    """When to apply a configured split relative to grouping.

    Attributes:
        AFTER_GROUPING: Apply the split after grouping has produced sequences.
        BEFORE_GROUPING: Apply the split on raw entries before grouping.
    """

    AFTER_GROUPING = "after_grouping"
    BEFORE_GROUPING = "before_grouping"


class StraddlingGroupPolicy(str, Enum):
    """How to handle grouped rows that cross a raw-entry split boundary.

    Attributes:
        SPLIT_PARTIAL_SEQUENCES: Emit one sequence per contiguous segment.
        ASSIGN_BY_FIRST_EVENT: Assign the whole group by the first segment.
        ASSIGN_BY_LAST_EVENT: Assign the whole group by the last segment.
        DROP_STRADDLERS: Drop groups that span both sides of the split.
    """

    SPLIT_PARTIAL_SEQUENCES = "split_partial_sequences"
    ASSIGN_BY_FIRST_EVENT = "assign_by_first_event"
    ASSIGN_BY_LAST_EVENT = "assign_by_last_event"
    DROP_STRADDLERS = "drop_straddlers"


class FixedWindowBasis(str, Enum):
    """What positional basis to use for fixed-size windows."""

    COMPACTED_ROWS = "compacted_rows"
    RAW_POSITIONS = "raw_positions"


class RawEntrySplitMode(str, Enum):
    """Chronological raw-entry split modes supported by sequence builders.

    Attributes:
        PREFIX_COUNT: Split by the first N raw entries.
        PREFIX_FRACTION: Split by the first fraction of raw entries.
        PREFIX_NORMAL_FRACTION: Split by the first fraction of normal entries.
    """

    PREFIX_COUNT = "raw_entry_prefix_count"
    PREFIX_FRACTION = "raw_entry_prefix_fraction"
    PREFIX_NORMAL_FRACTION = "raw_entry_prefix_normal_fraction"


@dataclass(slots=True, frozen=True)
class RawEntrySplitSummary:
    """Audit summary for a chronological raw-entry split.

    Attributes:
        split_mode (str): Configured raw-entry split mode.
        application_order (str): Whether the split was applied before or after
            grouping.
        cutoff_entry_index (int): Zero-based raw-entry cutoff where the test
            suffix begins.
        train_raw_entry_count (int): Raw entries assigned to train.
        train_normal_entry_count (int): Normal raw entries assigned to train.
        train_anomalous_entry_count (int): Anomalous raw entries assigned to train.
        test_raw_entry_count (int): Raw entries assigned to test.
        test_normal_entry_count (int): Normal raw entries assigned to test.
        test_anomalous_entry_count (int): Anomalous raw entries assigned to test.
        ignored_raw_entry_count (int): Raw entries withheld from both train and test.
        ignored_normal_entry_count (int): Normal raw entries withheld.
        ignored_anomalous_entry_count (int): Anomalous raw entries withheld.
        straddling_group_count (int): Number of grouped windows that crossed the
            split boundary.
        straddling_group_policy (str | None): Policy applied to straddling groups.
    """

    split_mode: str
    application_order: str
    cutoff_entry_index: int
    train_raw_entry_count: int
    train_normal_entry_count: int
    train_anomalous_entry_count: int
    test_raw_entry_count: int
    test_normal_entry_count: int
    test_anomalous_entry_count: int
    ignored_raw_entry_count: int = 0
    ignored_normal_entry_count: int = 0
    ignored_anomalous_entry_count: int = 0
    straddling_group_count: int = 0
    straddling_group_policy: str | None = None

    def as_dict(self) -> dict[str, int | str | None]:
        """Return a JSON-friendly representation.

        Returns:
            dict[str, int | str | None]: Serialisable split summary payload.
        """
        return {
            "split_mode": self.split_mode,
            "application_order": self.application_order,
            "cutoff_entry_index": self.cutoff_entry_index,
            "train_raw_entry_count": self.train_raw_entry_count,
            "train_normal_entry_count": self.train_normal_entry_count,
            "train_anomalous_entry_count": self.train_anomalous_entry_count,
            "test_raw_entry_count": self.test_raw_entry_count,
            "test_normal_entry_count": self.test_normal_entry_count,
            "test_anomalous_entry_count": self.test_anomalous_entry_count,
            "ignored_raw_entry_count": self.ignored_raw_entry_count,
            "ignored_normal_entry_count": self.ignored_normal_entry_count,
            "ignored_anomalous_entry_count": self.ignored_anomalous_entry_count,
            "straddling_group_count": self.straddling_group_count,
            "straddling_group_policy": self.straddling_group_policy,
        }


@dataclass(slots=True, frozen=True)
class SequenceSplitSummary:
    """Serialisable summary of requested versus effective split behavior.

    The requested train fraction may not equal the effective one after
    grouping-specific eligibility rules are applied. Persisting both protects
    downstream experiment manifests from silently overstating how much data was
    actually available for training.

    Attributes:
        requested_train_fraction (float): Requested fraction provided by the
            caller.
        requested_test_fraction (float): Requested test suffix fraction
            provided by the caller.
        train_on_normal_entities_only (bool | None): Whether training was restricted to
            normal entities only. Only applicable to entity grouping; `None` otherwise.
        train_pool_sequence_count (int): Number of sequences in the
            chronological train candidate window before detector-specific
            filtering is applied.
        ineligible_train_pool_count (int): Number of sequences in the train
            pool that were ineligible for training under the current policy.
        realised_train_sequence_count (int): Number of sequences actually used
            for training after any detector-specific filtering.
        excluded_from_train_count (int): Number of sequences withheld from the
            train pool before scoring, including the ignored middle band and
            any detector-ineligible prefix items.
        eligible_train_sequence_count (int): Number of sequences in the
            denominator for the effective train-fraction calculation. In
            entity-grouped mode this is the fixed chronological train pool, or
            the normal-only subset of that pool when normal-only training is
            enabled.
        ignored_sequence_count (int): Number of sequences withheld from the
            train pool because they fell outside the requested train prefix or
            were ineligible under the current filtering policy.
        effective_train_fraction_of_eligible (float): Realised train fraction
            over the eligible set.
        effective_train_fraction_overall (float): Realised train fraction over
            the full generated sequence population.
    """

    requested_train_fraction: float
    requested_test_fraction: float
    train_on_normal_entities_only: bool | None
    train_pool_sequence_count: int
    ineligible_train_pool_count: int
    realised_train_sequence_count: int
    excluded_from_train_count: int
    eligible_train_sequence_count: int
    ignored_sequence_count: int
    effective_train_fraction_of_eligible: float
    effective_train_fraction_overall: float

    def as_dict(self) -> dict[str, int | float | bool | str]:
        """Return a stable JSON-friendly representation.

        Returns:
            dict[str, int | float | bool | str]: Serialised split summary.
        """
        results: dict[str, int | float | bool | str] = {
            "requested_train_fraction": self.requested_train_fraction,
            "requested_test_fraction": self.requested_test_fraction,
            "train_pool_sequence_count": self.train_pool_sequence_count,
            "ineligible_train_pool_count": self.ineligible_train_pool_count,
            "realised_train_sequence_count": self.realised_train_sequence_count,
            "excluded_from_train_count": self.excluded_from_train_count,
            "eligible_train_sequence_count": self.eligible_train_sequence_count,
            "ignored_sequence_count": self.ignored_sequence_count,
            "effective_train_fraction_of_eligible": (
                self.effective_train_fraction_of_eligible
            ),
            "effective_train_fraction_overall": self.effective_train_fraction_overall,
        }

        if self.train_on_normal_entities_only is not None:
            results["train_on_normal_entities_only"] = (
                self.train_on_normal_entities_only
            )

        return results


@dataclass(slots=True, frozen=True)
class SequenceSplitCounts:
    """Exact split counts for a concrete sequence builder.

    Attributes:
        total_count (int): Total emitted sequence count.
        train_count (int): Count assigned to the current train prefix.
        ignored_count (int): Count withheld between train and test.
        test_count (int): Count assigned to the fixed test suffix.
    """

    total_count: int
    train_count: int
    ignored_count: int
    test_count: int


@dataclass(slots=True, frozen=True)
class TemplateSequence:
    """Grouped log window before any model-specific representation is applied.

    This keeps sequence semantics such as event ordering, labels, and entity
    membership. Model inputs derived from it live in `SequenceSample`.

    Attributes:
        events (list[tuple[str, list[str], int | None]]): Ordered sequence events
            as `(template, parameters, dt_prev_ms)` tuples.
        label (int): Sequence-level anomaly label derived from rows and group
            labels.
        entity_ids (list[str]): Unique entity ids present in the window in
            first-seen order.
        window_id (int): Stable window identifier assigned by the builder.
        split_label (SplitLabel): Dataset split assigned to the sequence.
        event_labels (tuple[int | None, ...] | None): Optional per-event anomaly
            labels aligned positionally with `events`. When present, each entry
            may be `None` if that event has no direct label.
        training_event_mask (tuple[bool, ...] | None): Optional per-event
            eligibility mask for training-target selection. This is used when a
            preserved chronological chunk must stay intact even though only a
            subset of its events are valid training targets.
        evaluation_event_mask (tuple[bool, ...] | None): Optional per-event
            eligibility mask for scoring targets. This is used when a
            preserved chronological chunk must stay intact even though only a
            subset of its events belong to the evaluation split.
        continuous_context (bool): Whether adjacent sequences should be treated
            as a single chronological stream for model state carryover.
    """

    events: list[
        tuple[str, list[str], int | None]
    ]  # (template, parameters, dt_prev_ms)
    label: int
    entity_ids: list[str]  # unique entity ids present (may be empty)
    window_id: int
    split_label: SplitLabel = SplitLabel.TRAIN
    event_labels: tuple[int | None, ...] | None = None
    training_event_mask: tuple[bool, ...] | None = None
    evaluation_event_mask: tuple[bool, ...] | None = None
    continuous_context: bool = False

    def __post_init__(self) -> None:
        """Validate that any event labels stay aligned with the events.

        Raises:
            ValueError: If `event_labels` is provided with a different length
                from `events`.
        """
        if self.event_labels is not None and len(self.event_labels) != len(self.events):
            msg = (
                "TemplateSequence.event_labels must match the number of events "
                "when provided."
            )
            raise ValueError(msg)
        if self.training_event_mask is not None and len(
            self.training_event_mask,
        ) != len(self.events):
            msg = (
                "TemplateSequence.training_event_mask must match the number of "
                "events when provided."
            )
            raise ValueError(msg)
        if self.evaluation_event_mask is not None and len(
            self.evaluation_event_mask,
        ) != len(self.events):
            msg = (
                "TemplateSequence.evaluation_event_mask must match the number "
                "of events when provided."
            )
            raise ValueError(msg)

    @property
    def templates(self) -> list[str]:
        """Return the ordered template strings for this sequence."""
        return [tpl for tpl, _, _ in self.events]

    @property
    def sole_entity_id(self) -> str | None:
        """Return the entity id when the sequence belongs to exactly one entity.

        If multiple entities appear in the window, None is returned to avoid
        implying a single owning entity.
        """
        if len(self.entity_ids) == 1:
            return self.entity_ids[0]
        return None


@dataclass(slots=True, frozen=True)
class SequenceBuilder(ABC, Iterable[TemplateSequence]):
    """Common sequence-building behavior shared across grouping strategies.

    Sequence builders stay lazy so expensive grouping, template inference, and
    label resolution only happen when a caller iterates. The shared base also
    centralises split assignment so experiment manifests can describe train/test
    semantics consistently across grouping modes.

    Attributes:
        sink (StructuredSink): Structured sink supplying grouped rows.
        infer_template (Callable[[str], tuple[LogTemplate, ExtractedParameters]]):
            Template inference function for row message text.
    label_for_group (Callable[[str], int | None]): Group-level anomaly label
        lookup by entity id.
        template_parser (TemplateParser | None): Optional parser object kept
            alongside `infer_template` so optimisation paths can inspect parser
            capabilities without peeking at bound-method metadata.
        split_mode (RawEntrySplitMode | None): Raw-entry split mode used for
            special reproduction protocols. `None` preserves the legacy
            sequence-fraction split behaviour.
        split_application_order (SplitApplicationOrder): Whether the split is
            applied before or after grouping.
        straddling_group_policy (StraddlingGroupPolicy): Policy for grouped rows
            that cross a raw-entry split boundary.
        train_entry_count (int | None): Requested raw-entry prefix length when
            `split_mode = PREFIX_COUNT`.
        train_entry_fraction (float | None): Requested raw-entry prefix
            fraction when `split_mode = PREFIX_FRACTION`.
        train_normal_entry_fraction (float | None): Requested normal-entry
            prefix fraction when `split_mode = PREFIX_NORMAL_FRACTION`.
        stream_chunk_size (int | None): Optional chunk size used by stream
            grouping strategies.
        train_frac (float): Requested training fraction for the builder.
        test_frac (float): Fixed test suffix fraction.
    """

    sink: StructuredSink
    infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]]
    label_for_group: Callable[[str], int | None]
    template_parser: TemplateParser | None = None
    raw_replay_state: _BeforeGroupingRawReplayState = field(
        default_factory=_BeforeGroupingRawReplayState,
        init=False,
        repr=False,
        compare=False,
    )
    split_mode: RawEntrySplitMode | None = None
    split_application_order: SplitApplicationOrder = (
        SplitApplicationOrder.AFTER_GROUPING
    )
    straddling_group_policy: StraddlingGroupPolicy = (
        StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES
    )
    train_entry_count: int | None = None
    train_entry_fraction: float | None = None
    train_normal_entry_fraction: float | None = None
    stream_chunk_size: int | None = None
    train_frac: float = 0.2
    test_frac: float = 0.8

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        """Validate the requested split fractions and raw-entry split inputs.

        Raises:
            ValueError: If the requested split settings are inconsistent.
        """
        if (
            self.split_mode is None
            or self.split_application_order == SplitApplicationOrder.AFTER_GROUPING
        ):
            validate_split_fractions(
                train_frac=self.train_frac,
                test_frac=self.test_frac,
            )
        if self.split_mode == RawEntrySplitMode.PREFIX_COUNT:
            if self.train_entry_count is None or self.train_entry_count < 0:
                msg = "train_entry_count must be a non-negative integer."
                raise ValueError(msg)
            if self.split_application_order == SplitApplicationOrder.AFTER_GROUPING:
                msg = (
                    "raw-entry count splits must use "
                    "split_application_order = BEFORE_GROUPING."
                )
                raise ValueError(msg)
        elif self.split_mode == RawEntrySplitMode.PREFIX_FRACTION:
            if self.train_entry_fraction is None:
                msg = (
                    "train_entry_fraction must be provided for raw-entry "
                    "fraction splits."
                )
                raise ValueError(msg)
            if self.train_entry_fraction <= 0.0 or self.train_entry_fraction > 1.0:
                msg = "train_entry_fraction must be between 0 and 1."
                raise ValueError(msg)
            if self.split_application_order == SplitApplicationOrder.AFTER_GROUPING:
                msg = (
                    "raw-entry fraction splits must use "
                    "split_application_order = BEFORE_GROUPING."
                )
                raise ValueError(msg)
        elif self.split_mode == RawEntrySplitMode.PREFIX_NORMAL_FRACTION:
            if self.train_normal_entry_fraction is None:
                msg = (
                    "train_normal_entry_fraction must be provided for raw-entry "
                    "normal-fraction splits."
                )
                raise ValueError(msg)
            if (
                self.train_normal_entry_fraction <= 0.0
                or self.train_normal_entry_fraction > 1.0
            ):
                msg = "train_normal_entry_fraction must be between 0 and 1."
                raise ValueError(msg)
            if self.split_application_order == SplitApplicationOrder.AFTER_GROUPING:
                msg = (
                    "raw-entry normal-fraction splits must use "
                    "split_application_order = BEFORE_GROUPING."
                )
                raise ValueError(msg)
            if (
                self.straddling_group_policy
                != StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES
            ):
                msg = (
                    "raw-entry normal-fraction splits only support "
                    "split_partial_sequences."
                )
                raise ValueError(msg)
        elif (
            self.split_mode is not None
            and self.split_application_order == SplitApplicationOrder.BEFORE_GROUPING
        ):
            if self.straddling_group_policy not in {
                StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES,
                StraddlingGroupPolicy.ASSIGN_BY_FIRST_EVENT,
                StraddlingGroupPolicy.ASSIGN_BY_LAST_EVENT,
                StraddlingGroupPolicy.DROP_STRADDLERS,
            }:
                msg = (
                    "Unsupported straddling policy: "
                    f"{self.straddling_group_policy.value}"
                )
                raise ValueError(msg)
        if self.stream_chunk_size is not None and self.stream_chunk_size <= 0:
            msg = "stream_chunk_size must be a positive integer."
            raise ValueError(msg)

    def _iter_source_order_rows(self) -> Iterator[StructuredLine]:
        """Yield structured rows in raw-entry order.

        Yields:
            StructuredLine: Structured rows ordered by `line_order`.

        The parquet sink already knows how to merge entity buckets by
        `line_order`, so this helper preserves source chronology without
        forcing callers to materialise the full dataset first.
        """
        source_order_iter = self.sink.iter_structured_lines_in_source_order()
        yield from source_order_iter()
        return

    def _build_row_split_labels(
        self,
    ) -> tuple[dict[int, SplitLabel], RawEntrySplitSummary | None]:
        """Build raw-entry split labels keyed by line order.

        Returns:
            tuple[dict[int, SplitLabel], RawEntrySplitSummary | None]: Row-level
                split labels and an audit summary when a raw-entry split is
                active.
        """
        if (
            self.split_mode is None
            or self.split_application_order != SplitApplicationOrder.BEFORE_GROUPING
        ):
            return {}, None

        plan = build_raw_entry_split_plan(
            list(self._iter_source_order_rows()),
            request=RawEntrySplitRequest(
                split_mode=self.split_mode.value,
                application_order=self.split_application_order.value,
                train_entry_count=self.train_entry_count,
                train_entry_fraction=self.train_entry_fraction,
                train_normal_entry_fraction=self.train_normal_entry_fraction,
            ),
        )
        summary = RawEntrySplitSummary(
            split_mode=plan.summary.split_mode,
            application_order=plan.summary.application_order,
            cutoff_entry_index=plan.summary.cutoff_entry_index,
            train_raw_entry_count=plan.summary.train_raw_entry_count,
            train_normal_entry_count=plan.summary.train_normal_entry_count,
            train_anomalous_entry_count=plan.summary.train_anomalous_entry_count,
            test_raw_entry_count=plan.summary.test_raw_entry_count,
            test_normal_entry_count=plan.summary.test_normal_entry_count,
            test_anomalous_entry_count=plan.summary.test_anomalous_entry_count,
            ignored_raw_entry_count=plan.summary.ignored_raw_entry_count,
            ignored_normal_entry_count=plan.summary.ignored_normal_entry_count,
            ignored_anomalous_entry_count=plan.summary.ignored_anomalous_entry_count,
        )
        return (
            {
                line_order: SplitLabel(label)
                for line_order, label in plan.row_labels.items()
            },
            summary,
        )

    def _split_counts(self, total_count: int) -> SequenceSplitCounts:
        """Return train, ignored, and test counts for one chronological split.

        Args:
            total_count (int): Total emitted sequence count.

        Returns:
            SequenceSplitCounts: Exact split counts for the requested split
                configuration.
        """
        if total_count <= 0:
            return SequenceSplitCounts(
                total_count=0,
                train_count=0,
                ignored_count=0,
                test_count=0,
            )
        test_count = min(total_count, math.ceil(self.test_frac * total_count))
        train_count = min(
            total_count - test_count,
            math.ceil(self.train_frac * total_count),
        )
        ignored_count = total_count - train_count - test_count
        return SequenceSplitCounts(
            total_count=total_count,
            train_count=train_count,
            ignored_count=ignored_count,
            test_count=test_count,
        )

    def train_fraction_eligible_sequence_count(
        self,
        *,
        sequence_summary: SequenceSummary,
    ) -> int:
        """Return the denominator for effective train-fraction accounting.

        Args:
            sequence_summary (SequenceSummary): Aggregate split and label counts.

        Returns:
            int: Count of sequences considered eligible when reporting the
                realised train fraction for this grouping strategy. This is not
                necessarily the number of sequences that were actually assigned
                to train.
        """
        if not self.test_frac:
            return sequence_summary.sequence_count
        return sequence_summary.sequence_count - sequence_summary.test_sequence_count

    def split_count_hint(self) -> SequenceSplitCounts | None:
        """Return a cheap exact split-count summary when the builder knows it.

        Returns:
            SequenceSplitCounts | None: Exact split counts when cheaply
                available, otherwise `None`.
        """
        del self
        return None

    def with_split_fractions(
        self,
        train_frac: float,
        test_frac: float,
    ) -> Self:
        """Return a copy with both split fractions updated together.

        Args:
            train_frac (float): Requested fraction of the total population to
                assign to the train prefix.
            test_frac (float): Requested fraction reserved for the fixed test
                suffix.

        Returns:
            Self: Copy with updated split fractions.
        """
        return replace(self, train_frac=train_frac, test_frac=test_frac)

    def represent_with(
        self,
        representation: SequenceRepresentation[TRepresentation],
    ) -> SequenceRepresentationView[TRepresentation]:
        """Return a lazy builder that applies a representation per sequence.

        Args:
            representation (SequenceRepresentation[TRepresentation]): Sequence
                representation to apply lazily to each built sequence.

        Returns:
            SequenceRepresentationView[TRepresentation]: Lazy represented view of
                the generated sequences.
        """
        return SequenceRepresentationView(sequences=self, representation=representation)

    def train_sequence_count_unit_hint(self) -> str | None:
        """Return a human-readable unit label for train-count progress.

        This is intended for progress reporting only. Builders should return a
        unit when it clarifies what the bounded train count represents, such as
        ``"entities"`` for entity grouping.

        Returns:
            str | None: Unit label for train-count progress when useful,
                otherwise ``None``.
        """
        del self
        return None

    def split_summary_train_on_normal_entities_only(self) -> bool | None:
        """Return split-summary metadata for entity-only normal training.

        Returns:
            bool | None: Whether train was restricted to normal entities only,
            or `None` when that concept does not apply to this builder.
        """
        del self
        return None

    def iter_training_sequences(self) -> Iterator[TemplateSequence]:
        """Yield the training slice used by model fitting."""
        for sequence in self:
            if sequence.split_label is SplitLabel.TRAIN:
                yield sequence

    def iter_test_sequences(self) -> Iterator[TemplateSequence]:
        """Yield the test slice used by model scoring."""
        for sequence in self:
            if sequence.split_label is SplitLabel.TEST:
                yield sequence

    @abstractmethod
    def __iter__(self) -> Iterator[TemplateSequence]:
        """Iterate over template sequences yielded by the configured grouping.

        Returns:
            Iterator[TemplateSequence]: Iterator yielding grouped and
                template-enriched sequences.
        """
        ...

    def build_split_summary(
        self,
        *,
        sequence_summary: SequenceSummary,
    ) -> SequenceSplitSummary:
        """Describe requested versus effective split semantics for one run.

        Args:
            sequence_summary (SequenceSummary): Aggregate split and label counts.

        Returns:
            SequenceSplitSummary: Requested and effective split metrics.
        """
        eligible_train_sequence_count = self.train_fraction_eligible_sequence_count(
            sequence_summary=sequence_summary,
        )
        realised_train_sequence_count = sequence_summary.train_sequence_count
        train_pool_sequence_count = (
            realised_train_sequence_count + sequence_summary.ignored_sequence_count
        )
        ineligible_train_pool_count = (
            sum(
                count
                for label, count in sequence_summary.ignored_label_counts.items()
                if is_anomalous_label(label)
            )
            if self.split_summary_train_on_normal_entities_only()
            else 0
        )
        excluded_from_train_count = (
            train_pool_sequence_count - realised_train_sequence_count
        )
        effective_train_fraction_of_eligible = (
            realised_train_sequence_count / eligible_train_sequence_count
            if eligible_train_sequence_count
            else 0.0
        )
        effective_train_fraction_overall = (
            realised_train_sequence_count / sequence_summary.sequence_count
            if sequence_summary.sequence_count
            else 0.0
        )
        return SequenceSplitSummary(
            requested_train_fraction=self.train_frac,
            requested_test_fraction=self.test_frac,
            train_on_normal_entities_only=(
                self.split_summary_train_on_normal_entities_only()
            ),
            train_pool_sequence_count=train_pool_sequence_count,
            ineligible_train_pool_count=ineligible_train_pool_count,
            realised_train_sequence_count=realised_train_sequence_count,
            excluded_from_train_count=excluded_from_train_count,
            eligible_train_sequence_count=eligible_train_sequence_count,
            ignored_sequence_count=sequence_summary.ignored_sequence_count,
            effective_train_fraction_of_eligible=round(
                effective_train_fraction_of_eligible,
                8,
            ),
            effective_train_fraction_overall=round(
                effective_train_fraction_overall,
                8,
            ),
        )

    def build_raw_entry_split_summary(self) -> RawEntrySplitSummary | None:
        """Return diagnostics for a configured raw-entry split, if any.

        Returns:
            RawEntrySplitSummary | None: Raw-entry split diagnostics when a
                before-grouping split is configured, otherwise `None`.
        """
        if (
            self.split_mode is None
            or self.split_application_order != SplitApplicationOrder.BEFORE_GROUPING
        ):
            return None
        cached_summary = self.raw_replay_state.raw_entry_split_summary
        if cached_summary is not None:
            return cached_summary
        streamed_summary = self._build_before_grouping_raw_entry_split_summary()
        if streamed_summary is not None:
            self.raw_replay_state.raw_entry_split_summary = streamed_summary
            return streamed_summary
        row_labels, summary = self._build_row_split_labels()
        if summary is None:
            return None
        row_label_values = {
            line_order: label.value for line_order, label in row_labels.items()
        }
        fallback_summary = replace(
            summary,
            straddling_group_count=count_straddling_groups(
                self.iter_grouped_rows(),
                row_label_values,
            ),
            straddling_group_policy=self.straddling_group_policy.value,
        )
        self.raw_replay_state.raw_entry_split_summary = fallback_summary
        return fallback_summary

    def _build_before_grouping_raw_entry_split_summary(
        self,
    ) -> RawEntrySplitSummary | None:
        """Return a specialised before-grouping summary when available."""
        del self
        return None

    @abstractmethod
    def iter_grouped_rows(self) -> Iterator[Collection[StructuredLine]]:
        """Return grouped rows for the configured strategy.

        Returns:
            Iterator[Collection[StructuredLine]]: Iterator over grouped windows
                of structured rows.
        """
        ...

    def _build_sequence(  # noqa: C901, PLR0913
        self,
        window_id: int,
        rows: Collection[StructuredLine],
        infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]],
        label_for_group: Callable[[str], int | None],
        split_label: SplitLabel,
        *,
        allow_group_label_fallback: bool = True,
        sequence_label: int | None = None,
        group_label_is_anomalous: bool | None = None,
        training_event_mask: tuple[bool, ...] | None = None,
        evaluation_event_mask: tuple[bool, ...] | None = None,
        continuous_context: bool = False,
    ) -> TemplateSequence | None:
        """Convert a non-empty row window into a labelled template sequence.

        Args:
            window_id (int): Monotonic window identifier for the generated
                sequence.
            rows (Collection[StructuredLine]): Structured rows in the current
                window.
            infer_template (Callable[[str], tuple[LogTemplate, ExtractedParameters]]):
                Template inference function for untemplated row text.
            label_for_group (Callable[[str], int | None]): Group-level anomaly
                label lookup by entity id.
            split_label (SplitLabel): Assigned dataset split for the sequence.
            allow_group_label_fallback (bool): Whether entity-level anomaly
                labels may promote an otherwise normal window to anomalous.
            sequence_label (int | None): Optional precomputed sequence label
                derived from the raw raw-line window.
            group_label_is_anomalous (bool | None): Optional precomputed entity
                label verdict to reuse when the caller has already resolved it.
            training_event_mask (tuple[bool, ...] | None): Optional per-event
                training-target eligibility mask for preserved chronological
                chunks.
            evaluation_event_mask (tuple[bool, ...] | None): Optional per-event
                scoring-target eligibility mask for preserved chronological
                chunks.
            continuous_context (bool): Whether the emitted sequence belongs to a
                stream that should carry context across chunk boundaries.

        Returns:
            TemplateSequence | None: Built sequence, or `None` for empty
                windows. Sequence labels are derived from both inline row
                labels and resolved group labels under the shared anomaly
                semantics.
        """
        if not rows:
            return None

        events: list[tuple[str, list[str], int | None]] = []
        event_labels: list[int | None] | None = None
        seq_label = 1 if is_anomalous_label(sequence_label) else 0
        prev_ts: int | None = None

        unique_ids = self._entity_ids_for_rows(rows)

        for r in rows:
            template, params = infer_template(r.untemplated_message_text)
            if r.raw_parameters is not None:
                event_parameters = list(r.raw_parameters)
            else:
                event_parameters = _parameters_as_list(params)
            dt, prev_ts = self._compute_dt(prev_ts, r.timestamp_unix_ms)

            events.append((template, event_parameters, dt))
            if r.anomalous is not None:
                if event_labels is None:
                    event_labels = [None] * (len(events) - 1)
                event_labels.append(r.anomalous)
            elif event_labels is not None:
                event_labels.append(None)

            if seq_label == 1:
                continue

            if is_anomalous_label(r.anomalous):
                seq_label = 1
                continue

            if not allow_group_label_fallback:
                continue

            if _should_promote_from_group_label(
                group_label_is_anomalous=group_label_is_anomalous,
                label_for_group=label_for_group,
                entity_id=r.entity_id,
            ):
                seq_label = 1

        return TemplateSequence(
            events=events,
            label=seq_label,
            entity_ids=unique_ids,
            window_id=window_id,
            split_label=split_label,
            event_labels=(tuple(event_labels) if event_labels is not None else None),
            training_event_mask=training_event_mask,
            evaluation_event_mask=evaluation_event_mask,
            continuous_context=continuous_context,
        )

    def _build_sequences_for_group(  # noqa: C901, PLR0913
        self,
        *,
        window_id: int,
        rows: Collection[StructuredLine],
        infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]],
        label_for_group: Callable[[str], int | None],
        split_label: SplitLabel,
        row_labels: dict[int, SplitLabel] | None = None,
        train_only_normal_entities: bool = False,
    ) -> Iterator[TemplateSequence]:
        """Build one or more template sequences for a grouped window.

        Args:
            window_id (int): Monotonic window identifier for the grouped rows.
            rows (Collection[StructuredLine]): Structured rows in the grouped
                window.
            infer_template (Callable[[str], tuple[LogTemplate, ExtractedParameters]]):
                Template inference function used to mine each event.
            label_for_group (Callable[[str], int | None]): Group-level anomaly
                lookup.
            split_label (SplitLabel): Default split label for the grouped rows.
            row_labels (dict[int, SplitLabel] | None): Optional raw-entry split
                labels keyed by `line_order`.
            train_only_normal_entities (bool): Whether train-side groups should
                be forced to ignored when the entity is anomalous.

        When raw-entry splitting is enabled before grouping, grouped rows can
        be segmented into multiple sequences depending on the configured
        straddling policy.

        Yields:
            TemplateSequence: One or more sequences derived from the grouped
                rows.
        """
        if not rows:
            return
        if (
            row_labels is None
            or self.split_application_order == SplitApplicationOrder.AFTER_GROUPING
        ):
            seq = self._build_sequence(
                window_id,
                rows,
                infer_template,
                label_for_group,
                split_label,
                allow_group_label_fallback=False,
            )
            if seq is not None:
                yield seq
            return

        row_label_values = {
            line_order: label.value for line_order, label in row_labels.items()
        }
        segments = list(split_rows_by_label(rows, row_label_values))
        if not segments:
            return

        if (
            self.straddling_group_policy
            == StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES
        ):
            for offset, (segment_label, segment_rows) in enumerate(segments):
                effective_label = SplitLabel(segment_label)
                if (
                    train_only_normal_entities
                    and effective_label is SplitLabel.TRAIN
                    and any(
                        row.entity_id is not None
                        and is_anomalous_label(label_for_group(row.entity_id))
                        for row in segment_rows
                    )
                ):
                    effective_label = SplitLabel.IGNORED
                seq = self._build_sequence(
                    window_id + offset,
                    segment_rows,
                    infer_template,
                    label_for_group,
                    effective_label,
                )
                if seq is not None:
                    yield seq
            return

        split_label_value = resolve_straddling_group_label(
            self.straddling_group_policy.value,
            segments,
        )
        if split_label_value is None:
            return
        split_label = SplitLabel(split_label_value)

        seq = self._build_sequence(
            window_id,
            rows,
            infer_template,
            label_for_group,
            (
                SplitLabel.IGNORED
                if (
                    train_only_normal_entities
                    and split_label is SplitLabel.TRAIN
                    and any(
                        row.entity_id is not None
                        and is_anomalous_label(label_for_group(row.entity_id))
                        for row in rows
                    )
                )
                else split_label
            ),
        )
        if seq is not None:
            yield seq

    def _entity_ids_for_rows(self, rows: Collection[StructuredLine]) -> list[str]:
        """Return unique entity ids for one window in first-seen order.

        Args:
            rows (Collection[StructuredLine]): Structured rows belonging to one
                grouped window.

        Returns:
            list[str]: Unique entity ids in first-seen order.
        """
        del self
        seen: set[str] = set()
        entity_ids: list[str] = []
        for row in rows:
            if row.entity_id is None or row.entity_id in seen:
                continue
            seen.add(row.entity_id)
            entity_ids.append(row.entity_id)
        return entity_ids

    @staticmethod
    def _compute_dt(
        prev_ts: int | None,
        ts: int | None,
    ) -> tuple[int | None, int | None]:
        """Compute delta time between events while preserving previous ts.

        Args:
            prev_ts (int | None): Previous event timestamp in milliseconds.
            ts (int | None): Current event timestamp in milliseconds.

        Examples:
            >>> SequenceBuilder._compute_dt(None, 1000)
            (None, 1000)
            >>> SequenceBuilder._compute_dt(1000, 1250)
            (250, 1250)
            >>> SequenceBuilder._compute_dt(2000, None)
            (None, 2000)

        Returns:
            tuple[int | None, int | None]: Delta from the previous timestamp and
                the updated previous timestamp to carry forward.
        """
        if ts is None:
            return None, prev_ts
        if prev_ts is None:
            return None, ts
        return int(ts) - int(prev_ts), ts

    @staticmethod
    def _count_fixed_windows(
        *,
        sink: StructuredSink,
        window_size: int,
        step: int | None,
    ) -> int:
        """Estimate number of fixed windows given window and step sizes.

        Args:
            sink (StructuredSink): Structured sink providing the total row count.
            window_size (int): Number of rows per window.
            step (int | None): Step between successive windows, or `None` to use
                `window_size`.

        Examples:
            >>> class _Sink:
            ...     def count_rows(self):
            ...         return 10
            ...
            >>> sb = FixedSequenceBuilder(
            ...     sink=_Sink(),
            ...     infer_template=lambda s: (s, ()),
            ...     label_for_group=lambda _: 0,
            ...     window_size=4,
            ...     step=2,
            ... )
            >>> sb.count_windows()
            4

        Returns:
            int: Estimated count of fixed windows for the sink.
        """
        if window_size <= 0:
            return 0
        step = step or window_size
        if step <= 0:
            return 0
        n = sink.count_rows()
        if n <= 0:
            return 0

        if n < window_size:
            return 0
        if n == window_size:
            return 1
        return 1 + ((n - window_size) // step)

    @staticmethod
    def _count_time_windows(
        *,
        sink: StructuredSink,
        time_span_ms: int,
        step: int | None,
    ) -> int:
        """Estimate number of time windows from sink timestamp bounds.

        Args:
            sink (StructuredSink): Structured sink providing timestamp bounds.
            time_span_ms (int): Width of each time window in milliseconds.
            step (int | None): Step between successive windows, or `None` to use
                `time_span_ms`.

        Examples:
            >>> class _Sink:
            ...     def timestamp_bounds(self):
            ...         return 1_000, 3_500
            ...
            >>> sb = TimeSequenceBuilder(
            ...     sink=_Sink(),
            ...     infer_template=lambda s: (s, ()),
            ...     label_for_group=lambda _: 0,
            ...     time_span_ms=1_000,
            ...     step=500,
            ... )
            >>> sb.count_windows()
            4

        Returns:
            int: Estimated count of time windows for the sink.
        """
        if time_span_ms <= 0:
            return 0
        step = step or time_span_ms
        if step <= 0:
            return 0
        first_ts, last_ts = sink.timestamp_bounds()
        if first_ts is None or last_ts is None:
            return 0
        if last_ts < first_ts:
            return 0
        span = time_span_ms
        duration = last_ts - first_ts

        if duration < span:
            return 1

        return (duration - span) // step + 1


@dataclass(slots=True, frozen=True, kw_only=True)
class EntitySequenceBuilder(SequenceBuilder):
    """Sequence builder for per-entity grouping.

    Attributes:
        train_on_normal_entities_only (bool): Whether anomalous entities are
            excluded from the training split budget.
        continuous_context (bool): Whether adjacent entity windows should
            carry state across sequence boundaries.
    """

    train_on_normal_entities_only: bool = False
    continuous_context: bool = False

    @classmethod
    def from_dataset(
        cls,
        td: TemplatedDataset,
    ) -> Self:
        """Create an entity-grouped builder from a templated dataset.

        Args:
            td (TemplatedDataset): Templated dataset to bind into the builder.

        Returns:
            Self: Builder bound to the templated dataset.
        """
        return cls(
            sink=td.sink,
            infer_template=td.template_parser.inference,
            label_for_group=td.anomaly_labels.label_for_group,
            template_parser=td.template_parser,
        )

    def with_train_on_normal_entities_only(
        self,
        *,
        enabled: bool = True,
    ) -> Self:
        """Limit training sequences to entities without anomalies.

        Args:
            enabled (bool): Whether to restrict train sequences to normal
                entities only.

        Returns:
            Self: Copy with updated normal-only training behavior.
        """
        return replace(self, train_on_normal_entities_only=enabled)

    def with_continuous_context(
        self,
        *,
        enabled: bool = True,
    ) -> Self:
        """Treat consecutive entity windows as one continuous stream.

        Args:
            enabled (bool): Whether to carry model state across entity
                boundaries.

        Returns:
            Self: Copy with updated continuity behaviour.
        """
        return replace(self, continuous_context=enabled)

    def _entity_split_counts(
        self,
        *,
        label_for_group: Callable[[str], int | None],
    ) -> tuple[SequenceSplitCounts, int]:
        """Return cached-free split counts for the fixed entity chronology.

        Args:
            label_for_group (Callable[[str], int | None]): Entity label lookup
                used to determine whether a group is anomalous.

        Returns:
            tuple[SequenceSplitCounts, int]: Exact split counts and eligible
                normal count within the train pool.
        """
        if isinstance(self.sink, _SupportsEntityCount):
            total_entities = self.sink.load_entity_count()
            if total_entities is not None:
                counts = self._split_counts(total_entities)
                train_pool_count = total_entities - counts.test_count
                if not self.train_on_normal_entities_only:
                    return counts, train_pool_count

        chronology_index = (
            self.sink.load_entity_chronology_index()
            if isinstance(self.sink, _SupportsEntityChronologyIndex)
            else {}
        )
        if chronology_index:
            return self._entity_split_counts_from_chronology(
                chronology_index=chronology_index,
                label_for_group=label_for_group,
            )

        return self._entity_split_counts_by_scan(label_for_group=label_for_group)

    def _entity_split_counts_from_chronology(
        self,
        *,
        chronology_index: dict[str, EntityChronologyKey],
        label_for_group: Callable[[str], int | None],
    ) -> tuple[SequenceSplitCounts, int]:
        """Return entity split counts from a materialised chronology sidecar.

        Args:
            chronology_index (dict[str, EntityChronologyKey]): Materialised
                chronology metadata keyed by entity id.
            label_for_group (Callable[[str], int | None]): Entity label lookup
                used to determine whether a group is anomalous.

        Returns:
            tuple[SequenceSplitCounts, int]: Exact split counts and eligible
                normal count within the train pool.
        """
        total_entities = len(chronology_index)
        counts = self._split_counts(total_entities)
        train_pool_count = total_entities - counts.test_count
        if not self.train_on_normal_entities_only:
            return counts, train_pool_count

        if isinstance(self.sink, _SupportsInlineLabelCache):
            _, group_labels = self.sink.load_inline_label_cache()
            if not group_labels:
                return counts, train_pool_count
            normal_pool_count = sum(
                1
                for chronology in sorted(chronology_index.values())[:train_pool_count]
                if not is_anomalous_label(group_labels.get(chronology.entity_id))
            )
            return counts, normal_pool_count

        return self._entity_split_counts_by_scan(label_for_group=label_for_group)

    def _entity_split_counts_by_scan(
        self,
        *,
        label_for_group: Callable[[str], int | None],
    ) -> tuple[SequenceSplitCounts, int]:
        """Return entity split counts by scanning the grouped structured data.

        Args:
            label_for_group (Callable[[str], int | None]): Entity label lookup
                used to determine whether a group is anomalous.

        Returns:
            tuple[SequenceSplitCounts, int]: Exact split counts and eligible
                normal count within the train pool.
        """
        entity_counts = self.sink.count_entities_by_label(label_for_group)
        total_entities = entity_counts.total_entities
        counts = self._split_counts(total_entities)
        train_pool_count = total_entities - counts.test_count
        normal_pool_count = train_pool_count
        if self.train_on_normal_entities_only:
            normal_pool_count = 0
            for index, rows in enumerate(self.iter_grouped_rows()):
                if index >= train_pool_count:
                    break
                entity_id = next(
                    (row.entity_id for row in rows if row.entity_id is not None),
                    None,
                )
                if entity_id is None:
                    continue
                if not is_anomalous_label(label_for_group(entity_id)):
                    normal_pool_count += 1
        return counts, normal_pool_count

    @override
    def train_fraction_eligible_sequence_count(
        self,
        *,
        sequence_summary: SequenceSummary,
    ) -> int:
        """Return the denominator for effective train-fraction accounting.

        Args:
            sequence_summary (SequenceSummary): Aggregate split and label counts.

        Returns:
            int: Eligible entity-sequence count under the current policy. When
                normal-only training is enabled, this counts only normal
                entities in the fixed train pool before the hold-out suffix.
        """
        if not self.train_on_normal_entities_only:
            return (
                sequence_summary.sequence_count - sequence_summary.test_sequence_count
            )
        normal_train_count = sum(
            count
            for label, count in sequence_summary.train_label_counts.items()
            if not is_anomalous_label(label)
        )
        normal_train_count += sum(
            count
            for label, count in sequence_summary.ignored_label_counts.items()
            if not is_anomalous_label(label)
        )
        if normal_train_count:
            return normal_train_count
        return 0

    @override
    def split_count_hint(self) -> SequenceSplitCounts:
        """Return the exact split-count summary for entity grouping.

        Returns:
            SequenceSplitCounts: Exact split counts for the entity builder.
        """
        if (
            self.split_mode is not None
            and self.split_application_order == SplitApplicationOrder.BEFORE_GROUPING
        ):
            return self._before_grouping_split_counts()
        counts, _ = self._entity_split_counts(label_for_group=self.label_for_group)
        return counts

    def _before_grouping_split_counts(self) -> SequenceSplitCounts:
        """Count emitted sequences for a raw-entry split applied pre-grouping.

        The raw-entry split can fragment one entity into multiple emitted
        sequences when ``split_partial_sequences`` is enabled, so the cheap
        entity-count hint under-reports the actual number of sequences the
        training loop will consume. This helper counts the realised emitted
        sequences directly while still avoiding template inference.

        Returns:
            SequenceSplitCounts: Exact emitted sequence counts for the current
                raw-entry split policy.
        """
        row_labels, _ = self._build_row_split_labels()
        if not row_labels:
            return SequenceSplitCounts(
                total_count=0,
                train_count=0,
                ignored_count=0,
                test_count=0,
            )

        row_label_values = {
            line_order: label.value for line_order, label in row_labels.items()
        }
        label_for_group = functools.lru_cache(maxsize=100_000)(self.label_for_group)
        total_count = 0
        train_count = 0
        ignored_count = 0
        test_count = 0
        for rows in self.iter_grouped_rows():
            if not rows:
                continue
            (
                group_total_count,
                group_train_count,
                group_ignored_count,
                group_test_count,
            ) = self._before_grouping_group_counts(
                rows=rows,
                row_label_values=row_label_values,
                label_for_group=label_for_group,
            )
            total_count += group_total_count
            train_count += group_train_count
            ignored_count += group_ignored_count
            test_count += group_test_count

        return SequenceSplitCounts(
            total_count=total_count,
            train_count=train_count,
            ignored_count=ignored_count,
            test_count=test_count,
        )

    def _before_grouping_group_counts(
        self,
        *,
        rows: Collection[StructuredLine],
        row_label_values: dict[int, str],
        label_for_group: Callable[[str], int | None],
    ) -> tuple[int, int, int, int]:
        """Count emitted sequences for one grouped window under raw splitting.

        Args:
            rows (Collection[StructuredLine]): Grouped structured rows.
            row_label_values (dict[int, str]): Raw-entry split labels keyed by
                ``line_order``.
            label_for_group (Callable[[str], int | None]): Group-level anomaly
                lookup used for normal-only training.

        Returns:
            tuple[int, int, int, int]: Total, train, ignored, and test emitted
                sequence counts for the group.
        """
        if (
            self.straddling_group_policy
            == StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES
        ):
            return self._before_grouping_partial_group_counts(
                rows=rows,
                row_label_values=row_label_values,
                label_for_group=label_for_group,
            )
        return self._before_grouping_resolved_group_counts(
            rows=rows,
            row_label_values=row_label_values,
            label_for_group=label_for_group,
        )

    def _before_grouping_partial_group_counts(
        self,
        *,
        rows: Collection[StructuredLine],
        row_label_values: dict[int, str],
        label_for_group: Callable[[str], int | None],
    ) -> tuple[int, int, int, int]:
        """Count emitted sequences when straddlers are split into segments.

        Returns:
            tuple[int, int, int, int]: Total, train, ignored, and test emitted
                sequence counts for the group.
        """
        total_count = 0
        train_count = 0
        ignored_count = 0
        test_count = 0
        for segment_label, segment_rows in split_rows_by_label(
            rows,
            row_label_values,
        ):
            effective_label = SplitLabel(segment_label)
            if (
                self.train_on_normal_entities_only
                and effective_label is SplitLabel.TRAIN
                and any(
                    row.entity_id is not None
                    and is_anomalous_label(label_for_group(row.entity_id))
                    for row in segment_rows
                )
            ):
                effective_label = SplitLabel.IGNORED
            total_count += 1
            if effective_label is SplitLabel.TRAIN:
                train_count += 1
            elif effective_label is SplitLabel.TEST:
                test_count += 1
            else:
                ignored_count += 1
        return total_count, train_count, ignored_count, test_count

    def _before_grouping_resolved_group_counts(
        self,
        *,
        rows: Collection[StructuredLine],
        row_label_values: dict[int, str],
        label_for_group: Callable[[str], int | None],
    ) -> tuple[int, int, int, int]:
        """Count emitted sequences when straddlers resolve to one label.

        Returns:
            tuple[int, int, int, int]: Total, train, ignored, and test emitted
                sequence counts for the group.
        """
        segments = list(split_rows_by_label(rows, row_label_values))
        if not segments:
            return 0, 0, 0, 0
        split_label_value = resolve_straddling_group_label(
            self.straddling_group_policy.value,
            segments,
        )
        if split_label_value is None:
            return 0, 0, 0, 0
        effective_label = SplitLabel(split_label_value)
        if (
            self.train_on_normal_entities_only
            and effective_label is SplitLabel.TRAIN
            and any(
                row.entity_id is not None
                and is_anomalous_label(label_for_group(row.entity_id))
                for row in rows
            )
        ):
            effective_label = SplitLabel.IGNORED
        if effective_label is SplitLabel.TRAIN:
            return 1, 1, 0, 0
        if effective_label is SplitLabel.TEST:
            return 1, 0, 0, 1
        return 1, 0, 1, 0

    @override
    def train_sequence_count_unit_hint(self) -> str:
        """Return the unit label for entity-grouped train progress.

        Returns:
            str: Unit label for entity-grouped train progress.
        """
        return "entities"

    @override
    def split_summary_train_on_normal_entities_only(self) -> bool:
        """Return entity split-summary metadata for normal-only training.

        Returns:
            bool: Whether train was restricted to normal entities only.
        """
        return self.train_on_normal_entities_only

    def iter_training_sequences(self) -> Iterator[TemplateSequence]:
        """Yield only the train split used for detector fitting.

        For HDFS-style raw-entry prefix protocols, fitting only needs the
        train population implied by the selected policy. We therefore avoid
        replaying the full suffix and instead materialise just the selected
        train entities or raw-prefix rows, depending on the straddler policy.

        Raises:
            ValueError: If the requested raw-entry split metadata is missing
                for a configured before-grouping prefix protocol.
        """
        if (
            self.split_mode is None
            or self.split_application_order != SplitApplicationOrder.BEFORE_GROUPING
        ):
            yield from SequenceBuilder.iter_training_sequences(self)
            return

        if self.split_mode not in {
            RawEntrySplitMode.PREFIX_COUNT,
            RawEntrySplitMode.PREFIX_FRACTION,
            RawEntrySplitMode.PREFIX_NORMAL_FRACTION,
        }:
            yield from SequenceBuilder.iter_training_sequences(self)
            return

        infer_template = functools.lru_cache(maxsize=50_000)(self.infer_template)
        label_for_group = functools.lru_cache(maxsize=100_000)(self.label_for_group)

        if self.split_mode is RawEntrySplitMode.PREFIX_COUNT:
            cutoff_entry_index = self.train_entry_count
            if cutoff_entry_index is None:
                msg = "train_entry_count must be set for PREFIX_COUNT splits."
                raise ValueError(msg)
        else:
            row_count = self.sink.count_rows()
            if self.split_mode is RawEntrySplitMode.PREFIX_FRACTION:
                train_entry_fraction = self.train_entry_fraction
                if train_entry_fraction is None:
                    msg = "train_entry_fraction must be set for PREFIX_FRACTION splits."
                    raise ValueError(msg)
                cutoff_entry_index = math.ceil(train_entry_fraction * row_count)
            else:
                yield from SequenceBuilder.iter_training_sequences(self)
                return

        if (
            self.straddling_group_policy
            is StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES
        ):
            if self._can_use_before_grouping_raw_source_test_path():
                yield from self._iter_training_sequences_from_raw_source_prefix(
                    cutoff_entry_index=cutoff_entry_index,
                    infer_template=infer_template,
                    label_for_group=label_for_group,
                )
                return
            yield from self._iter_training_sequences_from_raw_prefix(
                cutoff_entry_index=cutoff_entry_index,
                infer_template=infer_template,
                label_for_group=label_for_group,
            )
            return

        if self.straddling_group_policy is StraddlingGroupPolicy.ASSIGN_BY_FIRST_EVENT:
            yield from self._iter_training_sequences_from_prefix_entities(
                cutoff_entry_index=cutoff_entry_index,
                infer_template=infer_template,
                label_for_group=label_for_group,
            )
            return

        yield from SequenceBuilder.iter_training_sequences(self)

    @override
    def iter_test_sequences(self) -> Iterator[TemplateSequence]:
        """Yield only the test suffix used for detector scoring.

        Yields:
            TemplateSequence: Sequences assigned to the test split.
        """
        if self.train_on_normal_entities_only:
            yield from SequenceBuilder.iter_test_sequences(self)
            return

        infer_template = functools.lru_cache(maxsize=50_000)(self.infer_template)
        label_for_group = functools.lru_cache(maxsize=100_000)(self.label_for_group)

        if self._can_use_before_grouping_raw_prefix_test_path():
            yield from self._iter_before_grouping_test_sequences(
                infer_template=infer_template,
                label_for_group=label_for_group,
            )
            return

        if (
            self.split_mode is not None
            and self.split_application_order == SplitApplicationOrder.BEFORE_GROUPING
        ):
            yield from SequenceBuilder.iter_test_sequences(self)
            return

        counts, _ = self._entity_split_counts(label_for_group=label_for_group)
        test_start_index = counts.total_count - counts.test_count

        for window_id, rows in enumerate(self.iter_grouped_rows()):
            entity_id = next(
                (row.entity_id for row in rows if row.entity_id is not None),
                None,
            )
            prefixed_split_label = _split_label_from_prefixed_entity_id(entity_id)
            if prefixed_split_label is not None and self.split_mode is None:
                split_label = prefixed_split_label
            elif self.split_application_order == SplitApplicationOrder.BEFORE_GROUPING:
                continue
            elif window_id >= test_start_index:
                split_label = SplitLabel.TEST
            else:
                continue
            seq = self._build_sequence(
                window_id,
                rows,
                infer_template,
                label_for_group,
                split_label,
                continuous_context=self.continuous_context,
                allow_group_label_fallback=False,
            )
            if seq is not None:
                yield seq

    def _can_use_before_grouping_raw_prefix_test_path(self) -> bool:
        """Return whether the test suffix can be replayed without full replay.

        Returns:
            bool: `True` when the before-grouping raw split can be served by a
                dedicated suffix iterator.
        """
        return (
            self.split_mode is not None
            and self.split_application_order == SplitApplicationOrder.BEFORE_GROUPING
            and self.split_mode
            in {
                RawEntrySplitMode.PREFIX_COUNT,
                RawEntrySplitMode.PREFIX_FRACTION,
            }
        )

    def _before_grouping_cutoff_entry_index(self) -> int:
        """Return the raw-entry cutoff used by before-grouping prefix splits.

        Returns:
            int: Zero-based raw-entry index where the test suffix begins.

        Raises:
            ValueError: If the configured raw-entry split metadata is missing.
        """
        if self.split_mode is RawEntrySplitMode.PREFIX_COUNT:
            cutoff_entry_index = self.train_entry_count
            if cutoff_entry_index is None:
                msg = "train_entry_count must be set for PREFIX_COUNT splits."
                raise ValueError(msg)
            return cutoff_entry_index

        row_count = self.sink.count_rows()
        train_entry_fraction = self.train_entry_fraction
        if train_entry_fraction is None:
            msg = "train_entry_fraction must be set for PREFIX_FRACTION splits."
            raise ValueError(msg)
        return math.ceil(train_entry_fraction * row_count)

    @override
    def _build_before_grouping_raw_entry_split_summary(
        self,
    ) -> RawEntrySplitSummary | None:
        """Return a before-grouping raw-entry summary without full replay.

        Returns:
            RawEntrySplitSummary | None: Cached or streamed summary for the
            current before-grouping raw-entry split, or `None` when the split
            is not active.
        """
        if self.split_mode is None:
            return None
        if self.split_mode in {
            RawEntrySplitMode.PREFIX_COUNT,
            RawEntrySplitMode.PREFIX_FRACTION,
        }:
            return self._build_before_grouping_raw_entry_split_summary_from_stream()
        return self._build_before_grouping_raw_entry_split_summary_from_plan()

    def _build_before_grouping_raw_entry_split_summary_from_stream(
        self,
    ) -> RawEntrySplitSummary:
        """Stream raw-entry split diagnostics for count/fraction prefixes.

        Returns:
            RawEntrySplitSummary: Streaming summary for prefix count or
            prefix fraction raw-entry splits.

        Raises:
            RuntimeError: If the method is reached without an active raw-entry
                split.
        """
        split_mode = self.split_mode
        if split_mode is None:
            msg = "split_mode must be set for before-grouping raw summaries."
            raise RuntimeError(msg)
        total_rows = self.sink.count_rows()
        if total_rows <= 0:
            return RawEntrySplitSummary(
                split_mode=split_mode.value,
                application_order=self.split_application_order.value,
                cutoff_entry_index=0,
                train_raw_entry_count=0,
                train_normal_entry_count=0,
                train_anomalous_entry_count=0,
                test_raw_entry_count=0,
                test_normal_entry_count=0,
                test_anomalous_entry_count=0,
                straddling_group_count=0,
                straddling_group_policy=self.straddling_group_policy.value,
            )

        cutoff_entry_index = min(self._before_grouping_cutoff_entry_index(), total_rows)
        counts = _count_before_grouping_raw_entry_summary(
            self.iter_grouped_rows(),
            cutoff_entry_index=cutoff_entry_index,
        )

        return RawEntrySplitSummary(
            split_mode=split_mode.value,
            application_order=self.split_application_order.value,
            cutoff_entry_index=cutoff_entry_index,
            train_raw_entry_count=cutoff_entry_index,
            train_normal_entry_count=counts.train_normal_entry_count,
            train_anomalous_entry_count=counts.train_anomalous_entry_count,
            test_raw_entry_count=total_rows - cutoff_entry_index,
            test_normal_entry_count=counts.test_normal_entry_count,
            test_anomalous_entry_count=counts.test_anomalous_entry_count,
            straddling_group_count=counts.straddling_group_count,
            straddling_group_policy=self.straddling_group_policy.value,
        )

    def _build_before_grouping_raw_entry_split_summary_from_plan(
        self,
    ) -> RawEntrySplitSummary | None:
        """Fallback to the legacy planned-row summary for other policies.

        Returns:
            RawEntrySplitSummary | None: Legacy summary for policies that
            still rely on raw-row label planning, or `None` when the split is
            inactive.
        """
        if self.split_mode is None:
            return None
        row_labels, summary = self._build_row_split_labels()
        if summary is None:
            return None
        row_label_values = {
            line_order: label.value for line_order, label in row_labels.items()
        }
        return replace(
            summary,
            straddling_group_count=count_straddling_groups(
                self.iter_grouped_rows(),
                row_label_values,
            ),
            straddling_group_policy=self.straddling_group_policy.value,
        )

    def _iter_before_grouping_test_sequences(
        self,
        *,
        infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]],
        label_for_group: Callable[[str], int | None],
    ) -> Iterator[TemplateSequence]:
        """Yield test sequences for before-grouping raw-entry prefix splits.

        Args:
            infer_template (Callable[[str], tuple[LogTemplate, ExtractedParameters]]):
                Template inference function for the emitted suffix rows.
            label_for_group (Callable[[str], int | None]): Entity-level anomaly
                label lookup.

        Yields:
            TemplateSequence: Test-split sequences emitted from the suffix.
        """
        cutoff_entry_index = self._before_grouping_cutoff_entry_index()
        if self._can_use_before_grouping_raw_source_test_path():
            cached_offset = self._before_grouping_test_start_byte_offset()
            if cached_offset is None:
                _LOGGER.info(
                    (
                        "Resuming before-grouping test replay for %s from raw "
                        "entry %s; the first test sequence may take a while "
                        "while the cutoff is scanned"
                    ),
                    self.sink.dataset_name,
                    cutoff_entry_index,
                )
            else:
                _LOGGER.info(
                    (
                        "Resuming before-grouping test replay for %s from "
                        "cached raw byte offset %s (raw entry %s)"
                    ),
                    self.sink.dataset_name,
                    cached_offset,
                    cutoff_entry_index,
                )
            if self._uses_identity_template_parser() and (
                self._uses_dense_raw_session_parser()
            ):
                yield from (
                    self._iter_before_grouping_test_sequences_from_dense_raw_source(
                        cutoff_entry_index=cutoff_entry_index,
                    )
                )
                return
            yield from self._iter_before_grouping_test_sequences_from_raw_source(
                cutoff_entry_index=cutoff_entry_index,
                infer_template=infer_template,
                label_for_group=label_for_group,
            )
            return
        if isinstance(self.sink, _SupportsEntitySequenceSuffixScan) and (
            self.straddling_group_policy
            in {
                StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES,
                StraddlingGroupPolicy.ASSIGN_BY_FIRST_EVENT,
            }
        ):
            chronology_index = (
                self.sink.load_entity_chronology_index()
                if isinstance(self.sink, _SupportsEntityChronologyIndex)
                else {}
            )
            if (
                self.straddling_group_policy
                is StraddlingGroupPolicy.ASSIGN_BY_FIRST_EVENT
                and not chronology_index
            ):
                yield from self._iter_before_grouping_test_sequences_from_first_event(
                    cutoff_entry_index=cutoff_entry_index,
                    infer_template=infer_template,
                    label_for_group=label_for_group,
                )
                return
            yield from self._iter_before_grouping_test_sequences_from_suffix_groups(
                cutoff_entry_index=cutoff_entry_index,
                infer_template=infer_template,
                label_for_group=label_for_group,
                suffix_group_iter_factory=self.sink.iter_entity_sequences_from_line_order,
                chronology_index=chronology_index,
            )
            return
        if (
            self.straddling_group_policy
            is StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES
        ):
            yield from self._iter_before_grouping_test_sequences_from_partial_split(
                cutoff_entry_index=cutoff_entry_index,
                infer_template=infer_template,
                label_for_group=label_for_group,
            )
            return
        if self.straddling_group_policy is StraddlingGroupPolicy.ASSIGN_BY_FIRST_EVENT:
            yield from self._iter_before_grouping_test_sequences_from_first_event(
                cutoff_entry_index=cutoff_entry_index,
                infer_template=infer_template,
                label_for_group=label_for_group,
            )
            return
        yield from SequenceBuilder.iter_test_sequences(self)

    def _can_use_before_grouping_raw_source_test_path(self) -> bool:
        """Return whether the raw source can resume from the test boundary."""
        return (
            self.split_mode is not None
            and self.split_application_order == SplitApplicationOrder.BEFORE_GROUPING
            and self.split_mode
            in {
                RawEntrySplitMode.PREFIX_COUNT,
                RawEntrySplitMode.PREFIX_FRACTION,
            }
            and self.straddling_group_policy
            in {
                StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES,
                StraddlingGroupPolicy.ASSIGN_BY_FIRST_EVENT,
            }
            and self.template_parser is not None
            and self.sink.raw_dataset_path.exists()
        )

    def _cache_before_grouping_test_start_byte_offset(self, offset: int) -> None:
        """Remember where the test suffix starts in the raw source."""
        if self.raw_replay_state.test_start_byte_offset is None:
            self.raw_replay_state.test_start_byte_offset = offset

    def _before_grouping_test_start_byte_offset(self) -> int | None:
        """Return the cached raw-source offset for the test suffix, if known."""
        return self.raw_replay_state.test_start_byte_offset

    def _uses_identity_template_parser(self) -> bool:
        """Return whether the builder is backed by the identity parser."""
        return bool(
            self.template_parser is not None
            and self.template_parser.is_identity_parser,
        )

    def _uses_dense_raw_session_parser(self) -> bool:
        """Return whether the raw compat source can be parsed directly."""
        return self.sink.parser.__class__.__name__ == "DelimitedLabelledEventParser"

    def _iter_training_sequences_from_raw_source_prefix(  # noqa: C901
        self,
        *,
        cutoff_entry_index: int,
        infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]],
        label_for_group: Callable[[str], int | None],
    ) -> Iterator[TemplateSequence]:
        """Yield train sequences by scanning the raw source up to the cutoff.

        The raw-source scan populates the cached test boundary offset so the
        paired test replay can resume at the exact cutoff without rescanning
        the train prefix. The cached state stays O(1) because it stores one
        integer byte offset.
        """
        _LOGGER.info(
            "Scanning raw source for train replay of %s up to raw entry %s",
            self.sink.dataset_name,
            cutoff_entry_index,
        )
        raw_dataset_path = self.sink.raw_dataset_path
        parser = self.sink.parser
        current_entity_id: str | None = None
        current_rows: list[StructuredLine] = []
        current_entity_emittable = True
        window_id = 0
        raw_line_order = 0

        with raw_dataset_path.open(
            encoding="utf-8",
            errors="replace",
        ) as handle:
            while True:
                line_start = handle.tell()
                if raw_line_order >= cutoff_entry_index:
                    self._cache_before_grouping_test_start_byte_offset(line_start)
                    break

                raw_line = handle.readline()
                if not raw_line:
                    self._cache_before_grouping_test_start_byte_offset(line_start)
                    break

                parsed = parser.parse_line(raw_line.rstrip("\n").rstrip("\r"))
                if parsed is None:
                    raw_line_order += 1
                    continue

                row = StructuredLine.with_line_order(
                    line_order=raw_line_order,
                    base=parsed,
                )
                raw_line_order += 1
                entity_id = row.entity_id
                if entity_id is None:
                    continue

                if entity_id != current_entity_id:
                    if current_rows and current_entity_emittable:
                        seq = self._build_sequence(
                            window_id,
                            current_rows,
                            infer_template,
                            label_for_group,
                            SplitLabel.TRAIN,
                            continuous_context=self.continuous_context,
                            allow_group_label_fallback=False,
                        )
                        if seq is not None:
                            yield seq
                            window_id += 1
                    current_entity_id = entity_id
                    current_rows = []
                    current_entity_emittable = not (
                        self.train_on_normal_entities_only
                        and is_anomalous_label(label_for_group(entity_id))
                    )

                if current_entity_emittable:
                    current_rows.append(row)

        if current_rows and current_entity_emittable:
            seq = self._build_sequence(
                window_id,
                current_rows,
                infer_template,
                label_for_group,
                SplitLabel.TRAIN,
                continuous_context=self.continuous_context,
                allow_group_label_fallback=False,
            )
            if seq is not None:
                yield seq

    def _iter_before_grouping_test_sequences_from_raw_source(  # noqa: C901
        self,
        *,
        cutoff_entry_index: int,
        infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]],
        label_for_group: Callable[[str], int | None],
    ) -> Iterator[TemplateSequence]:
        """Yield test sequences by resuming the raw source at the cutoff.

        The compat session streams are already ordered by the raw-entry split
        key, so resuming from the raw file avoids rescanning the parquet cache
        while keeping the replay memory bounded to the current entity segment.
        """
        _LOGGER.info(
            "Scanning raw source for test replay of %s from raw entry %s",
            self.sink.dataset_name,
            cutoff_entry_index,
        )
        chronology_index = (
            self.sink.load_entity_chronology_index()
            if isinstance(self.sink, _SupportsEntityChronologyIndex)
            else {}
        )
        raw_dataset_path = self.sink.raw_dataset_path
        parser = self.sink.parser
        current_entity_id: str | None = None
        current_rows: list[StructuredLine] = []
        current_entity_emittable = True
        window_id = 0
        cached_offset = self._before_grouping_test_start_byte_offset()

        with raw_dataset_path.open(
            encoding="utf-8",
            errors="replace",
        ) as handle:
            if cached_offset is not None:
                handle.seek(cached_offset)
                raw_line_iter: Iterator[str] = handle
            else:
                raw_line_iter = islice(handle, cutoff_entry_index, None)
            raw_line_order = cutoff_entry_index

            for raw_line in raw_line_iter:
                parsed = parser.parse_line(raw_line.rstrip("\n").rstrip("\r"))
                if parsed is None:
                    raw_line_order += 1
                    continue

                row = StructuredLine.with_line_order(
                    line_order=raw_line_order,
                    base=parsed,
                )
                raw_line_order += 1
                entity_id = row.entity_id
                if entity_id is None:
                    continue

                if entity_id != current_entity_id:
                    if current_rows and current_entity_emittable:
                        seq = self._build_sequence(
                            window_id,
                            current_rows,
                            infer_template,
                            label_for_group,
                            SplitLabel.TEST,
                            continuous_context=self.continuous_context,
                            allow_group_label_fallback=False,
                        )
                        if seq is not None:
                            yield seq
                            window_id += 1
                    current_entity_id = entity_id
                    current_rows = []
                    current_entity_emittable = True
                    if (
                        self.straddling_group_policy
                        is StraddlingGroupPolicy.ASSIGN_BY_FIRST_EVENT
                    ):
                        chronology = chronology_index.get(entity_id)
                        current_entity_emittable = (
                            chronology is None
                            or chronology.first_line_order >= cutoff_entry_index
                        )

                if current_entity_emittable:
                    current_rows.append(row)

        if current_rows and current_entity_emittable:
            seq = self._build_sequence(
                window_id,
                current_rows,
                infer_template,
                label_for_group,
                SplitLabel.TEST,
                continuous_context=self.continuous_context,
                allow_group_label_fallback=False,
            )
            if seq is not None:
                yield seq

    def _iter_before_grouping_test_sequences_from_dense_raw_source(  # noqa: C901, PLR0912, PLR0914
        self,
        *,
        cutoff_entry_index: int,
    ) -> Iterator[TemplateSequence]:
        """Yield test sequences by slicing a dense raw session stream.

        The HDFS compat stream is a dense tab-separated session/event file and
        the identity template parser preserves the raw event text. In that
        configuration we can skip the generic structured-row wrapper entirely
        and build template sequences directly from the raw suffix.
        """
        _LOGGER.info(
            "Scanning dense raw source for test replay of %s from raw entry %s",
            self.sink.dataset_name,
            cutoff_entry_index,
        )
        raw_dataset_path = self.sink.raw_dataset_path
        chronology_index = (
            self.sink.load_entity_chronology_index()
            if isinstance(self.sink, _SupportsEntityChronologyIndex)
            else {}
        )
        current_entity_id: str | None = None
        current_events: list[tuple[str, list[str], int | None]] = []
        current_event_labels: list[int | None] | None = None
        current_entity_emittable = True
        current_sequence_is_anomalous = False
        window_id = 0
        cached_offset = self._before_grouping_test_start_byte_offset()

        with raw_dataset_path.open(
            encoding="utf-8",
            errors="replace",
        ) as handle:
            if cached_offset is not None:
                handle.seek(cached_offset)
                raw_line_iter: Iterator[str] = handle
            else:
                raw_line_iter = islice(handle, cutoff_entry_index, None)

            for raw_line in raw_line_iter:
                s = raw_line.rstrip("\n").rstrip("\r")
                if not s:
                    continue
                parts = s.split("\t")
                if len(parts) != _DENSE_RAW_SESSION_FIELD_COUNT:
                    continue
                entity_id, label_s, event_id = parts
                if entity_id != current_entity_id:
                    if current_events and current_entity_emittable:
                        yield TemplateSequence(
                            events=current_events,
                            label=1 if current_sequence_is_anomalous else 0,
                            entity_ids=[current_entity_id] if current_entity_id else [],
                            window_id=window_id,
                            split_label=SplitLabel.TEST,
                            event_labels=(
                                tuple(current_event_labels)
                                if current_event_labels is not None
                                else None
                            ),
                            continuous_context=self.continuous_context,
                        )
                        window_id += 1
                    current_entity_id = entity_id
                    current_events = []
                    current_event_labels = None
                    current_entity_emittable = True
                    current_sequence_is_anomalous = False
                    if (
                        self.straddling_group_policy
                        is StraddlingGroupPolicy.ASSIGN_BY_FIRST_EVENT
                    ):
                        chronology = chronology_index.get(entity_id)
                        current_entity_emittable = (
                            chronology is None
                            or chronology.first_line_order >= cutoff_entry_index
                        )
                if not current_entity_emittable:
                    continue
                current_events.append((event_id, [], None))
                label = int(label_s)
                if not current_sequence_is_anomalous and is_anomalous_label(label):
                    current_sequence_is_anomalous = True
                if label is not None:
                    if current_event_labels is None:
                        current_event_labels = [None] * (len(current_events) - 1)
                    current_event_labels.append(label)
                elif current_event_labels is not None:
                    current_event_labels.append(None)

        if current_events and current_entity_emittable:
            yield TemplateSequence(
                events=current_events,
                label=1 if current_sequence_is_anomalous else 0,
                entity_ids=[current_entity_id] if current_entity_id else [],
                window_id=window_id,
                split_label=SplitLabel.TEST,
                event_labels=(
                    tuple(current_event_labels)
                    if current_event_labels is not None
                    else None
                ),
                continuous_context=self.continuous_context,
            )

    def _iter_before_grouping_test_sequences_from_suffix_groups(
        self,
        *,
        cutoff_entry_index: int,
        infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]],
        label_for_group: Callable[[str], int | None],
        suffix_group_iter_factory: Callable[
            [int],
            Callable[[], Iterator[Collection[StructuredLine]]],
        ],
        chronology_index: dict[str, EntityChronologyKey],
    ) -> Iterator[TemplateSequence]:
        """Yield test sequences from a suffix-only grouped row scan.

        Args:
            cutoff_entry_index (int): Raw-entry boundary where the test suffix
                begins.
            infer_template (Callable[[str], tuple[LogTemplate, ExtractedParameters]]):
                Template inference function for the retained suffix rows.
            label_for_group (Callable[[str], int | None]): Entity-level anomaly
                label lookup.
            suffix_group_iter_factory: Sink-level suffix iterator factory for
                the suffix replay path.
            chronology_index (dict[str, EntityChronologyKey]): Entity chronology
                metadata used to exclude entities that started before the
                cutoff when assign-by-first-event is active.

        Yields:
            TemplateSequence: Test-side grouped suffix sequences.
        """
        for window_id, rows in enumerate(
            suffix_group_iter_factory(cutoff_entry_index)(),
        ):
            if not rows:
                continue
            entity_id = next(
                (row.entity_id for row in rows if row.entity_id is not None),
                None,
            )
            if entity_id is None:
                continue
            if (
                self.straddling_group_policy
                is StraddlingGroupPolicy.ASSIGN_BY_FIRST_EVENT
            ):
                chronology = chronology_index.get(entity_id)
                if (
                    chronology is not None
                    and chronology.first_line_order < cutoff_entry_index
                ):
                    continue
            seq = self._build_sequence(
                window_id,
                rows,
                infer_template,
                label_for_group,
                SplitLabel.TEST,
                continuous_context=self.continuous_context,
                allow_group_label_fallback=False,
            )
            if seq is not None:
                yield seq

    def _iter_before_grouping_test_sequences_from_partial_split(
        self,
        *,
        cutoff_entry_index: int,
        infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]],
        label_for_group: Callable[[str], int | None],
    ) -> Iterator[TemplateSequence]:
        """Yield test-side partial sequences from a raw-entry prefix split.

        Args:
            cutoff_entry_index (int): Raw-entry boundary where the test suffix
                begins.
            infer_template (Callable[[str], tuple[LogTemplate, ExtractedParameters]]):
                Template inference function for the retained suffix rows.
            label_for_group (Callable[[str], int | None]): Entity-level anomaly
                label lookup.

        Yields:
            TemplateSequence: Test-side partial sequences.
        """
        window_id = 0
        for rows in self.iter_grouped_rows():
            rows_before = [
                row
                for row in rows
                if row.line_order is not None and row.line_order < cutoff_entry_index
            ]
            rows_after = [
                row
                for row in rows
                if row.line_order is not None and row.line_order >= cutoff_entry_index
            ]
            if rows_before and rows_after:
                seq = self._build_sequence(
                    window_id + 1,
                    rows_after,
                    infer_template,
                    label_for_group,
                    SplitLabel.TEST,
                )
                if seq is not None:
                    yield seq
                window_id += 2
                continue
            if rows_after:
                seq = self._build_sequence(
                    window_id,
                    rows_after,
                    infer_template,
                    label_for_group,
                    SplitLabel.TEST,
                )
                if seq is not None:
                    yield seq
                window_id += 1
                continue
            if rows_before:
                window_id += 1

    def _iter_before_grouping_test_sequences_from_first_event(
        self,
        *,
        cutoff_entry_index: int,
        infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]],
        label_for_group: Callable[[str], int | None],
    ) -> Iterator[TemplateSequence]:
        """Yield test-side whole entities assigned by their first event.

        Args:
            cutoff_entry_index (int): Raw-entry boundary where the test suffix
                begins.
            infer_template (Callable[[str], tuple[LogTemplate, ExtractedParameters]]):
                Template inference function for the retained suffix rows.
            label_for_group (Callable[[str], int | None]): Entity-level anomaly
                label lookup.

        Yields:
            TemplateSequence: Test-side whole entities.
        """
        window_id = 0
        for rows in self.iter_grouped_rows():
            if not rows:
                continue
            first_line_order = next(
                (row.line_order for row in rows if row.line_order is not None),
                None,
            )
            if first_line_order is not None and first_line_order >= cutoff_entry_index:
                seq = self._build_sequence(
                    window_id,
                    rows,
                    infer_template,
                    label_for_group,
                    SplitLabel.TEST,
                )
                if seq is not None:
                    yield seq
            window_id += 1

    @override
    def iter_grouped_rows(self) -> Iterator[Collection[StructuredLine]]:
        """Return rows grouped by entity.

        Returns:
            Iterator[Collection[StructuredLine]]: Entity-grouped structured rows.
        """
        return self.sink.iter_entity_sequences()()

    @override
    def _entity_ids_for_rows(self, rows: Collection[StructuredLine]) -> list[str]:
        """Return the single entity id for an entity-grouped window.

        Args:
            rows (Collection[StructuredLine]): Structured rows belonging to one
                entity-grouped window.

        Returns:
            list[str]: Single entity id when present, otherwise an empty list.
        """
        for row in rows:
            if row.entity_id is not None:
                return [row.entity_id]
        return []

    def _iter_training_sequences_from_raw_prefix(
        self,
        *,
        cutoff_entry_index: int,
        infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]],
        label_for_group: Callable[[str], int | None],
    ) -> Iterator[TemplateSequence]:
        """Yield train sequences built only from the raw prefix rows.

        Args:
            cutoff_entry_index (int): Raw-entry boundary where the test suffix
                begins.
            infer_template (Callable[[str], tuple[LogTemplate, ExtractedParameters]]):
                Template inference function for the retained prefix rows.
            label_for_group (Callable[[str], int | None]): Entity-level anomaly
                label lookup.

        Yields:
            TemplateSequence: Train-split sequences from the raw prefix only.
        """
        entity_rows: dict[str, list[StructuredLine]] = {}
        entity_order: list[str] = []
        for row in self._iter_source_order_rows():
            if row.line_order is not None and row.line_order >= cutoff_entry_index:
                break
            if row.entity_id is None:
                continue
            rows = entity_rows.setdefault(row.entity_id, [])
            if not rows:
                entity_order.append(row.entity_id)
            rows.append(row)

        window_id = 0
        for entity_id in entity_order:
            rows = entity_rows[entity_id]
            if self.train_on_normal_entities_only and is_anomalous_label(
                label_for_group(entity_id),
            ):
                continue
            seq = self._build_sequence(
                window_id,
                rows,
                infer_template,
                label_for_group,
                SplitLabel.TRAIN,
            )
            if seq is not None and seq.split_label is SplitLabel.TRAIN:
                yield seq
                window_id += 1

    def _iter_training_sequences_from_prefix_entities(
        self,
        *,
        cutoff_entry_index: int,
        infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]],
        label_for_group: Callable[[str], int | None],
    ) -> Iterator[TemplateSequence]:
        """Yield train sequences for assign-first policies without suffix replay.

        Args:
            cutoff_entry_index (int): Raw-entry boundary where the test suffix
                begins.
            infer_template (Callable[[str], tuple[LogTemplate, ExtractedParameters]]):
                Template inference function for the selected train entities.
            label_for_group (Callable[[str], int | None]): Entity-level anomaly
                label lookup.

        Yields:
            TemplateSequence: Train-split sequences for entities that first
                appeared in the train prefix.
        """
        entity_order: list[str] = []
        seen_entities: set[str] = set()
        for row in self._iter_source_order_rows():
            if row.line_order is not None and row.line_order >= cutoff_entry_index:
                break
            if row.entity_id is None or row.entity_id in seen_entities:
                continue
            seen_entities.add(row.entity_id)
            entity_order.append(row.entity_id)

        if not entity_order:
            return

        entity_rows: dict[str, list[StructuredLine]] = {
            entity_id: [] for entity_id in entity_order
        }
        selected_entities = set(entity_order)
        for rows in self.sink.iter_entity_sequences()():
            entity_id = next(
                (row.entity_id for row in rows if row.entity_id is not None),
                None,
            )
            if entity_id is None or entity_id not in selected_entities:
                continue
            entity_rows[entity_id].extend(rows)

        window_id = 0
        for entity_id in entity_order:
            rows = entity_rows[entity_id]
            if not rows:
                continue
            rows.sort(key=lambda row: row.line_order)
            split_label = (
                SplitLabel.IGNORED
                if (
                    self.train_on_normal_entities_only
                    and is_anomalous_label(label_for_group(entity_id))
                )
                else SplitLabel.TRAIN
            )
            seq = self._build_sequence(
                window_id,
                rows,
                infer_template,
                label_for_group,
                split_label,
            )
            if seq is not None and seq.split_label is SplitLabel.TRAIN:
                yield seq
                window_id += 1

    def __iter__(self) -> Iterator[TemplateSequence]:  # noqa: C901, PLR0912
        """Iterate over template sequences yielded by the configured grouping.

        Yields:
            TemplateSequence: One grouped and template-enriched sequence.
        """
        infer_template = functools.lru_cache(maxsize=50_000)(self.infer_template)
        label_for_group = functools.lru_cache(maxsize=100_000)(self.label_for_group)
        row_labels, _ = self._build_row_split_labels()
        counts, _ = self._entity_split_counts(label_for_group=label_for_group)
        normals_seen_in_train = 0
        test_start_index = counts.total_count - counts.test_count

        if self.split_application_order == SplitApplicationOrder.BEFORE_GROUPING:
            window_id = 0
            for rows in self.iter_grouped_rows():
                entity_id = next(
                    (row.entity_id for row in rows if row.entity_id is not None),
                    None,
                )
                entity_is_anomalous = entity_id is not None and is_anomalous_label(
                    label_for_group(entity_id),
                )
                prefixed_split_label = _split_label_from_prefixed_entity_id(entity_id)
                if entity_id is None:
                    split_label = SplitLabel.TRAIN
                elif prefixed_split_label is not None and self.split_mode is None:
                    split_label = prefixed_split_label
                elif self.train_on_normal_entities_only and entity_is_anomalous:
                    split_label = SplitLabel.IGNORED
                elif self.split_mode in {
                    RawEntrySplitMode.PREFIX_NORMAL_FRACTION,
                    RawEntrySplitMode.PREFIX_COUNT,
                }:
                    split_label = SplitLabel.TRAIN
                else:
                    split_label = SplitLabel.TRAIN
                for seq in self._build_sequences_for_group(
                    window_id=window_id,
                    rows=rows,
                    infer_template=infer_template,
                    label_for_group=label_for_group,
                    split_label=split_label,
                    row_labels=row_labels,
                    train_only_normal_entities=self.train_on_normal_entities_only,
                ):
                    yield seq
                    window_id += 1
            return

        for window_id, rows in enumerate(self.iter_grouped_rows()):
            entity_id = next(
                (row.entity_id for row in rows if row.entity_id is not None),
                None,
            )
            entity_is_anomalous = entity_id is not None and is_anomalous_label(
                label_for_group(entity_id),
            )
            prefixed_split_label = _split_label_from_prefixed_entity_id(entity_id)
            if prefixed_split_label is not None and self.split_mode is None:
                split_label = prefixed_split_label
            elif window_id >= test_start_index:
                split_label = SplitLabel.TEST
            elif self.train_on_normal_entities_only:
                split_label = (
                    SplitLabel.TRAIN
                    if (not entity_is_anomalous)
                    and (normals_seen_in_train < counts.train_count)
                    else SplitLabel.IGNORED
                )
            else:
                split_label = (
                    SplitLabel.TRAIN
                    if window_id < counts.train_count
                    else SplitLabel.IGNORED
                )

            seq = self._build_sequence(
                window_id,
                rows,
                infer_template,
                label_for_group,
                split_label,
                group_label_is_anomalous=entity_is_anomalous,
                continuous_context=self.continuous_context,
            )
            if seq is not None:
                if (
                    self.train_on_normal_entities_only
                    and split_label is SplitLabel.TRAIN
                    and seq.label == 0
                ):
                    normals_seen_in_train += 1
                yield seq


def _split_label_from_prefixed_entity_id(entity_id: str | None) -> SplitLabel | None:
    """Return a split label from known preprocessed-session entity prefixes.

    Args:
        entity_id (str | None): Entity id to inspect.

    Returns:
        SplitLabel | None: Derived split label, or `None` when no known
            prefix is present.
    """
    if entity_id is None:
        return None
    prefix = entity_id.split(":", 1)[0].strip().lower()
    if not prefix:
        return None
    # For post-processed pre-split datasets, entity ids are prefixed with the
    # split file alias (for example `hdfs_train:*`, `openstack_test_normal:*`).
    # We infer train/test directly from that alias instead of hardcoding
    # dataset-specific names in the sequence builder.
    if "train" in prefix:
        return SplitLabel.TRAIN
    if "test" in prefix:
        return SplitLabel.TEST
    return None


def _should_promote_from_group_label(
    *,
    group_label_is_anomalous: bool | None,
    label_for_group: Callable[[str], int | None],
    entity_id: str | None,
) -> bool:
    """Return whether a sequence should be promoted by group-level labels.

    Args:
        group_label_is_anomalous (bool | None): Optional precomputed entity
            label verdict.
        label_for_group (Callable[[str], int | None]): Group-level anomaly
            lookup.
        entity_id (str | None): Entity identifier from the current row.

    Returns:
        bool: `True` when the sequence should be marked anomalous from the
            shared group label contract.
    """
    if group_label_is_anomalous is True:
        return True
    if group_label_is_anomalous is False:
        return False
    if entity_id is None:
        return False
    return is_anomalous_label(label_for_group(entity_id))


def _parameters_as_list(params: ExtractedParameters) -> list[str]:
    """Return template parameters as a concrete list.

    Args:
        params (ExtractedParameters): Parameters returned by template inference.

    Returns:
        list[str]: Concrete parameter list for sequence materialisation.
    """
    if _is_str_list(params):
        return params
    return list(params)


def _is_str_list(params: ExtractedParameters) -> TypeGuard[list[str]]:
    """Return whether the parameter collection is already a concrete list.

    Args:
        params (ExtractedParameters): Parameters returned by template inference.

    Returns:
        TypeGuard[list[str]]: `True` when `params` is already a concrete
            `list[str]`.
    """
    return type(params) is list


def _split_label_from_row_split_labels(
    row_split_labels: Collection[SplitLabel],
) -> SplitLabel:
    """Return the preserved split label for one grouped window.

    Args:
        row_split_labels (Collection[SplitLabel]): Raw-entry split labels
            aligned with one grouped window.

    Returns:
        SplitLabel: Window-level split label, keeping train precedence when a
            window straddles the raw-entry cutoff.
    """
    if any(label is SplitLabel.TRAIN for label in row_split_labels):
        return SplitLabel.TRAIN
    if any(label is SplitLabel.TEST for label in row_split_labels):
        return SplitLabel.TEST
    return SplitLabel.IGNORED


@dataclass(slots=True, frozen=True, kw_only=True)
class NonEntitySequenceBuilder(SequenceBuilder):
    """Sequence builder for non-entity grouping strategies.

    This is a marker subclass to clarify when normal entity logic
    does not apply, such as for fixed-size or time-based windowing.
    """

    @abstractmethod
    def count_windows(self) -> int:
        """Return the total number of windows implied by the sink and config.

        Returns:
            int: Count of windows implied by the sink and current builder config.
        """
        ...

    @override
    def split_count_hint(self) -> SequenceSplitCounts:
        """Return the exact split-count summary for non-entity grouping.

        Returns:
            SequenceSplitCounts: Exact split counts for the grouping strategy.
        """
        return self._split_counts(self.count_windows())

    @override
    def iter_test_sequences(self) -> Iterator[TemplateSequence]:
        """Yield only the test suffix used for detector scoring.

        Yields:
            TemplateSequence: Sequences assigned to the test split.
        """
        rows_iter = self.iter_grouped_rows()
        infer_template = functools.lru_cache(maxsize=50_000)(self.infer_template)
        label_for_group = functools.lru_cache(maxsize=100_000)(self.label_for_group)

        split_counts = self.split_count_hint()
        train_limit = split_counts.train_count
        test_start = split_counts.total_count - split_counts.test_count

        for window_id, rows in enumerate(rows_iter):
            if window_id >= test_start:
                split_label = SplitLabel.TEST
            elif window_id < train_limit:
                split_label = SplitLabel.TRAIN
            else:
                split_label = SplitLabel.IGNORED
            if split_label is not SplitLabel.TEST:
                continue
            seq = self._build_sequence(
                window_id,
                rows,
                infer_template,
                label_for_group,
                split_label,
                allow_group_label_fallback=False,
            )
            if seq is not None:
                yield seq

    def __iter__(self) -> Iterator[TemplateSequence]:
        """Iterate over template sequences yielded by the configured grouping.

        Yields:
            TemplateSequence: One grouped and template-enriched sequence.
        """
        # Non-entity grouping: simple positional cutoff
        rows_iter = self.iter_grouped_rows()
        infer_template = functools.lru_cache(maxsize=50_000)(self.infer_template)
        label_for_group = functools.lru_cache(maxsize=100_000)(self.label_for_group)

        split_counts = self.split_count_hint()
        train_limit = split_counts.train_count
        test_start = split_counts.total_count - split_counts.test_count

        for window_id, rows in enumerate(rows_iter):
            if window_id >= test_start:
                split_label = SplitLabel.TEST
            elif window_id < train_limit:
                split_label = SplitLabel.TRAIN
            else:
                split_label = SplitLabel.IGNORED
            seq = self._build_sequence(
                window_id,
                rows,
                infer_template,
                label_for_group,
                split_label,
                allow_group_label_fallback=False,
            )
            if seq is not None:
                yield seq


@dataclass(slots=True, frozen=True, kw_only=True)
class _RawPositionFixedWindow:
    """Raw-position fixed-window payload used by Thunderbird reconstruction."""

    window_id: int
    rows: tuple[StructuredLine, ...]
    label: int


@dataclass(slots=True, frozen=True, kw_only=True)
class FixedSequenceBuilder(NonEntitySequenceBuilder):
    """Sequence builder for fixed-size window grouping.

    Attributes:
        window_size (int): Number of rows per emitted window.
        step (int | None): Row advance between windows. `None` means
            non-overlapping windows.
        window_basis (FixedWindowBasis): Whether windows are built over the
            compacted structured rows or over the raw line positions.
        window_alignment_offset (int): Raw-position offset before the first
            full raw-position window.
    """

    window_size: int
    step: int | None = None
    window_basis: FixedWindowBasis = FixedWindowBasis.COMPACTED_ROWS
    window_alignment_offset: int = 0

    @override
    def iter_grouped_rows(self) -> Iterator[Collection[StructuredLine]]:
        """Yield rows grouped by fixed-size windows.

        Yields:
            Collection[StructuredLine]: Fixed-size row windows.
        """
        if self.window_basis is FixedWindowBasis.RAW_POSITIONS:
            for window in self._iter_raw_position_windows_with_labels():
                yield window.rows
            return
        yield from self.sink.iter_fixed_window_sequences(
            self.window_size,
            step_size=self.step,
        )()

    @override
    def count_windows(self) -> int:
        """Return the number of fixed-size windows.

        Returns:
            int: Count of fixed-size windows implied by the sink and config.
        """
        if self.window_basis is FixedWindowBasis.RAW_POSITIONS:
            return self._count_raw_position_windows()
        return self._count_fixed_windows(
            sink=self.sink,
            window_size=self.window_size,
            step=self.step,
        )

    def _count_raw_position_windows(self) -> int:
        """Return the number of fixed windows over raw source positions."""
        step = self._raw_position_window_step()

        raw_count = self._count_raw_positions()
        usable = raw_count - self.window_alignment_offset
        if usable < self.window_size:
            return 0
        return 1 + ((usable - self.window_size) // step)

    def _raw_position_window_step(self) -> int:
        """Validate and return the configured raw-position step size.

        Returns:
            int: Raw-position step size, or zero when no full windows fit.

        Raises:
            ValueError: If the raw-position window contract is invalid.
        """
        if self.window_size <= 0:
            return 0
        step = self.step or self.window_size
        if step <= 0:
            return 0
        if step != self.window_size:
            msg = "raw-position fixed windows require step to match window_size."
            raise ValueError(msg)
        if self.window_alignment_offset < 0:
            msg = "window_alignment_offset must be non-negative."
            raise ValueError(msg)
        return step

    def _count_raw_positions(self) -> int:
        """Count raw positions available in the materialised raw slice.

        Returns:
            int: Number of raw positions in the source file.
        """
        with self.sink.raw_dataset_path.open(
            encoding="utf-8",
            errors="replace",
        ) as handle:
            return sum(1 for _ in handle)

    def _iter_raw_position_windows_with_labels(
        self,
    ) -> Iterator[_RawPositionFixedWindow]:
        """Yield raw-position windows and their raw-line anomaly labels.

        Yields:
            _RawPositionFixedWindow: Full raw-position window payload.

        Raises:
            ValueError: If the raw-position window contract is invalid.
            TypeError: If the builder is not bound to a Thunderbird parser.
        """
        if self.window_basis is not FixedWindowBasis.RAW_POSITIONS:
            msg = "raw-position windows are only available in raw-position mode."
            raise ValueError(msg)
        if not isinstance(self.sink.parser, ThunderbirdParser):
            msg = "raw-position fixed windows currently require ThunderbirdParser."
            raise TypeError(msg)

        window_rows: list[StructuredLine] = []
        window_label = 0
        usable_position = 0
        window_id = 0

        with self.sink.raw_dataset_path.open(
            encoding="utf-8",
            errors="replace",
        ) as handle:
            for raw_line_order, raw_line in enumerate(handle):
                if raw_line_order < self.window_alignment_offset:
                    continue

                usable_position += 1
                raw_label = ThunderbirdParser.raw_label_for_line(raw_line)
                if is_anomalous_label(raw_label):
                    window_label = 1

                parsed = self.sink.parser.parse_line(raw_line)
                if parsed is not None:
                    window_rows.append(
                        StructuredLine.with_line_order(
                            line_order=raw_line_order,
                            base=parsed,
                        ),
                    )

                if usable_position % self.window_size == 0:
                    yield _RawPositionFixedWindow(
                        window_id=window_id,
                        rows=tuple(window_rows),
                        label=window_label,
                    )
                    window_rows = []
                    window_label = 0
                    window_id += 1

    @override
    def __iter__(self) -> Iterator[TemplateSequence]:
        """Yield fixed-size windows with the configured basis.

        Yields:
            TemplateSequence: One grouped template sequence per fixed window.
        """
        if self.window_basis is FixedWindowBasis.COMPACTED_ROWS:
            yield from NonEntitySequenceBuilder.__iter__(self)
            return

        infer_template = functools.lru_cache(maxsize=50_000)(self.infer_template)
        label_for_group = functools.lru_cache(maxsize=100_000)(self.label_for_group)
        split_counts = self.split_count_hint()
        test_start = split_counts.total_count - split_counts.test_count

        for window in self._iter_raw_position_windows_with_labels():
            if window.window_id >= test_start:
                split_label = SplitLabel.TEST
            elif window.window_id < split_counts.train_count:
                split_label = SplitLabel.TRAIN
            else:
                split_label = SplitLabel.IGNORED
            seq = self._build_sequence(
                window.window_id,
                window.rows,
                infer_template,
                label_for_group,
                split_label,
                allow_group_label_fallback=False,
                sequence_label=window.label,
            )
            if seq is not None:
                yield seq

    @override
    def train_sequence_count_unit_hint(self) -> str:
        """Return the unit label for fixed-window train progress.

        Returns:
            str: Unit label for fixed-window train progress.
        """
        return "windows"


@dataclass(slots=True, frozen=True, kw_only=True)
class TimeSequenceBuilder(NonEntitySequenceBuilder):
    """Sequence builder for time-window grouping.

    Attributes:
        time_span_ms (int): Duration of each emitted window in milliseconds.
        step (int | None): Window advance in milliseconds. `None` means
            non-overlapping windows.
    """

    time_span_ms: int
    step: int | None = None

    @override
    def train_sequence_count_unit_hint(self) -> str:
        """Return the unit label for time-window train progress.

        Returns:
            str: Unit label for time-window train progress.
        """
        return "windows"

    @override
    def iter_grouped_rows(self) -> Iterator[Collection[StructuredLine]]:
        """Return rows grouped by time windows.

        Returns:
            Iterator[Collection[StructuredLine]]: Time-based row windows.
        """
        return self.sink.iter_time_window_sequences(
            self.time_span_ms,
            step_span_ms=self.step,
        )()

    @override
    def count_windows(self) -> int:
        """Return the number of time windows.

        Returns:
            int: Count of time windows implied by the sink timestamps and config.
        """
        return self._count_time_windows(
            sink=self.sink,
            time_span_ms=self.time_span_ms,
            step=self.step,
        )

    @override
    def __iter__(self) -> Iterator[TemplateSequence]:
        """Iterate over time windows with optional raw-entry split semantics.

        Yields:
            TemplateSequence: One grouped time window, optionally segmented
                according to a raw-entry split applied before grouping.
        """
        if self.split_application_order == SplitApplicationOrder.AFTER_GROUPING:
            yield from NonEntitySequenceBuilder.__iter__(self)
            return

        infer_template = functools.lru_cache(maxsize=50_000)(self.infer_template)
        label_for_group = functools.lru_cache(maxsize=100_000)(self.label_for_group)
        row_labels, _ = self._build_row_split_labels()

        for window_id, rows in enumerate(self.iter_grouped_rows()):
            split_label = _split_label_from_row_split_labels(
                [row_labels.get(row.line_order, SplitLabel.TRAIN) for row in rows],
            )
            yield from self._build_sequences_for_group(
                window_id=window_id,
                rows=rows,
                infer_template=infer_template,
                label_for_group=label_for_group,
                split_label=split_label,
                row_labels=row_labels,
                train_only_normal_entities=False,
            )


@dataclass(slots=True, frozen=True, kw_only=True)
class ChronologicalStreamSequenceBuilder(NonEntitySequenceBuilder):
    """Sequence builder for chronological raw-entry stream chunks.

    Attributes:
        chunk_size (int): Maximum number of raw entries per emitted chunk.
        continuous_context (bool): Whether adjacent chunks should carry model
            state across sequence boundaries.
    """

    chunk_size: int = 100_000
    continuous_context: bool = True

    @override
    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            msg = "chunk_size must be a positive integer."
            raise ValueError(msg)
        SequenceBuilder.__post_init__(self)

    @override
    def train_sequence_count_unit_hint(self) -> str:
        """Return the unit label for stream chunks.

        Returns:
            str: Human-readable unit label for stream progress.
        """
        return "chunks"

    @override
    def iter_grouped_rows(self) -> Iterator[Collection[StructuredLine]]:
        """Return rows grouped into deterministic chronological chunks.

        Returns:
            Iterator[Collection[StructuredLine]]: Deterministic chronological
                chunks of structured rows.
        """

        def _iter() -> Iterator[Collection[StructuredLine]]:
            chunk: list[StructuredLine] = []
            for row in self._iter_source_order_rows():
                chunk.append(row)
                if len(chunk) >= self.chunk_size:
                    yield tuple(chunk)
                    chunk = []
            if chunk:
                yield tuple(chunk)

        return _iter()

    @override
    def count_windows(self) -> int:
        """Return the number of chronological stream chunks.

        Returns:
            int: Count of chronological stream chunks implied by the sink.
        """
        row_count = sum(1 for _ in self._iter_source_order_rows())
        if row_count <= 0:
            return 0
        return math.ceil(row_count / self.chunk_size)

    def __iter__(self) -> Iterator[TemplateSequence]:
        """Iterate over chronological stream chunks with optional raw splits.

        Yields:
            TemplateSequence: One preserved chronological chunk per emitted
            sequence. When a raw-entry split is active, per-event training
            eligibility is attached through `training_event_mask` instead of
            fragmenting the chunk.
        """
        if self.split_application_order == SplitApplicationOrder.AFTER_GROUPING:
            yield from NonEntitySequenceBuilder.__iter__(self)
            return

        infer_template = functools.lru_cache(maxsize=50_000)(self.infer_template)
        label_for_group = functools.lru_cache(maxsize=100_000)(self.label_for_group)
        row_labels, _ = self._build_row_split_labels()

        for window_id, rows in enumerate(self.iter_grouped_rows()):
            row_split_labels = [
                row_labels.get(row.line_order, SplitLabel.TRAIN) for row in rows
            ]
            split_label = self._split_label_for_chronological_chunk(
                row_split_labels,
            )
            evaluation_event_mask = tuple(
                row_labels.get(row.line_order, SplitLabel.TRAIN) is SplitLabel.TEST
                for row in rows
            )
            training_event_mask = tuple(
                (row_labels.get(row.line_order, SplitLabel.TRAIN) is SplitLabel.TRAIN)
                and not is_anomalous_label(row.anomalous)
                for row in rows
            )
            sequence = self._build_sequence(
                window_id=window_id,
                rows=rows,
                infer_template=infer_template,
                label_for_group=label_for_group,
                split_label=split_label,
                training_event_mask=training_event_mask,
                evaluation_event_mask=evaluation_event_mask,
                continuous_context=self.continuous_context,
            )
            if sequence is not None:
                yield sequence

    def with_continuous_context(
        self,
        *,
        enabled: bool = True,
    ) -> Self:
        """Treat consecutive stream chunks as one continuous stream.

        Args:
            enabled (bool): Whether to carry model state across chunk
                boundaries.

        Returns:
            Self: Copy with updated continuity behaviour.
        """
        return replace(self, continuous_context=enabled)

    @staticmethod
    def _split_label_for_chronological_chunk(
        row_split_labels: Collection[SplitLabel],
    ) -> SplitLabel:
        """Return the preserved split label for one chronological chunk.

        Args:
            row_split_labels (Collection[SplitLabel]): Raw-entry split labels
                aligned with one preserved chronological chunk.

        Returns:
            SplitLabel: Chunk-level split label.

        The raw-entry stream keeps chunk boundaries intact. When a chunk
        straddles the split cutoff, training takes precedence so the chunk
        remains available to the training prefix while the event-level mask
        suppresses ineligible targets.
        """
        if any(label is SplitLabel.TRAIN for label in row_split_labels):
            return SplitLabel.TRAIN
        if any(label is SplitLabel.TEST for label in row_split_labels):
            return SplitLabel.TEST
        return SplitLabel.IGNORED
