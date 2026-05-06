"""Utilities for building template sequences from structured log lines.

The module groups parsed log lines into windows (entity, fixed-size, or
time-based) and decorates them with inferred templates and anomaly labels.
"""

from __future__ import annotations

import functools
import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

from typing_extensions import Self, override

from anomalog.parsers.structured.contracts import is_anomalous_label
from anomalog.representations import (
    SequenceRepresentation,
    SequenceRepresentationView,
    TRepresentation,
)
from anomalog.split_validation import validate_split_fractions

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator

    from anomalog.parsers.structured.contracts import StructuredLine, StructuredSink
    from anomalog.parsers.template.dataset import (
        ExtractedParameters,
        LogTemplate,
        TemplatedDataset,
    )
    from experiments.models.base import SequenceSummary


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
        source_order_iter = getattr(
            self.sink,
            "iter_structured_lines_in_source_order",
            None,
        )
        if callable(source_order_iter):
            yield from source_order_iter()()
            return

        rows = list(self.sink.iter_structured_lines()())
        rows.sort(key=lambda row: row.line_order)
        yield from rows

    def _build_row_split_labels(  # noqa: C901
        self,
    ) -> tuple[dict[int, SplitLabel], RawEntrySplitSummary | None]:
        """Build raw-entry split labels keyed by line order.

        Returns:
            tuple[dict[int, SplitLabel], RawEntrySplitSummary | None]: Row-level
                split labels and an audit summary when a raw-entry split is
                active.

        Raises:
            ValueError: If the configured raw-entry split mode is unsupported.
        """
        if (
            self.split_mode is None
            or self.split_application_order != SplitApplicationOrder.BEFORE_GROUPING
        ):
            return {}, None

        split_mode = self.split_mode
        if split_mode is None:
            msg = "raw-entry split mode must be set when building a split summary."
            raise ValueError(msg)

        ordered_rows = list(self._iter_source_order_rows())
        total_rows = len(ordered_rows)
        labels: dict[int, SplitLabel] = {}

        def _empty_summary(cutoff_entry_index: int) -> RawEntrySplitSummary:
            return RawEntrySplitSummary(
                split_mode=split_mode.value,
                application_order=self.split_application_order.value,
                cutoff_entry_index=cutoff_entry_index,
                train_raw_entry_count=0,
                train_normal_entry_count=0,
                train_anomalous_entry_count=0,
                test_raw_entry_count=0,
                test_normal_entry_count=0,
                test_anomalous_entry_count=0,
            )

        if total_rows == 0:
            return {}, _empty_summary(0)

        if split_mode == RawEntrySplitMode.PREFIX_COUNT:
            requested_train_rows = min(
                total_rows,
                int(self.train_entry_count or 0),
            )
            cutoff_entry_index = requested_train_rows
            for index, row in enumerate(ordered_rows):
                label = (
                    SplitLabel.TRAIN if index < cutoff_entry_index else SplitLabel.TEST
                )
                labels[row.line_order] = label
            train_rows = ordered_rows[:cutoff_entry_index]
            test_rows = ordered_rows[cutoff_entry_index:]
            return labels, RawEntrySplitSummary(
                split_mode=split_mode.value,
                application_order=self.split_application_order.value,
                cutoff_entry_index=cutoff_entry_index,
                train_raw_entry_count=len(train_rows),
                train_normal_entry_count=sum(
                    1 for row in train_rows if not is_anomalous_label(row.anomalous)
                ),
                train_anomalous_entry_count=sum(
                    1 for row in train_rows if is_anomalous_label(row.anomalous)
                ),
                test_raw_entry_count=len(test_rows),
                test_normal_entry_count=sum(
                    1 for row in test_rows if not is_anomalous_label(row.anomalous)
                ),
                test_anomalous_entry_count=sum(
                    1 for row in test_rows if is_anomalous_label(row.anomalous)
                ),
            )

        if split_mode == RawEntrySplitMode.PREFIX_FRACTION:
            train_fraction = float(self.train_entry_fraction or 0.0)
            requested_train_rows = min(
                total_rows,
                math.ceil(train_fraction * total_rows),
            )
            cutoff_entry_index = requested_train_rows
            for index, row in enumerate(ordered_rows):
                label = (
                    SplitLabel.TRAIN if index < cutoff_entry_index else SplitLabel.TEST
                )
                labels[row.line_order] = label
            train_rows = ordered_rows[:cutoff_entry_index]
            test_rows = ordered_rows[cutoff_entry_index:]
            return labels, RawEntrySplitSummary(
                split_mode=split_mode.value,
                application_order=self.split_application_order.value,
                cutoff_entry_index=cutoff_entry_index,
                train_raw_entry_count=len(train_rows),
                train_normal_entry_count=sum(
                    1 for row in train_rows if not is_anomalous_label(row.anomalous)
                ),
                train_anomalous_entry_count=sum(
                    1 for row in train_rows if is_anomalous_label(row.anomalous)
                ),
                test_raw_entry_count=len(test_rows),
                test_normal_entry_count=sum(
                    1 for row in test_rows if not is_anomalous_label(row.anomalous)
                ),
                test_anomalous_entry_count=sum(
                    1 for row in test_rows if is_anomalous_label(row.anomalous)
                ),
            )

        if split_mode == RawEntrySplitMode.PREFIX_NORMAL_FRACTION:
            normal_total = sum(
                1 for row in ordered_rows if not is_anomalous_label(row.anomalous)
            )
            target_normal_rows = min(
                normal_total,
                math.ceil((self.train_normal_entry_fraction or 0.0) * normal_total),
            )
            normal_rows_seen = 0
            cutoff_entry_index = total_rows
            for index, row in enumerate(ordered_rows):
                if normal_rows_seen >= target_normal_rows:
                    labels[row.line_order] = SplitLabel.TEST
                    continue
                if is_anomalous_label(row.anomalous):
                    labels[row.line_order] = SplitLabel.IGNORED
                    continue
                labels[row.line_order] = SplitLabel.TRAIN
                normal_rows_seen += 1
                cutoff_entry_index = index + 1
            train_rows = [
                row
                for row in ordered_rows
                if labels[row.line_order] is SplitLabel.TRAIN
            ]
            test_rows = [
                row for row in ordered_rows if labels[row.line_order] is SplitLabel.TEST
            ]
            ignored_rows = [
                row
                for row in ordered_rows
                if labels[row.line_order] is SplitLabel.IGNORED
            ]
            return labels, RawEntrySplitSummary(
                split_mode=split_mode.value,
                application_order=self.split_application_order.value,
                cutoff_entry_index=cutoff_entry_index,
                train_raw_entry_count=len(train_rows),
                train_normal_entry_count=sum(
                    1 for row in train_rows if not is_anomalous_label(row.anomalous)
                ),
                train_anomalous_entry_count=sum(
                    1 for row in train_rows if is_anomalous_label(row.anomalous)
                ),
                test_raw_entry_count=len(test_rows),
                test_normal_entry_count=sum(
                    1 for row in test_rows if not is_anomalous_label(row.anomalous)
                ),
                test_anomalous_entry_count=sum(
                    1 for row in test_rows if is_anomalous_label(row.anomalous)
                ),
                ignored_raw_entry_count=len(ignored_rows),
                ignored_normal_entry_count=sum(
                    1 for row in ignored_rows if not is_anomalous_label(row.anomalous)
                ),
                ignored_anomalous_entry_count=sum(
                    1 for row in ignored_rows if is_anomalous_label(row.anomalous)
                ),
            )

        msg = f"Unsupported raw-entry split mode: {split_mode.value}"
        raise ValueError(msg)

    @staticmethod
    def _split_rows_by_label(
        rows: Collection[StructuredLine],
        row_labels: dict[int, SplitLabel],
    ) -> Iterator[tuple[SplitLabel, list[StructuredLine]]]:
        """Yield contiguous row segments that share the same split label.

        Args:
            rows (Collection[StructuredLine]): Structured rows in grouped
                source order.
            row_labels (dict[int, SplitLabel]): Raw-entry split labels keyed by
                `line_order`.

        Yields:
            tuple[SplitLabel, list[StructuredLine]]: Contiguous segments that
                share the same split label.
        """
        current_label: SplitLabel | None = None
        current_rows: list[StructuredLine] = []
        for row in rows:
            label = row_labels.get(row.line_order, SplitLabel.TRAIN)
            if current_label is None or label is current_label:
                current_label = label if current_label is None else current_label
                current_rows.append(row)
                continue
            yield current_label, current_rows
            current_label = label
            current_rows = [row]
        if current_label is not None and current_rows:
            yield current_label, current_rows

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
        row_labels, summary = self._build_row_split_labels()
        if summary is None:
            return None
        return replace(
            summary,
            straddling_group_count=self._count_straddling_groups(row_labels),
            straddling_group_policy=self.straddling_group_policy.value,
        )

    def _count_straddling_groups(self, row_labels: dict[int, SplitLabel]) -> int:
        """Count grouped windows that cross the raw-entry split boundary.

        Args:
            row_labels (dict[int, SplitLabel]): Raw-entry split labels keyed by
                `line_order`.

        Returns:
            int: Number of grouped windows that contain rows from both sides of
                the raw-entry split boundary.
        """
        if not row_labels:
            return 0
        straddling_groups = 0
        for rows in self.iter_grouped_rows():
            labels = {row_labels.get(row.line_order, SplitLabel.TRAIN) for row in rows}
            if len(labels) > 1:
                straddling_groups += 1
        return straddling_groups

    @abstractmethod
    def iter_grouped_rows(self) -> Iterator[Collection[StructuredLine]]:
        """Return grouped rows for the configured strategy.

        Returns:
            Iterator[Collection[StructuredLine]]: Iterator over grouped windows
                of structured rows.
        """
        ...

    def _build_sequence(  # noqa: PLR0913, PLR0917
        self,
        window_id: int,
        rows: Collection[StructuredLine],
        infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]],
        label_for_group: Callable[[str], int | None],
        split_label: SplitLabel,
        training_event_mask: tuple[bool, ...] | None = None,
        evaluation_event_mask: tuple[bool, ...] | None = None,
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
            training_event_mask (tuple[bool, ...] | None): Optional per-event
                training-target eligibility mask for preserved chronological
                chunks.
            evaluation_event_mask (tuple[bool, ...] | None): Optional per-event
                scoring-target eligibility mask for preserved chronological
                chunks.

        Returns:
            TemplateSequence | None: Built sequence, or `None` for empty
                windows. Sequence labels are derived from both inline row
                labels and resolved group labels under the shared anomaly
                semantics.
        """
        if not rows:
            return None

        events: list[tuple[str, list[str], int | None]] = []
        seq_label = 0
        prev_ts: int | None = None

        unique_ids = self._entity_ids_for_rows(rows)
        event_labels = tuple(r.anomalous for r in rows)

        for r in rows:
            template, params = infer_template(r.untemplated_message_text)
            dt, prev_ts = self._compute_dt(prev_ts, r.timestamp_unix_ms)

            events.append((template, list(params), dt))

            if seq_label == 1:
                continue

            line_lab = getattr(r, "anomalous", None)
            if is_anomalous_label(line_lab):
                seq_label = 1
                continue

            ent = r.entity_id
            if ent is not None and is_anomalous_label(label_for_group(ent)):
                seq_label = 1

        return TemplateSequence(
            events=events,
            label=seq_label,
            entity_ids=unique_ids,
            window_id=window_id,
            split_label=split_label,
            event_labels=(
                event_labels
                if any(label is not None for label in event_labels)
                else None
            ),
            training_event_mask=training_event_mask,
            evaluation_event_mask=evaluation_event_mask,
        )

    def _build_sequences_for_group(  # noqa: C901, PLR0912, PLR0913
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

        Raises:
            ValueError: If the configured straddling policy is unsupported.
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
            )
            if seq is not None:
                yield seq
            return

        segments = list(self._split_rows_by_label(rows, row_labels))
        if not segments:
            return

        if (
            self.straddling_group_policy
            == StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES
        ):
            for offset, (segment_label, segment_rows) in enumerate(segments):
                effective_label = segment_label
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

        if self.straddling_group_policy == StraddlingGroupPolicy.DROP_STRADDLERS:
            unique_labels = {segment_label for segment_label, _ in segments}
            if len(unique_labels) > 1:
                return
            seq = self._build_sequence(
                window_id,
                rows,
                infer_template,
                label_for_group,
                next(iter(unique_labels)),
            )
            if seq is not None:
                yield seq
            return

        if self.straddling_group_policy == StraddlingGroupPolicy.ASSIGN_BY_FIRST_EVENT:
            split_label = segments[0][0]
        elif self.straddling_group_policy == StraddlingGroupPolicy.ASSIGN_BY_LAST_EVENT:
            split_label = segments[-1][0]
        else:
            msg = f"Unsupported straddling policy: {self.straddling_group_policy.value}"
            raise ValueError(msg)

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

        if n <= window_size:
            return 1
        return 1 + math.ceil((n - window_size) / step)

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
    """

    train_on_normal_entities_only: bool = False

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
        counts, _ = self._entity_split_counts(label_for_group=self.label_for_group)
        return counts

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
                if entity_id is None:
                    split_label = SplitLabel.TRAIN
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
            if window_id >= test_start_index:
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
            )
            if seq is not None:
                if (
                    self.train_on_normal_entities_only
                    and split_label is SplitLabel.TRAIN
                    and seq.label == 0
                ):
                    normals_seen_in_train += 1
                yield seq


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
            )
            if seq is not None:
                yield seq


@dataclass(slots=True, frozen=True, kw_only=True)
class FixedSequenceBuilder(NonEntitySequenceBuilder):
    """Sequence builder for fixed-size window grouping.

    Attributes:
        window_size (int): Number of rows per emitted window.
        step (int | None): Row advance between windows. `None` means
            non-overlapping windows.
    """

    window_size: int
    step: int | None = None

    @override
    def iter_grouped_rows(self) -> Iterator[Collection[StructuredLine]]:
        """Return rows grouped by fixed-size windows.

        Returns:
            Iterator[Collection[StructuredLine]]: Fixed-size row windows.
        """
        return self.sink.iter_fixed_window_sequences(
            self.window_size,
            step_size=self.step,
        )()

    @override
    def count_windows(self) -> int:
        """Return the number of fixed-size windows.

        Returns:
            int: Count of fixed-size windows implied by the sink and config.
        """
        return self._count_fixed_windows(
            sink=self.sink,
            window_size=self.window_size,
            step=self.step,
        )

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


@dataclass(slots=True, frozen=True, kw_only=True)
class ChronologicalStreamSequenceBuilder(NonEntitySequenceBuilder):
    """Sequence builder for chronological raw-entry stream chunks.

    Attributes:
        chunk_size (int): Maximum number of raw entries per emitted chunk.
    """

    chunk_size: int = 100_000

    @override
    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            msg = "chunk_size must be a positive integer."
            raise ValueError(msg)
        super().__post_init__()

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
            yield from super().__iter__()
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
            )
            if sequence is not None:
                yield sequence

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
