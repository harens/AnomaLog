"""Private helpers for raw-entry split planning and straddler policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from anomalog.parsers.structured.contracts import is_anomalous_label

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Iterator, Sequence

    from anomalog.parsers.structured.contracts import StructuredLine


RAW_ENTRY_SPLIT_TRAIN = "train"
RAW_ENTRY_SPLIT_TEST = "test"
RAW_ENTRY_SPLIT_IGNORED = "ignored"

RAW_ENTRY_SPLIT_MODE_PREFIX_COUNT = "raw_entry_prefix_count"
RAW_ENTRY_SPLIT_MODE_PREFIX_FRACTION = "raw_entry_prefix_fraction"
RAW_ENTRY_SPLIT_MODE_PREFIX_NORMAL_FRACTION = "raw_entry_prefix_normal_fraction"

STRADDLING_GROUP_POLICY_SPLIT_PARTIAL_SEQUENCES = "split_partial_sequences"
STRADDLING_GROUP_POLICY_ASSIGN_BY_FIRST_EVENT = "assign_by_first_event"
STRADDLING_GROUP_POLICY_ASSIGN_BY_LAST_EVENT = "assign_by_last_event"
STRADDLING_GROUP_POLICY_DROP_STRADDLERS = "drop_straddlers"


@dataclass(frozen=True, slots=True)
class RawEntrySplitSummaryData:
    """Primitive raw-entry split summary fields.

    Attributes:
        split_mode (str): Raw-entry split strategy used to produce the plan.
        application_order (str): Whether the split was applied before or after
            grouping.
        cutoff_entry_index (int): Zero-based raw-entry index where the test suffix
            begins.
        train_raw_entry_count (int): Raw entries assigned to the training prefix.
        train_normal_entry_count (int): Normal training rows included in the prefix.
        train_anomalous_entry_count (int): Anomalous training rows included in the
            prefix.
        test_raw_entry_count (int): Raw entries assigned to the test suffix.
        test_normal_entry_count (int): Normal rows assigned to the test suffix.
        test_anomalous_entry_count (int): Anomalous rows assigned to the test suffix.
        ignored_raw_entry_count (int): Raw entries excluded from both splits.
        ignored_normal_entry_count (int): Normal rows excluded from both splits.
        ignored_anomalous_entry_count (int): Anomalous rows excluded from both
            splits.
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


@dataclass(frozen=True, slots=True)
class RawEntrySplitPlan:
    """Computed labels and summary data for a raw-entry split.

    Attributes:
        row_labels (dict[int, str]): Per-line split labels keyed by `line_order`.
        summary (RawEntrySplitSummaryData): Primitive counts describing the
            realised split.
    """

    row_labels: dict[int, str]
    summary: RawEntrySplitSummaryData


@dataclass(frozen=True, slots=True)
class RawEntrySplitRequest:
    """Inputs required to build a raw-entry split plan.

    Attributes:
        split_mode (str): Requested chronological raw-entry split strategy.
        application_order (str): Whether the split should happen before or after
            grouping.
        train_entry_count (int | None): Fixed training prefix size for
            count-based splits.
        train_entry_fraction (float | None): Fraction of raw entries to keep
            in training for fraction-based splits.
        train_normal_entry_fraction (float | None): Fraction of normal entries
            to keep in the training prefix for normal-only prefix splits.
    """

    split_mode: str
    application_order: str
    train_entry_count: int | None
    train_entry_fraction: float | None
    train_normal_entry_fraction: float | None


def build_raw_entry_split_plan(
    ordered_rows: Sequence[StructuredLine],
    *,
    request: RawEntrySplitRequest,
) -> RawEntrySplitPlan:
    """Build raw-entry split labels and summary data for one chronology.

    Args:
        ordered_rows (Sequence[StructuredLine]): Structured rows already sorted
            into source chronology.
        request (RawEntrySplitRequest): Split request describing the desired
            raw-entry policy.

    Returns:
        RawEntrySplitPlan: Computed row labels and primitive summary fields.

    Raises:
        ValueError: If the requested split mode is unsupported.
    """
    split_mode = request.split_mode
    if split_mode == RAW_ENTRY_SPLIT_MODE_PREFIX_COUNT:
        return _build_prefix_count_plan(ordered_rows, request=request)
    if split_mode == RAW_ENTRY_SPLIT_MODE_PREFIX_FRACTION:
        return _build_prefix_fraction_plan(ordered_rows, request=request)
    if split_mode == RAW_ENTRY_SPLIT_MODE_PREFIX_NORMAL_FRACTION:
        return _build_prefix_normal_fraction_plan(ordered_rows, request=request)

    msg = f"Unsupported raw-entry split mode: {split_mode}"
    raise ValueError(msg)


def _build_prefix_count_plan(
    ordered_rows: Sequence[StructuredLine],
    *,
    request: RawEntrySplitRequest,
) -> RawEntrySplitPlan:
    total_rows = len(ordered_rows)
    labels: dict[int, str] = {}
    if total_rows == 0:
        return RawEntrySplitPlan(
            row_labels=labels,
            summary=RawEntrySplitSummaryData(
                split_mode=request.split_mode,
                application_order=request.application_order,
                cutoff_entry_index=0,
                train_raw_entry_count=0,
                train_normal_entry_count=0,
                train_anomalous_entry_count=0,
                test_raw_entry_count=0,
                test_normal_entry_count=0,
                test_anomalous_entry_count=0,
            ),
        )

    requested_train_rows = min(total_rows, int(request.train_entry_count or 0))
    cutoff_entry_index = requested_train_rows
    for index, row in enumerate(ordered_rows):
        labels[row.line_order] = (
            RAW_ENTRY_SPLIT_TRAIN
            if index < cutoff_entry_index
            else RAW_ENTRY_SPLIT_TEST
        )
    train_rows = list(ordered_rows[:cutoff_entry_index])
    test_rows = list(ordered_rows[cutoff_entry_index:])
    return RawEntrySplitPlan(
        row_labels=labels,
        summary=RawEntrySplitSummaryData(
            split_mode=request.split_mode,
            application_order=request.application_order,
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
        ),
    )


def _build_prefix_fraction_plan(
    ordered_rows: Sequence[StructuredLine],
    *,
    request: RawEntrySplitRequest,
) -> RawEntrySplitPlan:
    total_rows = len(ordered_rows)
    labels: dict[int, str] = {}
    if total_rows == 0:
        return RawEntrySplitPlan(
            row_labels=labels,
            summary=RawEntrySplitSummaryData(
                split_mode=request.split_mode,
                application_order=request.application_order,
                cutoff_entry_index=0,
                train_raw_entry_count=0,
                train_normal_entry_count=0,
                train_anomalous_entry_count=0,
                test_raw_entry_count=0,
                test_normal_entry_count=0,
                test_anomalous_entry_count=0,
            ),
        )

    requested_train_rows = min(
        total_rows,
        math.ceil(float(request.train_entry_fraction or 0.0) * total_rows),
    )
    cutoff_entry_index = requested_train_rows
    for index, row in enumerate(ordered_rows):
        labels[row.line_order] = (
            RAW_ENTRY_SPLIT_TRAIN
            if index < cutoff_entry_index
            else RAW_ENTRY_SPLIT_TEST
        )
    train_rows = list(ordered_rows[:cutoff_entry_index])
    test_rows = list(ordered_rows[cutoff_entry_index:])
    return RawEntrySplitPlan(
        row_labels=labels,
        summary=RawEntrySplitSummaryData(
            split_mode=request.split_mode,
            application_order=request.application_order,
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
        ),
    )


def _build_prefix_normal_fraction_plan(
    ordered_rows: Sequence[StructuredLine],
    *,
    request: RawEntrySplitRequest,
) -> RawEntrySplitPlan:
    total_rows = len(ordered_rows)
    labels: dict[int, str] = {}
    if total_rows == 0:
        return RawEntrySplitPlan(
            row_labels=labels,
            summary=RawEntrySplitSummaryData(
                split_mode=request.split_mode,
                application_order=request.application_order,
                cutoff_entry_index=0,
                train_raw_entry_count=0,
                train_normal_entry_count=0,
                train_anomalous_entry_count=0,
                test_raw_entry_count=0,
                test_normal_entry_count=0,
                test_anomalous_entry_count=0,
            ),
        )

    normal_total = sum(
        1 for row in ordered_rows if not is_anomalous_label(row.anomalous)
    )
    target_normal_rows = min(
        normal_total,
        math.ceil(float(request.train_normal_entry_fraction or 0.0) * normal_total),
    )
    normal_rows_seen = 0
    cutoff_entry_index = total_rows
    for index, row in enumerate(ordered_rows):
        if normal_rows_seen >= target_normal_rows:
            labels[row.line_order] = RAW_ENTRY_SPLIT_TEST
            continue
        if is_anomalous_label(row.anomalous):
            labels[row.line_order] = RAW_ENTRY_SPLIT_IGNORED
            continue
        labels[row.line_order] = RAW_ENTRY_SPLIT_TRAIN
        normal_rows_seen += 1
        cutoff_entry_index = index + 1
    train_rows = [
        row for row in ordered_rows if labels[row.line_order] == RAW_ENTRY_SPLIT_TRAIN
    ]
    test_rows = [
        row for row in ordered_rows if labels[row.line_order] == RAW_ENTRY_SPLIT_TEST
    ]
    ignored_rows = [
        row for row in ordered_rows if labels[row.line_order] == RAW_ENTRY_SPLIT_IGNORED
    ]
    return RawEntrySplitPlan(
        row_labels=labels,
        summary=RawEntrySplitSummaryData(
            split_mode=request.split_mode,
            application_order=request.application_order,
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
        ),
    )


def split_rows_by_label(
    rows: Collection[StructuredLine],
    row_labels: dict[int, str],
    *,
    default_label: str = RAW_ENTRY_SPLIT_TRAIN,
) -> Iterator[tuple[str, list[StructuredLine]]]:
    """Yield contiguous row segments that share the same split label.

    Args:
        rows (Collection[StructuredLine]): Structured rows to group into
            contiguous label runs.
        row_labels (dict[int, str]): Explicit split labels keyed by `line_order`.
        default_label (str): Label used when a row is missing from `row_labels`.

    Yields:
        tuple[str, list[StructuredLine]]: One label and the contiguous rows
            carrying that label.
    """
    current_label: str | None = None
    current_rows: list[StructuredLine] = []
    for row in rows:
        label = row_labels.get(row.line_order, default_label)
        if current_label is None or label == current_label:
            current_label = label if current_label is None else current_label
            current_rows.append(row)
            continue
        yield current_label, current_rows
        current_label = label
        current_rows = [row]
    if current_label is not None and current_rows:
        yield current_label, current_rows


def count_straddling_groups(
    grouped_rows: Iterable[Collection[StructuredLine]],
    row_labels: dict[int, str],
    *,
    default_label: str = RAW_ENTRY_SPLIT_TRAIN,
) -> int:
    """Count grouped windows that cross the raw-entry split boundary.

    Args:
        grouped_rows (Iterable[Collection[StructuredLine]]): Grouped row
            windows to inspect.
        row_labels (dict[int, str]): Explicit split labels keyed by `line_order`.
        default_label (str): Label used when a row is missing from `row_labels`.

    Returns:
        int: Number of grouped windows that contain rows from both sides of the
            raw-entry split boundary.
    """
    straddling_groups = 0
    for rows in grouped_rows:
        labels = {row_labels.get(row.line_order, default_label) for row in rows}
        if len(labels) > 1:
            straddling_groups += 1
    return straddling_groups


def resolve_straddling_group_label(
    policy: str,
    segments: Sequence[tuple[str, list[StructuredLine]]],
) -> str | None:
    """Return the split label assigned to a straddling group, if any.

    Args:
        policy (str): Policy name controlling how straddlers are resolved.
        segments (Sequence[tuple[str, list[StructuredLine]]]): Contiguous label
            segments that compose the grouped rows.

    Returns:
        str | None: Assigned split label, or `None` when the group should be
            dropped.

    Raises:
        ValueError: If the policy is unsupported or requires caller handling.
    """
    if policy == STRADDLING_GROUP_POLICY_DROP_STRADDLERS:
        unique_labels = {segment_label for segment_label, _ in segments}
        if len(unique_labels) > 1:
            return None
        return next(iter(unique_labels))
    if policy == STRADDLING_GROUP_POLICY_ASSIGN_BY_FIRST_EVENT:
        return segments[0][0]
    if policy == STRADDLING_GROUP_POLICY_ASSIGN_BY_LAST_EVENT:
        return segments[-1][0]
    if policy == STRADDLING_GROUP_POLICY_SPLIT_PARTIAL_SEQUENCES:
        msg = "split_partial_sequences is handled by the caller."
        raise ValueError(msg)
    msg = f"Unsupported straddling policy: {policy}"
    raise ValueError(msg)
