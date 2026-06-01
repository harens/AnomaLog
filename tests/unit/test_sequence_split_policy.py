"""Tests for raw-entry split policy helpers."""

from __future__ import annotations

import pytest

from anomalog import _sequence_split_policy as split_policy
from anomalog._sequence_split_policy import (
    RawEntrySplitRequest,
    build_raw_entry_split_plan,
    count_straddling_groups,
    resolve_straddling_group_label,
    split_rows_by_label,
)
from tests.unit.helpers import structured_line


def _rows(*labels: int | None) -> list:
    return [
        structured_line(
            line_order=index,
            timestamp_unix_ms=index * 10,
            entity_id=f"entity-{index}",
            untemplated_message_text=f"event-{index}",
            anomalous=label,
        )
        for index, label in enumerate(labels)
    ]


def test_build_raw_entry_split_plan_rejects_unknown_mode() -> None:
    """Unknown raw-entry split modes should fail fast."""
    with pytest.raises(ValueError, match="Unsupported raw-entry split mode"):
        build_raw_entry_split_plan(
            _rows(0, 0),
            request=RawEntrySplitRequest(
                split_mode="missing",
                application_order="before_grouping",
                train_entry_count=None,
                train_entry_fraction=None,
                train_normal_entry_fraction=None,
            ),
        )


@pytest.mark.parametrize(
    "request_",
    [
            RawEntrySplitRequest(
                split_mode=split_policy.RAW_ENTRY_SPLIT_MODE_PREFIX_COUNT,
                application_order="before_grouping",
                train_entry_count=0,
                train_entry_fraction=None,
                train_normal_entry_fraction=None,
            ),
            RawEntrySplitRequest(
                split_mode=split_policy.RAW_ENTRY_SPLIT_MODE_PREFIX_FRACTION,
                application_order="before_grouping",
                train_entry_count=None,
                train_entry_fraction=0.0,
                train_normal_entry_fraction=None,
            ),
            RawEntrySplitRequest(
                split_mode=split_policy.RAW_ENTRY_SPLIT_MODE_PREFIX_NORMAL_FRACTION,
                application_order="before_grouping",
                train_entry_count=None,
                train_entry_fraction=None,
                train_normal_entry_fraction=0.5,
            ),
        ],
)
def test_build_raw_entry_split_plan_handles_empty_rows(
    request_: RawEntrySplitRequest,
) -> None:
    """Empty raw-entry splits should produce zeroed summaries."""
    plan = build_raw_entry_split_plan([], request=request_)

    assert plan.row_labels == {}
    summary = plan.summary
    assert summary.split_mode == request_.split_mode
    assert summary.application_order == request_.application_order
    assert summary.cutoff_entry_index == 0
    assert summary.train_raw_entry_count == 0
    assert summary.train_normal_entry_count == 0
    assert summary.train_anomalous_entry_count == 0
    assert summary.test_raw_entry_count == 0
    assert summary.test_normal_entry_count == 0
    assert summary.test_anomalous_entry_count == 0
    assert summary.ignored_raw_entry_count == 0
    assert summary.ignored_normal_entry_count == 0
    assert summary.ignored_anomalous_entry_count == 0


def test_build_raw_entry_split_plan_counts_ignored_rows_for_normal_only_prefix() -> (
    None
):
    """Normal-only prefix splits should count ignored anomalous rows separately."""
    rows = _rows(1, 0, 0, 1)
    plan = build_raw_entry_split_plan(
        rows,
        request=RawEntrySplitRequest(
            split_mode=split_policy.RAW_ENTRY_SPLIT_MODE_PREFIX_NORMAL_FRACTION,
            application_order="before_grouping",
            train_entry_count=None,
            train_entry_fraction=None,
            train_normal_entry_fraction=0.5,
        ),
    )

    assert plan.row_labels == {
        0: split_policy.RAW_ENTRY_SPLIT_IGNORED,
        1: split_policy.RAW_ENTRY_SPLIT_TRAIN,
        2: split_policy.RAW_ENTRY_SPLIT_TEST,
        3: split_policy.RAW_ENTRY_SPLIT_TEST,
    }
    assert plan.summary.ignored_raw_entry_count == 1
    assert plan.summary.ignored_anomalous_entry_count == 1
    assert plan.summary.train_normal_entry_count == 1
    assert plan.summary.test_raw_entry_count == 2


def test_split_rows_by_label_and_straddling_policy_helpers() -> None:
    """Contiguous label grouping and straddler policies should stay stable."""
    rows = _rows(0, 0, 1, 1)
    row_labels = {0: "train", 1: "train", 2: "test", 3: "ignored"}

    assert list(split_rows_by_label(rows, row_labels)) == [
        ("train", rows[:2]),
        ("test", [rows[2]]),
        ("ignored", [rows[3]]),
    ]
    assert count_straddling_groups([rows[:2], rows[1:3], rows[2:]], row_labels) == 2
    assert resolve_straddling_group_label(
        split_policy.STRADDLING_GROUP_POLICY_DROP_STRADDLERS,
        [("train", rows[:2]), ("train", [rows[2]])],
    ) == "train"
    assert resolve_straddling_group_label(
        split_policy.STRADDLING_GROUP_POLICY_ASSIGN_BY_FIRST_EVENT,
        [("train", rows[:2]), ("test", [rows[2]])],
    ) == "train"
    assert resolve_straddling_group_label(
        split_policy.STRADDLING_GROUP_POLICY_ASSIGN_BY_LAST_EVENT,
        [("train", rows[:2]), ("test", [rows[2]])],
    ) == "test"
    assert resolve_straddling_group_label(
        split_policy.STRADDLING_GROUP_POLICY_DROP_STRADDLERS,
        [("train", rows[:1]), ("test", rows[1:2])],
    ) is None
    with pytest.raises(ValueError, match="split_partial_sequences is handled"):
        resolve_straddling_group_label(
            split_policy.STRADDLING_GROUP_POLICY_SPLIT_PARTIAL_SEQUENCES,
            [("train", rows[:1]), ("test", rows[1:2])],
        )
    with pytest.raises(ValueError, match="Unsupported straddling policy"):
        resolve_straddling_group_label("missing", [("train", rows[:1])])
