"""Unit tests for the Thunderbird slice contract audit helpers."""

from __future__ import annotations

import numpy as np

from experiments.audit.thunderbird_slice_audit import (
    count_fixed_window_flags,
    expand_raw_position_flags,
    find_matching_offsets,
)


def test_expand_raw_position_flags_pads_skipped_rows_as_normal() -> None:
    """Gap expansion should preserve retained labels and fill skips as normal."""
    raw = expand_raw_position_flags(
        [10, 12, 13],
        [False, True, False],
        start_line_order=10,
        end_line_order=13,
    )

    assert raw.tolist() == [0, 0, 1, 0]


def test_count_fixed_window_flags_respects_offset_alignment() -> None:
    """Window counts should shift when the raw-position offset changes."""
    flags = np.asarray([0, 1, 0, 0, 1, 0], dtype=np.uint8)
    expected_aligned_windows = 3
    expected_shifted_windows = 2
    expected_aligned_train_anomalous = 1
    expected_aligned_test_anomalous = 1
    expected_shifted_train_anomalous = 2
    expected_shifted_test_anomalous = 0

    aligned = count_fixed_window_flags(
        flags,
        ordering="raw_positions",
        train_window_count=2,
        window_size=2,
        offset=0,
    )
    shifted = count_fixed_window_flags(
        flags,
        ordering="raw_positions",
        train_window_count=2,
        window_size=2,
        offset=1,
    )

    assert aligned.total_windows == expected_aligned_windows
    assert aligned.train_anomalous == expected_aligned_train_anomalous
    assert aligned.test_anomalous == expected_aligned_test_anomalous
    assert shifted.total_windows == expected_shifted_windows
    assert shifted.train_anomalous == expected_shifted_train_anomalous
    assert shifted.test_anomalous == expected_shifted_test_anomalous


def test_find_matching_offsets_returns_offsets_that_hit_target_counts() -> None:
    """Offset search should surface every alignment that hits the target counts."""
    flags = np.asarray(
        [
            0,
            0,
            1,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
        ],
        dtype=np.uint8,
    )

    matches = find_matching_offsets(
        flags,
        target_train_anomalous=1,
        target_test_anomalous=0,
        train_window_count=3,
        window_size=2,
        search_limit=4,
    )

    assert [match.offset for match in matches] == [3]
