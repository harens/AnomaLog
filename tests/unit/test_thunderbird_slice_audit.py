"""Unit tests for the Thunderbird slice contract audit helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa
import pytest

from experiments.audit import thunderbird_slice_audit
from experiments.audit.thunderbird_slice_audit import (
    count_fixed_window_flags,
    expand_raw_position_flags,
    find_matching_offsets,
)

if TYPE_CHECKING:
    from pathlib import Path


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


def test_count_fixed_window_flags_rejects_invalid_contracts() -> None:
    """Window counting should validate the size and offset contract."""
    flags = np.asarray([0, 1], dtype=np.uint8)

    with pytest.raises(ValueError, match="window_size must be positive"):
        count_fixed_window_flags(
            flags,
            ordering="raw_positions",
            window_size=0,
        )
    with pytest.raises(ValueError, match="offset must be non-negative"):
        count_fixed_window_flags(
            flags,
            ordering="raw_positions",
            offset=-1,
        )


def test_audit_thunderbird_slice_uses_fake_parquet_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The audit helper should summarise the cached slice and JSON wrapper.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the parquet sink with a
            deterministic fake so the audit path can run without cached data.
        tmp_path (Path): Temporary directory used as the fake cache root.
    """

    class _FakeScanner:
        @staticmethod
        def to_batches() -> list[pa.RecordBatch]:
            return [
                pa.record_batch(
                    [pa.array([0, 1])],
                    names=["anomalous"],
                ),
            ]

    class _FakeDataset:
        @staticmethod
        def to_table(*, columns: list[str], **kwargs: object) -> pa.Table:
            del columns, kwargs
            return pa.table(
                {
                    "line_order": [160_000_000, 160_000_001],
                    "anomalous": [0, 1],
                },
            )

        @staticmethod
        def scanner(**kwargs: object) -> _FakeScanner:
            del kwargs
            return _FakeScanner()

    class _FakeSink:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        @staticmethod
        def _dataset() -> _FakeDataset:
            return _FakeDataset()

    monkeypatch.setattr(
        thunderbird_slice_audit,
        "ParquetStructuredSink",
        _FakeSink,
    )

    payload = thunderbird_slice_audit.audit_thunderbird_slice(
        cache_root=tmp_path,
        start_line_order=160_000_000,
        end_line_order=160_000_001,
    )

    assert payload["cache_root"] == str(tmp_path)
    audit_json = thunderbird_slice_audit.audit_thunderbird_slice_json(
        cache_root=tmp_path,
        start_line_order=160_000_000,
        end_line_order=160_000_001,
    )
    assert '"materialised_rows": 2' in audit_json
    assert '"total_windows": 0' in audit_json
