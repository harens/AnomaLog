"""Thunderbird slice contract audit helpers.

This module reconstructs the Thunderbird benchmark slice from a cached
structured parquet dataset, then compares the fixed-window contract under a
few plausible ordering and compaction rules. It is intentionally separate from
the experiment runner so the audit can be rerun without a full model training
job.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pyarrow.dataset as ds

from anomalog.cache.core import CachePathsConfig
from anomalog.parsers.structured.parquet.sink import ParquetStructuredSink
from anomalog.parsers.structured.parsers import ThunderbirdParser

if TYPE_CHECKING:
    from collections.abc import Sequence


_DEFAULT_START_LINE_ORDER = 159_999_999
_DEFAULT_END_LINE_ORDER = 169_999_998
_DEFAULT_TRAIN_WINDOW_COUNT = 79_996
_DEFAULT_WINDOW_SIZE = 100
_DEFAULT_OFFSET_SEARCH_LIMIT = 100


@dataclass(frozen=True, slots=True)
class WindowContractSummary:
    """Summarise one fixed-window label contract.

    Attributes:
        ordering (Literal["scan", "source", "raw_positions"]): Row ordering
            used to construct windows.
        offset (int): Raw-position alignment offset applied before the first
            full window is emitted.
        total_positions (int): Number of positions considered for windowing.
        total_windows (int): Total number of full windows emitted.
        train_windows (int): Number of windows assigned to the training split.
        test_windows (int): Number of windows assigned to the test split.
        train_anomalous (int): Anomalous windows in the training split.
        test_anomalous (int): Anomalous windows in the test split.
        remainder (int): Trailing positions discarded because they do not form
            a full window.
    """

    ordering: Literal["scan", "source", "raw_positions"]
    offset: int
    total_positions: int
    total_windows: int
    train_windows: int
    test_windows: int
    train_anomalous: int
    test_anomalous: int
    remainder: int

    def as_dict(self) -> dict[str, int | str]:
        """Return a JSON-friendly representation."""
        return {
            "ordering": self.ordering,
            "offset": self.offset,
            "total_positions": self.total_positions,
            "total_windows": self.total_windows,
            "train_windows": self.train_windows,
            "test_windows": self.test_windows,
            "train_anomalous": self.train_anomalous,
            "test_anomalous": self.test_anomalous,
            "remainder": self.remainder,
        }


def expand_raw_position_flags(
    line_orders: Sequence[int],
    flags: Sequence[bool],
    *,
    start_line_order: int,
    end_line_order: int,
) -> np.ndarray:
    """Expand a compacted slice back to raw positions.

    Args:
        line_orders (Sequence[int]): Sorted raw line positions for retained
            structured events.
        flags (Sequence[bool]): Per-row anomaly flags aligned with
            `line_orders`.
        start_line_order (int): Inclusive raw-position start of the slice.
        end_line_order (int): Inclusive raw-position end of the slice.

    Returns:
        np.ndarray: Raw-position anomaly mask where skipped rows are filled in
        as `False`.
    """
    total_positions = end_line_order - start_line_order + 1
    raw = np.zeros(total_positions, dtype=np.uint8)
    cursor = 0
    previous = start_line_order - 1
    for line_order, flag in zip(line_orders, flags, strict=True):
        gap = line_order - previous - 1
        if gap > 0:
            cursor += gap
        if flag:
            raw[cursor] = 1
        cursor += 1
        previous = line_order
    return raw


def count_fixed_window_flags(
    flags: Sequence[bool] | np.ndarray,
    *,
    ordering: Literal["scan", "source", "raw_positions"],
    train_window_count: int = _DEFAULT_TRAIN_WINDOW_COUNT,
    window_size: int = _DEFAULT_WINDOW_SIZE,
    offset: int = 0,
) -> WindowContractSummary:
    """Count fixed windows and anomaly labels for one ordered flag sequence.

    Args:
        flags (Sequence[bool] | np.ndarray): Boolean anomaly mask in the order
            under test.
        ordering (Literal["scan", "source", "raw_positions"]): Label for the
            contract variant being counted.
        train_window_count (int): Number of windows assigned to train.
        window_size (int): Number of positions per non-overlapping window.
        offset (int): Initial positional offset before the first full window.

    Returns:
        WindowContractSummary: Window counts and anomaly totals for the
        requested contract variant.

    Raises:
        ValueError: If the window size or offset is invalid.
    """
    if window_size <= 0:
        msg = "window_size must be positive."
        raise ValueError(msg)
    if offset < 0:
        msg = "offset must be non-negative."
        raise ValueError(msg)

    array = np.asarray(flags, dtype=np.uint8)
    if offset >= len(array):
        return WindowContractSummary(
            ordering=ordering,
            offset=offset,
            total_positions=len(array),
            total_windows=0,
            train_windows=0,
            test_windows=0,
            train_anomalous=0,
            test_anomalous=0,
            remainder=len(array),
        )

    usable = len(array) - offset
    total_windows = usable // window_size
    remainder = usable % window_size
    if total_windows == 0:
        return WindowContractSummary(
            ordering=ordering,
            offset=offset,
            total_positions=len(array),
            total_windows=0,
            train_windows=0,
            test_windows=0,
            train_anomalous=0,
            test_anomalous=0,
            remainder=remainder,
        )

    trimmed = array[offset : offset + (total_windows * window_size)]
    windows = trimmed.reshape(total_windows, window_size)
    train_limit = min(train_window_count, total_windows)
    train_windows = train_limit
    test_windows = total_windows - train_limit
    train_anomalous = 0
    test_anomalous = 0
    for index, window in enumerate(windows):
        is_anomalous = bool(window.any())
        if index < train_limit:
            if is_anomalous:
                train_anomalous += 1
        elif is_anomalous:
            test_anomalous += 1

    return WindowContractSummary(
        ordering=ordering,
        offset=offset,
        total_positions=len(array),
        total_windows=total_windows,
        train_windows=train_windows,
        test_windows=test_windows,
        train_anomalous=train_anomalous,
        test_anomalous=test_anomalous,
        remainder=remainder,
    )


def find_matching_offsets(  # noqa: PLR0913
    flags: Sequence[bool] | np.ndarray,
    *,
    target_train_anomalous: int,
    target_test_anomalous: int,
    train_window_count: int = _DEFAULT_TRAIN_WINDOW_COUNT,
    window_size: int = _DEFAULT_WINDOW_SIZE,
    search_limit: int = _DEFAULT_OFFSET_SEARCH_LIMIT,
) -> list[WindowContractSummary]:
    """Return offsets that reproduce the requested anomalous-window totals."""
    matches: list[WindowContractSummary] = []
    for offset in range(search_limit):
        summary = count_fixed_window_flags(
            flags,
            ordering="raw_positions",
            train_window_count=train_window_count,
            window_size=window_size,
            offset=offset,
        )
        if (
            summary.train_anomalous == target_train_anomalous
            and summary.test_anomalous == target_test_anomalous
        ):
            matches.append(summary)
    return matches


def audit_thunderbird_slice(
    *,
    cache_root: Path | None = None,
    start_line_order: int = _DEFAULT_START_LINE_ORDER,
    end_line_order: int = _DEFAULT_END_LINE_ORDER,
) -> dict[str, object]:
    """Audit the cached Thunderbird slice without retraining a detector.

    Args:
        cache_root (Path | None): Optional override for the local AnomaLog
            cache root. Defaults to the user's cache directory.
        start_line_order (int): Inclusive raw line position for the slice.
        end_line_order (int): Inclusive raw line position for the slice.

    Returns:
        dict[str, object]: Compact JSON-serialisable audit summary.
    """
    resolved_cache_root = (
        CachePathsConfig().cache_root if cache_root is None else cache_root
    )
    sink = ParquetStructuredSink(
        dataset_name="THUNDERBIRD",
        raw_dataset_path=Path("/private/tmp/ignored-thunderbird.log"),
        parser=ThunderbirdParser(),
        cache_paths=CachePathsConfig(cache_root=resolved_cache_root),
    )
    dataset = sink._dataset()  # noqa: SLF001
    filter_expr = (ds.field("line_order") >= start_line_order) & (
        ds.field("line_order") <= end_line_order
    )
    materialised = dataset.to_table(
        columns=["line_order", "anomalous"],
        filter=filter_expr,
    )
    ordered = materialised.sort_by([("line_order", "ascending")])
    line_orders = ordered.column("line_order").to_pylist()
    raw_flags = [flag == 1 for flag in ordered.column("anomalous").to_pylist()]
    scan_flags = []
    for batch in dataset.scanner(
        columns=["anomalous"],
        filter=filter_expr,
        batch_size=1_000_000,
        use_threads=True,
    ).to_batches():
        scan_flags.extend(flag == 1 for flag in batch.column(0).to_pylist())

    raw_positions = expand_raw_position_flags(
        line_orders,
        raw_flags,
        start_line_order=start_line_order,
        end_line_order=end_line_order,
    )

    scan_summary = count_fixed_window_flags(
        scan_flags,
        ordering="scan",
    )
    source_summary = count_fixed_window_flags(
        raw_flags,
        ordering="source",
    )
    raw_summary = count_fixed_window_flags(
        raw_positions,
        ordering="raw_positions",
    )
    matches = find_matching_offsets(
        raw_positions,
        target_train_anomalous=837,
        target_test_anomalous=29,
    )

    return {
        "cache_root": str(resolved_cache_root),
        "slice": {
            "start_line_order": start_line_order,
            "end_line_order": end_line_order,
            "materialised_rows": len(raw_flags),
            "raw_anomalous_event_count": int(sum(raw_flags)),
            "line_order_min": line_orders[0] if line_orders else None,
            "line_order_max": line_orders[-1] if line_orders else None,
        },
        "contracts": {
            "scan_order_compacted": scan_summary.as_dict(),
            "source_order_compacted": source_summary.as_dict(),
            "raw_positions_compacted": raw_summary.as_dict(),
        },
        "matching_offsets_for_837_29": [summary.as_dict() for summary in matches],
        "notes": {
            "current_results_final_sequence_count": 99_996,
            "current_results_final_train_anomalous": 376,
            "current_results_final_test_anomalous": 51,
        },
    }


def audit_thunderbird_slice_json(
    *,
    cache_root: Path | None = None,
    start_line_order: int = _DEFAULT_START_LINE_ORDER,
    end_line_order: int = _DEFAULT_END_LINE_ORDER,
) -> str:
    """Return the Thunderbird slice audit as a compact JSON string."""
    return json.dumps(
        audit_thunderbird_slice(
            cache_root=cache_root,
            start_line_order=start_line_order,
            end_line_order=end_line_order,
        ),
        sort_keys=True,
    )
