"""Helpers for deriving bounded raw-log slices from extracted archives."""

from __future__ import annotations

import logging
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


def materialise_raw_log_prefix(
    source_root: Path,
    raw_logs_path: Path,
    *,
    source_log_relpath: Path = Path("Thunderbird.log"),
    line_limit: int,
) -> None:
    """Write the first ``line_limit`` lines from an explicit raw log file.

    Args:
        source_root (Path): Root directory containing the extracted archive.
        raw_logs_path (Path): Destination path for the bounded prefix.
        source_log_relpath (Path): Exact raw-log path relative to
            ``source_root``.
        line_limit (int): Maximum number of raw lines to retain.
    """
    materialise_raw_log_segment(
        source_root,
        raw_logs_path,
        source_log_relpath=source_log_relpath,
        start_line=1,
        line_limit=line_limit,
    )


def materialise_raw_log_segment(
    source_root: Path,
    raw_logs_path: Path,
    *,
    source_log_relpath: Path = Path("Thunderbird.log"),
    start_line: int,
    line_limit: int,
) -> None:
    """Write a bounded line range from an explicit raw log file.

    Args:
        source_root (Path): Root directory containing the extracted archive.
        raw_logs_path (Path): Destination path for the bounded slice.
        source_log_relpath (Path): Exact raw-log path relative to
            ``source_root``.
        start_line (int): 1-based line number at which to begin copying.
        line_limit (int): Maximum number of raw lines to retain.

    Raises:
        ValueError: If ``start_line`` or ``line_limit`` are not positive.
        FileNotFoundError: If ``source_log_relpath`` does not exist under
            ``source_root``.
        IsADirectoryError: If ``source_log_relpath`` resolves to a directory.
    """
    if start_line <= 0:
        msg = "start_line must be a positive integer."
        raise ValueError(msg)
    if line_limit <= 0:
        msg = "line_limit must be a positive integer."
        raise ValueError(msg)

    if source_log_relpath.is_absolute():
        msg = "source_log_relpath must be relative to the source root."
        raise ValueError(msg)

    source_log_path = source_root / source_log_relpath
    if not source_log_path.exists():
        raise FileNotFoundError(source_log_path)
    if not source_log_path.is_file():
        raise IsADirectoryError(source_log_path)

    raw_logs_path.parent.mkdir(parents=True, exist_ok=True)

    line_count = 0
    with (
        source_log_path.open(encoding="utf-8", errors="replace") as source,
        raw_logs_path.open(
            "w",
            encoding="utf-8",
        ) as target,
    ):
        for line_number, raw_line in enumerate(source, start=1):
            if line_number < start_line:
                continue
            if line_count >= line_limit:
                break
            target.write(raw_line)
            line_count += 1

    _LOGGER.info(
        "Wrote %s-line raw slice from %s starting at line %s to %s",
        line_count,
        source_log_path,
        start_line,
        raw_logs_path,
    )
