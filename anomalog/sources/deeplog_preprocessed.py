"""Generic post-processed dataset sources built on top of archive downloads."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, TextIO

from anomalog.sources.contracts import DatasetSource

SplitFileSpec = tuple[str, int]
SplitFileSpecs = tuple[SplitFileSpec, ...]
LabelledRawSplitFileSpec = tuple[str, str, int]
LabelledRawSplitFileSpecs = tuple[LabelledRawSplitFileSpec, ...]
PostProcessFn = Callable[[Path, Path], None]

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PostProcessedSource(DatasetSource):
    """Materialise a base source and derive a raw log file from it.

    Attributes:
        name (ClassVar[str]): Registry/config name for the derived source.
        base_source (DatasetSource): Upstream source that materialises the
            archive or directory containing the source files.
        post_process (PostProcessFn): Function that derives the raw log file
            from the materialised base source root.
        raw_logs_relpath (Path | None): Relative path of the derived raw log
            file inside the materialised dataset root.
    """

    name: ClassVar[str] = "post_processed"
    base_source: DatasetSource
    post_process: PostProcessFn
    raw_logs_relpath: Path | None = None

    def materialise(
        self,
        *,
        dst_dir: Path,
    ) -> Path:
        """Materialise the base source and derive the raw log file.

        Args:
            dst_dir (Path): Destination directory for the materialised dataset.

        Returns:
            Path: Dataset root containing the derived raw log file.

        Raises:
            FileNotFoundError: If the post-processing step fails to create the
                derived raw log file.
        """
        dataset_root = self.base_source.materialise(dst_dir=dst_dir)
        raw_logs_path = self._derived_raw_logs_path(
            dataset_name=dst_dir.name,
            dataset_root=dataset_root,
        )
        raw_logs_path.parent.mkdir(parents=True, exist_ok=True)
        self.post_process(dataset_root, raw_logs_path)
        if not raw_logs_path.exists():
            raise FileNotFoundError(raw_logs_path)
        return dataset_root

    def _derived_raw_logs_path(self, *, dataset_name: str, dataset_root: Path) -> Path:
        """Resolve the output raw-log path without requiring it to exist yet.

        Args:
            dataset_name (str): Dataset name used when no explicit raw-log path
                is configured.
            dataset_root (Path): Materialised dataset root directory.

        Returns:
            Path: Candidate raw-log path inside the dataset root.

        Raises:
            ValueError: If `raw_logs_relpath` is absolute or escapes the
                dataset root.
        """
        if self.raw_logs_relpath is None:
            candidate = dataset_root / f"{dataset_name}.log"
        else:
            if self.raw_logs_relpath.is_absolute():
                msg = "raw_logs_relpath must be relative to the dataset root."
                raise ValueError(msg)
            candidate = dataset_root / self.raw_logs_relpath

        resolved_root = dataset_root.resolve()
        resolved_candidate = candidate.resolve(strict=False)
        try:
            resolved_candidate.relative_to(resolved_root)
        except ValueError as exc:
            msg = "raw_logs_relpath must stay within the dataset root."
            raise ValueError(msg) from exc

        return candidate


def _find_source_file(dataset_root: Path, split_name: str) -> Path | None:
    for candidate in dataset_root.rglob(split_name):
        if candidate.is_file():
            return candidate
    return None


def _append_split(
    *,
    split_path: Path,
    label: int,
    split_name: str,
    output: TextIO,
) -> tuple[int, int]:
    """Append one split file to the synthetic event stream.

    Args:
        split_path (Path): Source file containing one preprocessed session per
            line.
        label (int): Anomaly label to apply to every event in the split.
        split_name (str): Stable split prefix used to derive session ids.
        output (TextIO): Open synthetic event stream written by the source.

    Returns:
        tuple[int, int]: Session and event counts written for the split. The
            original session tokens are preserved verbatim in the output.
    """
    sessions_written = 0
    events_written = 0
    with split_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            session = raw_line.strip()
            if not session:
                continue
            session_id = f"{split_name}:{sessions_written}"
            sessions_written += 1
            for token in session.split():
                output.write(f"{session_id}\t{label}\t{token}\n")
                events_written += 1
    return sessions_written, events_written


def materialise_labelled_session_stream(
    source_root: Path,
    raw_logs_path: Path,
    split_files: SplitFileSpecs,
) -> None:
    """Expand labelled session files into one event per line.

    Args:
        source_root (Path): Root containing the split files to read.
        raw_logs_path (Path): Destination path for the synthetic event stream.
        split_files (SplitFileSpecs): Session-file names plus their anomaly
            labels in the order they should be written.

    Raises:
        FileNotFoundError: If any expected split file is missing from the
            source root.
    """
    event_count = 0
    session_count = 0

    with raw_logs_path.open("w", encoding="utf-8") as output:
        for split_name, label in split_files:
            split_path = _find_source_file(source_root, split_name)
            if split_path is None:
                msg = f"Missing {split_name} in extracted archive at {source_root}."
                raise FileNotFoundError(msg)
            sessions_written, events_written = _append_split(
                split_path=split_path,
                label=label,
                split_name=split_name,
                output=output,
            )
            session_count += sessions_written
            event_count += events_written

    _LOGGER.info(
        "Wrote %s sessions and %s events to %s",
        session_count,
        event_count,
        raw_logs_path,
    )


def materialise_labelled_raw_stream(
    source_root: Path,
    raw_logs_path: Path,
    split_files: LabelledRawSplitFileSpecs,
) -> None:
    r"""Concatenate raw split files into a labelled stream.

    Each emitted row keeps the original raw line while adding a stable split
    name and anomaly label:
    `<split_name>\\t<label>\\t<raw_line>`.

    Args:
        source_root (Path): Root containing the split source files.
        raw_logs_path (Path): Destination path for the synthetic raw stream.
        split_files (LabelledRawSplitFileSpecs): Source filename, output split
            name, and anomaly label in output order.

    Raises:
        FileNotFoundError: If any expected split file is missing.
    """
    row_count = 0
    with raw_logs_path.open("w", encoding="utf-8") as output:
        for source_name, split_name, label in split_files:
            split_path = _find_source_file(source_root, source_name)
            if split_path is None:
                msg = f"Missing {source_name} in extracted archive at {source_root}."
                raise FileNotFoundError(msg)
            with split_path.open(encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.rstrip("\n")
                    if not line:
                        continue
                    output.write(f"{split_name}\t{label}\t{line}\n")
                    row_count += 1
    _LOGGER.info("Wrote %s labelled raw rows to %s", row_count, raw_logs_path)
