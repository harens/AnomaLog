"""Base class and shared helpers for dataset materialisation sources."""

from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable


@runtime_checkable
class DatasetSource(Protocol):
    """Download or copy a dataset into the given directory.

    Source implementations materialise a dataset root and then rely on the
    shared `raw_logs_path()` helper to validate the final raw-log location.

    Attributes:
        name (ClassVar[str]): Stable registry/config name for the source.
        raw_logs_relpath (Path | None): Optional raw-log path relative to the
            materialised dataset root. When omitted, `<dataset_name>.log` is used.
    """

    name: ClassVar[str]
    raw_logs_relpath: Path | None = None

    def materialise(
        self,
        *,
        dst_dir: Path,
    ) -> Path:
        """Ensure dataset exists under `dst_dir` and return the dataset root path.

        Args:
            dst_dir (Path): Target directory where the dataset should appear.

        Returns:
            Path: Materialised dataset root directory.
        """

    def raw_logs_path(
        self,
        *,
        dataset_name: str,
        dataset_root: Path,
    ) -> Path:
        """Return the validated raw log path inside dataset_root.

        Args:
            dataset_name (str): Dataset name used for the default log filename.
            dataset_root (Path): Materialised dataset root directory.

        Returns:
            Path: Validated path to the raw log file inside the dataset root.

        Raises:
            ValueError: If `raw_logs_relpath` is absolute or escapes the dataset root.
            FileNotFoundError: If the resolved log path does not exist.
            IsADirectoryError: If the resolved path is not a file.
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

        if not candidate.exists():
            raise FileNotFoundError(candidate)
        if not candidate.is_file():
            raise IsADirectoryError(candidate)
        return candidate
