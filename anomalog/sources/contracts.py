"""Base class and shared helpers for dataset materialisation sources."""

from abc import ABC, abstractmethod
from pathlib import Path


class DatasetSource(ABC):
    """Download or copy a dataset into the given directory."""

    raw_logs_relpath: Path | None = None

    @abstractmethod
    def materialise(
        self,
        *,
        dst_dir: Path,
    ) -> Path:
        """Ensure dataset exists under dst_dir and return the dataset root path."""

    def raw_logs_path(
        self,
        *,
        dataset_name: str,
        dataset_root: Path,
    ) -> Path:
        """Return the validated raw log path inside dataset_root."""
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
