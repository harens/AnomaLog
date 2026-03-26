"""Dataset sources backed by local directories or zip files."""

from dataclasses import dataclass
from pathlib import Path

from anomalog.io_utils import extract_zip, verify_md5
from anomalog.sources import DatasetSource


@dataclass(frozen=True)
class LocalDirSource(DatasetSource):
    """Use an existing local directory as the dataset source."""

    path: Path

    def materialise(self, dst_dir: Path) -> Path:  # noqa: ARG002 - not used, but part of the interface
        """Return the directory path after validating existence."""
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)
        return self.path


@dataclass(frozen=True)
class LocalZipSource(DatasetSource):
    """Use a local zip archive as the dataset source."""

    zip_path: Path
    md5_checksum: str | None = None

    def materialise(self, dst_dir: Path) -> Path:
        """Extract the zip file into dst_dir, verifying checksum when provided."""
        dst_dir.mkdir(parents=True, exist_ok=True)

        # fast path
        if dst_dir.exists() and dst_dir.is_dir() and any(dst_dir.iterdir()):
            return dst_dir

        if not self.zip_path.exists():
            raise FileNotFoundError(self.zip_path)

        if self.md5_checksum is not None:
            verify_md5(self.zip_path, self.md5_checksum)

        extract_zip(self.zip_path, dst_dir)
        return dst_dir
