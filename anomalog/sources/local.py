"""Dataset sources backed by local directories or zip files."""

from dataclasses import dataclass
from pathlib import Path

from anomalog.io_utils import extract_zip, verify_md5
from anomalog.sources.contracts import DatasetSource


@dataclass(frozen=True)
class LocalDirSource(DatasetSource):
    """Use an existing local directory as the dataset source."""

    name = "local_dir"
    path: Path
    raw_logs_relpath: Path | None = None

    def materialise(
        self,
        *,
        dst_dir: Path,
    ) -> Path:
        """Validate directory existence and return the dataset root."""
        del dst_dir
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)
        return self.path


@dataclass(frozen=True)
class LocalZipSource(DatasetSource):
    """Use a local zip archive as the dataset source."""

    name = "local_zip"
    zip_path: Path
    raw_logs_relpath: Path | None = None
    md5_checksum: str | None = None

    def materialise(
        self,
        *,
        dst_dir: Path,
    ) -> Path:
        """Extract the zip file into dst_dir and return the dataset root."""
        if dst_dir.exists() and dst_dir.is_dir() and any(dst_dir.iterdir()):
            return dst_dir

        if not self.zip_path.exists():
            raise FileNotFoundError(self.zip_path)

        if self.md5_checksum is not None:
            verify_md5(self.zip_path, self.md5_checksum)

        extract_zip(self.zip_path, dst_dir)
        return dst_dir
