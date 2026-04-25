"""Dataset sources backed by local directories or zip files."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from anomalog.io_utils import extract_zip, verify_md5
from anomalog.sources.contracts import DatasetSource


@dataclass(frozen=True)
class LocalDirSource(DatasetSource):
    """Use an existing local directory as the dataset source.

    Attributes:
        name (ClassVar[str]): Registry/config name for the source.
        path (Path): Existing directory treated as the dataset root.
        raw_logs_relpath (Path | None): Optional raw-log path relative to `path`.
    """

    name: ClassVar[str] = "local_dir"
    path: Path
    raw_logs_relpath: Path | None = None

    def materialise(
        self,
        *,
        dst_dir: Path,
    ) -> Path:
        """Validate directory existence and return the dataset root.

        Args:
            dst_dir (Path): Requested dataset destination. Ignored for local
                directory sources.

        Returns:
            Path: Existing dataset root directory.

        Raises:
            FileNotFoundError: If the configured path does not exist.
            NotADirectoryError: If the configured path is not a directory.
        """
        del dst_dir
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)
        return self.path


@dataclass(frozen=True)
class LocalZipSource(DatasetSource):
    """Use a local zip archive as the dataset source.

    Attributes:
        name (ClassVar[str]): Registry/config name for the source.
        zip_path (Path): Local zip archive to extract.
        raw_logs_relpath (Path | None): Optional raw-log path relative to the
            extracted dataset root.
        md5_checksum (str | None): Optional checksum used to verify the archive
            before extraction.
    """

    name: ClassVar[str] = "local_zip"
    zip_path: Path
    raw_logs_relpath: Path | None = None
    md5_checksum: str | None = None

    def materialise(
        self,
        *,
        dst_dir: Path,
    ) -> Path:
        """Extract the zip file into dst_dir and return the dataset root.

        Args:
            dst_dir (Path): Destination directory for extracted dataset files.

        Returns:
            Path: Extracted dataset root directory.

        Raises:
            FileNotFoundError: If the configured zip archive does not exist.
        """
        if dst_dir.exists() and dst_dir.is_dir() and any(dst_dir.iterdir()):
            return dst_dir

        if not self.zip_path.exists():
            raise FileNotFoundError(self.zip_path)

        if self.md5_checksum is not None:
            verify_md5(self.zip_path, self.md5_checksum)

        extract_zip(self.zip_path, dst_dir)
        return dst_dir
