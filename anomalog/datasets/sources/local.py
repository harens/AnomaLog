from dataclasses import dataclass
from pathlib import Path

from anomalog.datasets.io_utils import extract_zip, verify_md5
from anomalog.datasets.sources.base import DatasetSource
from anomalog.type_hints import MD5Hex


@dataclass(frozen=True)
class LocalDirSource(DatasetSource):
    path: Path

    def materialise(self, dst_dir: Path) -> Path:
        # TODO: dst_dir is ignored
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        if not self.path.is_dir():
            raise NotADirectoryError(self.path)
        return self.path


@dataclass(frozen=True)
class LocalZipSource(DatasetSource):
    zip_path: Path
    md5_checksum: MD5Hex | None = None

    def materialise(self, dst_dir: Path) -> Path:
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
