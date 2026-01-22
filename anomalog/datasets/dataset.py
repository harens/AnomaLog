from dataclasses import dataclass
from pathlib import Path

from anomalog.datasets.sources import DatasetSource, RemoteZipSource


@dataclass(frozen=True)
class Dataset:
    name: str
    raw_logs_relpath: Path
    source: DatasetSource
    cache_root: Path = Path("data")

    @property
    def extracted_path(self) -> Path:
        return self.cache_root / self.name

    def ensure_ready(self) -> Path:
        if self.extracted_path.exists():
            return self.extracted_path

        return self.source.materialise(self.extracted_path)

    def raw_logs_path(self) -> Path:
        root = self.ensure_ready()
        return root / self.raw_logs_relpath


# # See LogHub: https://zenodo.org/records/8196385
# # Originally tried using LogHub-2.0 (https://zenodo.org/record/8275861),
# # but HDFS doesn't seem to be annotated
hdfs_v1 = Dataset(
    name="HDFS_V1",
    raw_logs_relpath=Path("HDFS.log"),
    source=RemoteZipSource(
        url="https://zenodo.org/records/8196385/files/HDFS_v1.zip",
        md5_checksum="76a24b4d9a6164d543fb275f89773260",
    ),
)

bgl = Dataset(
    name="BGL",
    raw_logs_relpath=Path("BGL.log"),
    source=RemoteZipSource(
        url="https://zenodo.org/records/8196385/files/BGL.zip",
        md5_checksum="4452953c470f2d95fcb32d5f6e733f7a",
    ),
)
