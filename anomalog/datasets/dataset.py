from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from anomalog.datasets.sources import DatasetSource, RemoteZipSource

# from anomalog.datasets.parsers.base import Parser, ParsedDataset


@dataclass(frozen=True, slots=True)
class ProcessedLine:
    text: str
    is_anomaly: bool | None = None  # None if dataset doesn’t provide labels
    # meta: dict[str, Any] | None = None  # Additional metadata if needed


PreprocessFn = Callable[[str], ProcessedLine]


@dataclass(frozen=True)
class Dataset:
    name: str
    raw_logs_relpath: Path
    source: DatasetSource
    cache_root: Path = Path("data")
    preprocess: PreprocessFn = lambda line: ProcessedLine(text=line)

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

    def iter_lines(self) -> Iterator[str]:
        with open(self.raw_logs_path(), encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line.rstrip("\n")

    # def parse_with(self, parser: Parser) -> ParsedDataset:
    #     return parser.parse(self.read_csv_rows())


# See https://github.com/logpai/loghub/issues/61
# Datasets could have mistakes in labeling

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
    # TODO: Figure out what to do with this
    # Note from https://github.com/logpai/loghub/blob/dd61d0952749ee7963bde24220d1be5ede023033/BGL/README.md:
    # In the first column of the log, "-" indicates non-alert messages
    # while others are alert messages.
    preprocess=lambda line: ProcessedLine(
        text=line[2:] if line.startswith("- ") else line,
        is_anomaly=not line.startswith("- "),
    ),
)
