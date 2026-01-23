import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Protocol, Self, TypedDict, cast, runtime_checkable

from anomalog.datasets.sources import DatasetSource, RemoteZipSource

logger = logging.getLogger(__name__)

# from anomalog.datasets.parsers.base import Parser, ParsedDataset


# @dataclass(frozen=True, slots=True)
# class ProcessedLine:
#     text: str
#     is_anomaly: bool | None = None  # None if dataset doesn’t provide labels
#     # meta: dict[str, Any] | None = None  # Additional metadata if needed


# PreprocessFn = Callable[[str], ProcessedLine]


class DatasetBaseKwargs(TypedDict):
    name: str
    raw_logs_relpath: Path
    cache_root: Path


@dataclass(slots=True, kw_only=True)
class DatasetBase:
    name: str
    raw_logs_relpath: Path
    cache_root: Path = Path("data")
    # preprocess: PreprocessFn = lambda line: ProcessedLine(text=line)

    @property
    def extracted_path(self) -> Path:
        return self.cache_root / self.name

    @property
    def raw_logs_path(self) -> Path:
        root = self.extracted_path
        return root / self.raw_logs_relpath

    def iter_lines(self) -> Iterator[str]:
        with open(self.raw_logs_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line.rstrip("\n")

    def base_kwargs(self) -> DatasetBaseKwargs:
        result = {f.name: getattr(self, f.name) for f in fields(DatasetBase)}
        return cast(DatasetBaseKwargs, result)


@dataclass(slots=True)
class RawDataset(DatasetBase):
    source: DatasetSource

    def fetch_if_needed(self) -> Self:
        if self.extracted_path.exists():
            # TODO: This logger info is called twice when doing
            # parse_with then fetch_if_needed
            logger.info(f"Dataset {self.name} already fetched at {self.extracted_path}")
            return self

        logger.info(f"Fetching dataset {self.name} to {self.extracted_path}")
        self.source.materialise(self.extracted_path)
        return self

    def parse_with(self, parser: Parser) -> ParsedDataset:
        self.fetch_if_needed()
        return parser.parse(self)


@dataclass(slots=True)
class ParsedDataset(DatasetBase):
    get_template_and_params_for_log: Callable[[str], tuple[str, list[str]]]


@runtime_checkable
class Parser(Protocol):
    def parse(self, raw_dataset: RawDataset) -> ParsedDataset: ...


# See https://github.com/logpai/loghub/issues/61
# Datasets could have mistakes in labeling

# # See LogHub: https://zenodo.org/records/8196385
# # Originally tried using LogHub-2.0 (https://zenodo.org/record/8275861),
# # but HDFS doesn't seem to be annotated

hdfs_v1 = RawDataset(
    name="HDFS_V1",
    raw_logs_relpath=Path("HDFS.log"),
    source=RemoteZipSource(
        url="https://zenodo.org/records/8196385/files/HDFS_v1.zip",
        md5_checksum="76a24b4d9a6164d543fb275f89773260",
    ),
)
bgl = RawDataset(
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
    # preprocess=lambda line: ProcessedLine(
    #     text=line[2:] if line.startswith("- ") else line,
    #     is_anomaly=not line.startswith("- "),
    # ),
)
