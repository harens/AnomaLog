from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from prefect.logging import get_run_logger

from anomalog.cache import CachePathsConfig
from anomalog.sources import DatasetSource
from anomalog.structured_parsers.contracts import StructuredParser, StructuredSink
from anomalog.structured_parsers.parquet.sink import ParquetStructuredSink
from anomalog.structured_parsers.structured_dataset import StructuredDataset


@dataclass(slots=True, frozen=True)
class RawDataset:
    dataset_name: str
    source: DatasetSource
    structured_parser: StructuredParser
    cache_paths: CachePathsConfig = field(default_factory=CachePathsConfig)
    raw_logs_relpath: Path | None = None

    @property
    def raw_logs_path(self) -> Path:
        dir_name = self.cache_paths.data_root / self.dataset_name
        if self.raw_logs_relpath is None:
            return dir_name / f"{self.dataset_name}.log"
        return dir_name / self.raw_logs_relpath

    def iter_lines(self) -> Iterator[str]:
        with Path.open(self.raw_logs_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line.rstrip("\n")

    def fetch_if_needed(self) -> "RawDataset":
        logger = get_run_logger()
        logger.info(
            "Fetching dataset %s to %s",
            self.dataset_name,
            self.raw_logs_path.parent,
        )
        self.source.materialise(self.raw_logs_path.parent)
        return self

    def extract_structured_components(
        self,
        sink: StructuredSink | None = None,
    ) -> StructuredDataset:
        if sink is None:
            sink = ParquetStructuredSink(
                dataset_name=self.dataset_name,
                raw_dataset_path=self.raw_logs_path,
                parser=self.structured_parser,
                cache_paths=self.cache_paths,
            )

        sink.write_structured_lines()

        return StructuredDataset(
            sink,
            cache_paths=self.cache_paths,
        )
