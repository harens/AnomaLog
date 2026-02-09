from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pyarrow.dataset as ds

from anomalog.cache import (
    CachePathsConfig,
    asset_from_local_path,
    materialize,
    task,
)
from anomalog.structured_parsers.contracts import (
    StructuredParser,
    StructuredSink,
)
from anomalog.structured_parsers.parquet.writer_worker import (
    extract_structured_components,
)


@dataclass(frozen=True, slots=True)
class ParquetStructuredSink(StructuredSink):
    dataset_name: str
    raw_dataset_path: Path
    parser: StructuredParser

    cache_paths: CachePathsConfig
    cache_dir: ClassVar[str] = "structured_parquet"

    def structured_data_cache(self, dataset_name: str) -> Path:
        return self.cache_paths.cache_root / dataset_name / self.cache_dir

    def write_structured_lines(self, workers: int | None = None) -> None:
        base_extract_structured_components = materialize(
            asset_from_local_path(self.structured_data_cache(self.dataset_name)),
            asset_deps=[asset_from_local_path(self.raw_dataset_path)],
        )(extract_structured_components)

        base_extract_structured_components(
            raw_input_path=self.raw_dataset_path,
            parser=self.parser,
            parquet_out_dir=self.structured_data_cache(self.dataset_name),
            workers=workers,
        )

    def read_unstructured_free_text(self) -> Callable[[], Iterator[str]]:
        read_task = task(self._read_unstructured_free_text).with_options(
            asset_deps=[
                asset_from_local_path(self.structured_data_cache(self.dataset_name)),
            ],
        )
        return read_task()

    def _read_unstructured_free_text(
        self,
        *,
        batch_size: int = 65_536,
    ) -> Callable[[], Iterator[str]]:
        def _iter() -> Iterator[str]:
            dataset = ds.dataset(
                self.structured_data_cache(self.dataset_name),
                format="parquet",
            )

            scanner = dataset.scanner(
                columns=["untemplated_message_text"],
                batch_size=batch_size,
            )

            for batch in scanner.to_batches():
                arr = batch.column(0)
                for s in arr.to_pylist():
                    if s is not None:
                        yield s

        return _iter
