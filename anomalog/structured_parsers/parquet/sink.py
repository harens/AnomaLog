from collections.abc import Callable, Iterator, Sequence
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
    ANOMALOUS_FIELD,
    ENTITY_FIELD,
    LINE_FIELD,
    UNTEMPLATED_FIELD,
    StructuredLine,
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

    def write_structured_lines(self, workers: int | None = None) -> bool:
        base_extract_structured_components = materialize(
            asset_from_local_path(self.structured_data_cache(self.dataset_name)),
            asset_deps=[asset_from_local_path(self.raw_dataset_path)],
        )(extract_structured_components)

        return base_extract_structured_components(
            raw_input_path=self.raw_dataset_path,
            parser=self.parser,
            parquet_out_dir=self.structured_data_cache(self.dataset_name),
            workers=workers,
        )

    def read_unstructured_free_text(self) -> Callable[[], Iterator[str]]:
        def _iter() -> Iterator[str]:
            for row in self.iter_structured_lines(columns=[UNTEMPLATED_FIELD])():
                text = getattr(row, UNTEMPLATED_FIELD, None)
                if text is not None:
                    yield text

        return task(
            _iter,
            name=f"read-unstructured:{UNTEMPLATED_FIELD}",
        ).with_options(
            asset_deps=[
                asset_from_local_path(self.structured_data_cache(self.dataset_name)),
            ],
        )

    def iter_structured_lines(
        self,
        columns: Sequence[str] | None = None,
    ) -> Callable[[], Iterator[StructuredLine]]:
        """Iterate over StructuredLine objects with optional column projection."""

        # Ensure required fields are present; missing columns default to None.
        required = set(StructuredLine.__dataclass_fields__.keys())
        col_list = list(required if columns is None else set(columns) | required)

        def _iter() -> Iterator[StructuredLine]:
            dataset = ds.dataset(
                self.structured_data_cache(self.dataset_name),
                format="parquet",
            )

            scanner = dataset.scanner(columns=col_list, batch_size=65_536)

            for batch in scanner.to_batches():
                table = batch.to_pydict()
                n_rows = len(next(iter(table.values()))) if table else 0
                for i in range(n_rows):
                    kwargs = {col: table.get(col, [None])[i] for col in col_list}
                    yield StructuredLine(**kwargs)

        col_suffix = ",".join(sorted(col_list))
        return task(
            _iter,
            name=f"iter-structured-lines:{col_suffix}",
        ).with_options(
            asset_deps=[
                asset_from_local_path(self.structured_data_cache(self.dataset_name)),
            ],
        )

    def label_for_line(self, line_order: int) -> int | None:
        return self._scan_label(LINE_FIELD, line_order)

    def label_for_group(self, entity_id: str) -> int | None:
        return self._scan_label(ENTITY_FIELD, entity_id)

    def _scan_label(self, key_field: str, key_value: object) -> int | None:
        for row in self.iter_structured_lines(
            columns=[key_field, ANOMALOUS_FIELD],
        )():
            if getattr(row, key_field) == key_value:
                label = getattr(row, ANOMALOUS_FIELD)
                try:
                    return int(label) if label is not None else None
                except (TypeError, ValueError):
                    return None
        return None
