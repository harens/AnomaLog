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
    TIMESTAMP_FIELD,
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
    _SEQ_COLUMNS: ClassVar[list[str]] = [
        LINE_FIELD,
        TIMESTAMP_FIELD,
        ENTITY_FIELD,
        UNTEMPLATED_FIELD,
        ANOMALOUS_FIELD,
    ]
    dataset_name: str
    raw_dataset_path: Path
    parser: StructuredParser

    cache_paths: CachePathsConfig
    cache_dir: ClassVar[str] = "structured_parquet"

    def structured_data_cache(self, dataset_name: str) -> Path:
        return self.cache_paths.cache_root / dataset_name / self.cache_dir

    def _iter_rows(self, *, batch_size: int) -> Iterator[StructuredLine]:
        scanner = self._dataset().scanner(
            columns=self._SEQ_COLUMNS,
            batch_size=batch_size,
        )
        for batch in scanner.to_batches():
            yield from self._rows_from_batch(batch.to_pydict())

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

    def _rows_from_batch(self, table_dict: dict[str, list]) -> Iterator[StructuredLine]:
        n = len(next(iter(table_dict.values()))) if table_dict else 0
        for i in range(n):
            yield StructuredLine(
                line_order=table_dict.get(LINE_FIELD, [None])[i],
                timestamp_unix_ms=table_dict.get(TIMESTAMP_FIELD, [None])[i],
                entity_id=table_dict.get(ENTITY_FIELD, [None])[i],
                untemplated_message_text=table_dict.get(UNTEMPLATED_FIELD, [""])[i],
                anomalous=table_dict.get(ANOMALOUS_FIELD, [None])[i],
            )

    def _dataset(self) -> ds.Dataset:
        return ds.dataset(
            self.structured_data_cache(self.dataset_name),
            format="parquet",
        )

    def iter_entity_sequences(self) -> Callable[[], Iterator[list[StructuredLine]]]:
        def _iter() -> Iterator[list[StructuredLine]]:
            current_ent = None
            bucket: list[StructuredLine] = []
            for row in self._iter_rows(batch_size=65_536):
                ent = row.entity_id
                if ent is None:
                    continue
                if current_ent is None:
                    current_ent = ent
                if ent != current_ent:
                    if bucket:
                        yield bucket
                    bucket = []
                    current_ent = ent
                bucket.append(row)
            if bucket:
                yield bucket

        return task(
            _iter,
            name="iter-sequence-lines:entity",
        ).with_options(
            asset_deps=[
                asset_from_local_path(self.structured_data_cache(self.dataset_name)),
            ],
        )

    def _min_timestamp(self) -> int | None:
        ts_scanner = self._dataset().scanner(columns=[TIMESTAMP_FIELD])
        first_ts = None
        for b in ts_scanner.to_batches():
            col = b.column(0)
            if len(col) == 0:
                continue
            batch_min = col.min()
            if batch_min is not None:
                first_ts = batch_min if first_ts is None else min(first_ts, batch_min)
        return first_ts

    def iter_time_window_sequences(
        self,
        time_span_ms: int,
        step_span_ms: int | None = None,
    ) -> Callable[[], Iterator[list[StructuredLine]]]:
        def _iter() -> Iterator[list[StructuredLine]]:
            first_ts = self._min_timestamp()
            if first_ts is None:
                msg = "No timestamps available for time-based windowing."
                raise ValueError(msg)

            step = step_span_ms or time_span_ms
            current_end = first_ts + time_span_ms
            bucket: list[StructuredLine] = []
            for row in self._iter_rows(batch_size=65_536):
                ts = row.timestamp_unix_ms
                if ts is None:
                    continue
                while ts >= current_end:
                    if bucket:
                        bucket.sort(
                            key=lambda r: (
                                r.timestamp_unix_ms or 0,
                                r.line_order or 0,
                            ),
                        )
                        yield bucket
                        bucket = []
                    current_end += step
                bucket.append(row)
            if bucket:
                bucket.sort(key=lambda r: (r.timestamp_unix_ms or 0, r.line_order or 0))
                yield bucket

        return task(
            _iter,
            name=f"iter-sequence-lines:time-{time_span_ms}",
        ).with_options(
            asset_deps=[
                asset_from_local_path(self.structured_data_cache(self.dataset_name)),
            ],
        )

    def iter_fixed_window_sequences(
        self,
        window_size: int,
        step_size: int | None = None,
    ) -> Callable[[], Iterator[list[StructuredLine]]]:
        def _iter() -> Iterator[list[StructuredLine]]:
            buffer: list[StructuredLine] = []
            step = step_size or window_size
            for row in self._iter_rows(batch_size=window_size):
                buffer.append(row)
                if len(buffer) == window_size:
                    yield buffer
                    buffer = buffer[step:] if step < window_size else []
            if buffer:
                yield buffer

        return task(
            _iter,
            name=(
                f"iter-sequence-lines:fixed-{window_size}"
                f"-step-{step_size or window_size}"
            ),
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
