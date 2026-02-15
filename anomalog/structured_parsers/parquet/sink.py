import math
from collections import defaultdict, deque
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pyarrow.dataset as ds

from anomalog.cache import CachePathsConfig, asset_from_local_path, materialize
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
    ENTITY_BUCKET_FIELD,
    WriterConfig,
    extract_structured_components,
)


@dataclass(frozen=True, slots=True)
class ParquetStructuredSink(StructuredSink):
    _DEFAULT_BATCH_SIZE: ClassVar[int] = 65_536
    dataset_name: str
    raw_dataset_path: Path
    parser: StructuredParser
    cache_paths: CachePathsConfig

    cache_dir: ClassVar[str] = "structured_parquet"

    def structured_data_cache(self, dataset_name: str) -> Path:
        return self.cache_paths.cache_root / dataset_name / self.cache_dir

    def write_structured_lines(self, _workers: int | None = None) -> bool:
        base_extract_structured_components = materialize(
            asset_from_local_path(self.structured_data_cache(self.dataset_name)),
            asset_deps=[asset_from_local_path(self.raw_dataset_path)],
        )(extract_structured_components)

        return base_extract_structured_components(
            raw_input_path=self.raw_dataset_path,
            parser=self.parser,
            parquet_out_dir=self.structured_data_cache(self.dataset_name),
            config=WriterConfig(),
        )

    def iter_structured_lines(
        self,
        columns: Sequence[str] | None = None,
        *,
        filter_expr: ds.Expression | None = None,
        batch_size: int | None = None,
    ) -> Callable[[], Iterator[StructuredLine]]:
        """Iterate over StructuredLine objects with optional column projection."""

        def _iter() -> Iterator[StructuredLine]:
            col_list = self._projected_columns(columns)
            scanner = self._dataset().scanner(
                columns=col_list,
                batch_size=batch_size or self._DEFAULT_BATCH_SIZE,
                filter=filter_expr,
            )
            for batch in scanner.to_batches():
                yield from self._rows_from_batch(batch.to_pydict())

        return _iter

    def _rows_from_batch(self, table_dict: dict[str, list]) -> Iterator[StructuredLine]:
        n = len(next(iter(table_dict.values()), []))

        defaults = {
            LINE_FIELD: None,
            TIMESTAMP_FIELD: None,
            ENTITY_FIELD: None,
            UNTEMPLATED_FIELD: "",
            ANOMALOUS_FIELD: None,
        }
        columns = {
            name: table_dict.get(name, [default] * n)
            for name, default in defaults.items()
        }

        for i in range(n):
            yield StructuredLine(
                line_order=columns[LINE_FIELD][i],
                timestamp_unix_ms=columns[TIMESTAMP_FIELD][i],
                entity_id=columns[ENTITY_FIELD][i],
                untemplated_message_text=columns[UNTEMPLATED_FIELD][i],
                anomalous=columns[ANOMALOUS_FIELD][i],
            )

    def _dataset(self) -> ds.Dataset:
        return ds.dataset(
            self.structured_data_cache(self.dataset_name),
            format="parquet",
            partitioning="hive",
        )

    def _projected_columns(
        self,
        columns: Sequence[str] | None,
    ) -> list[str]:
        if columns is not None:
            # Preserve caller order, drop duplicates.
            return list(dict.fromkeys(columns))
        return list(self._structured_columns())

    @staticmethod
    def _structured_columns() -> tuple[str, ...]:
        return tuple(StructuredLine.__dataclass_fields__.keys())

    @staticmethod
    def _row_sort_key(row: StructuredLine) -> tuple[int, int]:
        return (row.timestamp_unix_ms or 0, row.line_order or 0)

    def iter_entity_sequences(
        self,
    ) -> Callable[[], Iterator[Collection[StructuredLine]]]:
        def _iter() -> Iterator[Collection[StructuredLine]]:
            for bucket_id in sorted(self._iter_buckets()):
                by_entity: dict[str, list[StructuredLine]] = {}
                for row in self.iter_structured_lines(
                    filter_expr=(ds.field(ENTITY_BUCKET_FIELD) == bucket_id),
                )():
                    if row.entity_id is None:
                        continue
                    by_entity.setdefault(row.entity_id, []).append(row)

                for _, rows in sorted(by_entity.items()):
                    rows.sort(key=self._row_sort_key)
                    yield rows

        return _iter

    def _iter_buckets(self) -> set[int]:
        buckets: set[int] = set()
        scanner = self._dataset().scanner(
            columns=[ENTITY_BUCKET_FIELD],
            batch_size=self._DEFAULT_BATCH_SIZE,
        )
        for batch in scanner.to_batches():
            for bucket in batch.column(0).to_pylist():
                if bucket is not None:
                    buckets.add(bucket)
        return buckets

    def _timestamp_bounds(self) -> tuple[int | None, int | None]:
        ts_scanner = self._dataset().scanner(
            columns=[TIMESTAMP_FIELD],
            batch_size=self._DEFAULT_BATCH_SIZE,
        )

        min_ts: int | None = None
        max_ts: int | None = None

        for batch in ts_scanner.to_batches():
            col = batch.column(0)
            if len(col) == 0:
                continue

            batch_min = col.min()
            batch_max = col.max()

            if batch_min is not None:
                min_ts = batch_min if min_ts is None else min(min_ts, batch_min)
            if batch_max is not None:
                max_ts = batch_max if max_ts is None else max(max_ts, batch_max)

        return min_ts, max_ts

    def iter_time_window_sequences(
        self,
        time_span_ms: int,
        step_span_ms: int | None = None,
    ) -> Callable[[], Iterator[Collection[StructuredLine]]]:
        def _iter() -> Iterator[Collection[StructuredLine]]:
            first_ts, last_ts = self._timestamp_bounds()
            if first_ts is None or last_ts is None:
                msg = "No timestamps available for time-based windowing."
                raise ValueError(msg)

            step = step_span_ms or time_span_ms
            if time_span_ms <= 0 or step <= 0:
                msg = "time_span_ms and step_span_ms must be positive integers"
                raise ValueError(msg)

            windowed_rows: dict[int, list[StructuredLine]] = defaultdict(list)
            scanner = self._dataset().scanner(batch_size=self._DEFAULT_BATCH_SIZE)

            for batch in scanner.to_batches():
                for row in self._rows_from_batch(batch.to_pydict()):
                    ts = row.timestamp_unix_ms
                    if ts is None:
                        continue

                    max_idx = (ts - first_ts) // step
                    min_idx = max(
                        0,
                        math.ceil((ts - first_ts - time_span_ms + 1) / step),
                    )

                    for idx in range(min_idx, max_idx + 1):
                        windowed_rows[idx].append(row)

            for idx in sorted(windowed_rows.keys()):
                bucket = windowed_rows[idx]
                if not bucket:
                    continue
                bucket.sort(key=self._row_sort_key)
                yield bucket

        return _iter

    def iter_fixed_window_sequences(
        self,
        window_size: int,
        step_size: int | None = None,
    ) -> Callable[[], Iterator[Collection[StructuredLine]]]:
        def _iter() -> Iterator[Collection[StructuredLine]]:
            step = step_size or window_size
            if window_size <= 0 or step <= 0:
                msg = "window_size and step_size must be positive integers"
                raise ValueError(msg)

            buffer: deque[StructuredLine] = deque()
            for row in self.iter_structured_lines(
                batch_size=self._DEFAULT_BATCH_SIZE,
            )():
                buffer.append(row)
                if len(buffer) == window_size:
                    yield buffer

                    for _ in range(step):
                        if not buffer:
                            break
                        buffer.popleft()

            if buffer:
                yield buffer

        return _iter
