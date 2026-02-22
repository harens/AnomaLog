import heapq
from collections import deque
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pyarrow as pa
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
        use_threads: bool = True,
    ) -> Callable[[], Iterator[StructuredLine]]:
        """Iterate over StructuredLine objects with optional column projection."""

        def _iter() -> Iterator[StructuredLine]:
            col_list = self._projected_columns(columns)
            scanner = self._dataset().scanner(
                columns=col_list,
                batch_size=batch_size or self._DEFAULT_BATCH_SIZE,
                filter=filter_expr,
                batch_readahead=2,
                fragment_readahead=1,
                use_threads=use_threads,
            )
            for batch in scanner.to_batches():
                yield from self._rows_from_batch(batch)

        return _iter

    def _rows_from_batch(self, batch: pa.RecordBatch) -> Iterator[StructuredLine]:
        """Yield StructuredLine rows without materializing whole batches as dicts."""

        def column_or_default(name: str) -> pa.Array | None:
            idx = batch.schema.get_field_index(name)
            return batch.column(idx) if idx != -1 else None

        line_col = column_or_default(LINE_FIELD)
        ts_col = column_or_default(TIMESTAMP_FIELD)
        entity_col = column_or_default(ENTITY_FIELD)
        msg_col = column_or_default(UNTEMPLATED_FIELD)
        anomalous_col = column_or_default(ANOMALOUS_FIELD)

        def value_at_int(arr: pa.Array | None, i: int) -> int | None:
            if arr is None:
                return None
            scalar = arr[i]
            if not scalar.is_valid:
                return None
            return scalar.as_py()

        def value_at_str(
            arr: pa.Array | None,
            i: int,
            *,
            default: str | None,
        ) -> str | None:
            if arr is None:
                return default
            scalar = arr[i]
            if not scalar.is_valid:
                return default
            return scalar.as_py()

        for i in range(batch.num_rows):
            yield StructuredLine(
                line_order=value_at_int(line_col, i) or 0,
                timestamp_unix_ms=value_at_int(ts_col, i),
                entity_id=value_at_str(entity_col, i, default=None),
                untemplated_message_text=value_at_str(msg_col, i, default="") or "",
                anomalous=value_at_int(anomalous_col, i),
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
            batch_readahead=2,
            fragment_readahead=1,
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

    @staticmethod
    def _stream_time_windows(
        rows: Iterator[StructuredLine],
        *,
        first_ts: int,
        last_ts: int,
        time_span_ms: int,
        step: int,
    ) -> Iterator[list[StructuredLine]]:
        buffer: deque[StructuredLine] = deque()
        window_start = first_ts
        window_end = window_start + time_span_ms

        for row in rows:
            ts = row.timestamp_unix_ms
            if ts is None:
                continue

            while ts >= window_end:
                if buffer:
                    yield buffer

                window_start += step
                window_end = window_start + time_span_ms
                while buffer and (buffer[0].timestamp_unix_ms or 0) < window_start:
                    buffer.popleft()

            buffer.append(row)

        while window_start <= last_ts:
            if buffer:
                yield buffer
            window_start += step
            window_end = window_start + time_span_ms
            while buffer and (buffer[0].timestamp_unix_ms or 0) < window_start:
                buffer.popleft()

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

            yield from self._stream_time_windows(
                self._iter_structured_lines_ordered(
                    key_field=TIMESTAMP_FIELD,
                )(),
                first_ts=first_ts,
                last_ts=last_ts,
                time_span_ms=time_span_ms,
                step=step,
            )

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
            for row in self._iter_structured_lines_ordered(
                key_field=LINE_FIELD,
            )():
                buffer.append(row)
                if len(buffer) == window_size:
                    yield buffer

                    for _ in range(step):
                        if not buffer:
                            break
                        buffer.popleft()

            if buffer:
                yield list(buffer)

        return _iter

    def _iter_structured_lines_ordered(
        self,
        *,
        key_field: str = LINE_FIELD,
    ) -> Callable[[], Iterator[StructuredLine]]:
        """Merge bucketed parquet fragments into a globally ordered stream."""

        def _iter() -> Iterator[StructuredLine]:
            buckets = sorted(self._iter_buckets())
            buckets_with_null = [*buckets, None]

            def _key_for_row(row: StructuredLine) -> tuple[int, int, int]:
                if key_field == TIMESTAMP_FIELD:
                    ts = row.timestamp_unix_ms
                    # Missing timestamps go after those with timestamps; line_order
                    # acts as a deterministic tiebreaker.
                    return (1 if ts is None else 0, int(ts or 0), row.line_order)
                return (0, row.line_order, 0)

            bucket_iters: list[Iterator[StructuredLine]] = []
            for bucket in buckets_with_null:
                if bucket is None:
                    expr = ds.field(ENTITY_BUCKET_FIELD).is_null()
                else:
                    expr = ds.field(ENTITY_BUCKET_FIELD) == bucket

                bucket_iters.append(
                    self.iter_structured_lines(
                        filter_expr=expr,
                        batch_size=self._DEFAULT_BATCH_SIZE,
                        use_threads=False,
                    )(),
                )

            heap: list[
                tuple[
                    tuple[int, int, int],
                    int,
                    StructuredLine,
                    Iterator[StructuredLine],
                ]
            ] = []

            for idx, it in enumerate(bucket_iters):
                try:
                    first = next(it)
                except StopIteration:
                    continue
                heap.append((_key_for_row(first), idx, first, it))

            heapq.heapify(heap)

            while heap:
                _, idx, row, it = heapq.heappop(heap)
                yield row
                try:
                    nxt = next(it)
                except StopIteration:
                    continue
                heapq.heappush(heap, (_key_for_row(nxt), idx, nxt, it))

        return _iter
