"""Parquet-backed implementation of StructuredSink."""

import heapq
import json
from collections import deque
from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from anomalog.cache import (
    CachePathsConfig,
    asset_from_local_path,
    materialize,
)
from anomalog.parsers.structured.contracts import (
    ANOMALOUS_FIELD,
    ENTITY_FIELD,
    LINE_FIELD,
    TIMESTAMP_FIELD,
    UNTEMPLATED_FIELD,
    EntityLabelCounts,
    StructuredLine,
    StructuredParser,
    StructuredSink,
    is_anomalous_label,
)
from anomalog.parsers.structured.parquet.writer_worker import (
    ENTITY_BUCKET_FIELD,
    ENTITY_CHRONOLOGY_INDEX_FILENAME,
    EntityChronologyKey,
    WriterConfig,
    extract_structured_components,
)


@dataclass(frozen=True, slots=True)
class _EntityGroup:
    """Entity-group rows paired with their chronological order key.

    Attributes:
        order (EntityChronologyKey): Deterministic sort key for the entity.
        rows (tuple[StructuredLine, ...]): Structured rows belonging to the
            entity group.
    """

    order: EntityChronologyKey
    rows: tuple[StructuredLine, ...]


@dataclass(frozen=True, slots=True)
class ParquetStructuredSink(StructuredSink):
    """StructuredSink backed by partitioned Parquet datasets.

    Provides efficient iteration, windowing helpers, and label-aware counts
    for downstream anomaly workflows.

    Attributes:
        dataset_name (str): Dataset identifier used to scope parquet cache paths.
        raw_dataset_path (Path): Materialsed raw log file parsed into parquet.
        parser (StructuredParser): Parser used to convert raw lines into
            structured records.
        cache_paths (CachePathsConfig): Data/cache roots used for parquet output.
        cache_dir (ClassVar[str]): Dataset-local cache directory name for
            structured parquet artifacts.
    """

    _DEFAULT_BATCH_SIZE: ClassVar[int] = 65_536
    dataset_name: str
    raw_dataset_path: Path
    parser: StructuredParser
    cache_paths: CachePathsConfig

    cache_dir: ClassVar[str] = "structured_parquet"

    def structured_data_cache(self, dataset_name: str) -> Path:
        """Return the cache directory for this dataset.

        Args:
            dataset_name (str): Dataset name whose parquet cache should be used.

        Returns:
            Path: Structured-parquet cache directory for the dataset.
        """
        return self.cache_paths.cache_root / dataset_name / self.cache_dir

    def write_structured_lines(self, _workers: int | None = None) -> bool:
        """Parse raw logs and persist structured lines to Parquet.

        Args:
            _workers (int | None): Reserved worker-count override. Currently
                unused by this sink implementation.

        Returns:
            bool: Whether any anomalous rows were observed during parsing.
        """
        base_extract_structured_components = materialize(
            self.structured_data_cache(self.dataset_name),
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
        """Iterate over StructuredLine objects with optional column projection.

        Args:
            columns (Sequence[str] | None): Optional projected column names to
                load from parquet.
            filter_expr (ds.Expression | None): Optional PyArrow dataset filter
                expression.
            batch_size (int | None): Optional scanner batch size override.

        Returns:
            Callable[[], Iterator[StructuredLine]]: Callable producing projected
                structured rows from the parquet dataset.
        """

        def _iter() -> Iterator[StructuredLine]:
            col_list = self._projected_columns(columns)
            scanner = self._dataset().scanner(
                columns=col_list,
                batch_size=batch_size or self._DEFAULT_BATCH_SIZE,
                filter=filter_expr,
                batch_readahead=2,
                fragment_readahead=1,
                use_threads=True,
            )
            for batch in scanner.to_batches():
                yield from self._rows_from_batch(batch)

        return _iter

    def iter_structured_lines_in_source_order(
        self,
    ) -> Callable[[], Iterator[StructuredLine]]:
        """Iterate over structured rows in raw-entry order.

        The parquet dataset is partitioned by entity buckets, so this merges the
        bucket-local scans by `line_order` to recover the original raw-entry
        chronology without materialising the entire dataset in memory.

        Returns:
            Callable[[], Iterator[StructuredLine]]: Zero-argument callable that
                yields structured rows ordered by `line_order`.
        """

        def _iter() -> Iterator[StructuredLine]:
            buckets = sorted(self._iter_buckets())
            buckets_with_null = [*buckets, None]
            bucket_iters: list[Iterator[StructuredLine]] = []
            for bucket in buckets_with_null:
                if bucket is None:
                    expr = ds.field(ENTITY_BUCKET_FIELD).is_null()
                else:
                    expr = ds.field(ENTITY_BUCKET_FIELD) == bucket
                bucket_iters.append(
                    self.iter_structured_lines(
                        batch_size=self._DEFAULT_BATCH_SIZE,
                        filter_expr=expr,
                    )(),
                )

            heap: list[tuple[int, int, StructuredLine, Iterator[StructuredLine]]] = []
            for idx, it in enumerate(bucket_iters):
                try:
                    first = next(it)
                except StopIteration:
                    continue
                heap.append((int(first.line_order), idx, first, it))

            heapq.heapify(heap)

            while heap:
                _, idx, row, it = heapq.heappop(heap)
                yield row
                try:
                    nxt = next(it)
                except StopIteration:
                    continue
                heapq.heappush(heap, (int(nxt.line_order), idx, nxt, it))

        return _iter

    def load_inline_label_cache(self) -> tuple[dict[int, int], dict[str, int]]:
        """Load sparse inline labels directly from parquet batches.

        Returns:
            tuple[dict[int, int], dict[str, int]]: Sparse per-line and per-group
                anomaly labels.
        """
        line_labels: dict[int, int] = {}
        group_labels: dict[str, int] = {}

        scanner = self._dataset().scanner(
            columns=[LINE_FIELD, ENTITY_FIELD, ANOMALOUS_FIELD],
            batch_size=self._DEFAULT_BATCH_SIZE,
            batch_readahead=2,
            fragment_readahead=1,
            use_threads=True,
        )

        for batch in scanner.to_batches():
            line_values = batch.column(
                batch.schema.get_field_index(LINE_FIELD),
            ).to_pylist()
            entity_values = batch.column(
                batch.schema.get_field_index(ENTITY_FIELD),
            ).to_pylist()
            label_values = batch.column(
                batch.schema.get_field_index(ANOMALOUS_FIELD),
            ).to_pylist()

            for line_order, entity_id, raw_label in zip(
                line_values,
                entity_values,
                label_values,
                strict=True,
            ):
                if raw_label is None:
                    continue

                label = int(raw_label)
                if label == 0:
                    continue

                if line_order is not None:
                    line_labels[int(line_order)] = label

                if entity_id is not None:
                    group_labels.setdefault(str(entity_id), label)

        return line_labels, group_labels

    @staticmethod
    def _rows_from_batch(batch: pa.RecordBatch) -> Iterator[StructuredLine]:
        """Yield StructuredLine rows without materialising whole batches as dicts.

        Args:
            batch (pa.RecordBatch): Arrow record batch to decode.

        Yields:
            StructuredLine: Structured rows decoded from the record batch.
        """

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
        """Return the underlying PyArrow dataset instance.

        Returns:
            ds.Dataset: PyArrow dataset over the structured parquet cache.
        """
        return ds.dataset(
            self.structured_data_cache(self.dataset_name),
            format="parquet",
            exclude_invalid_files=True,
            partitioning=ds.partitioning(
                schema=pa.schema([pa.field(ENTITY_BUCKET_FIELD, pa.int32())]),
                flavor="hive",
            ),
        )

    # Statistics helpers
    def count_rows(self) -> int:
        """Return total number of structured rows.

        Returns:
            int: Total number of structured rows.
        """
        return self._dataset().count_rows()

    def count_entities_by_label(
        self,
        label_for_group: Callable[[str], int | None],
    ) -> EntityLabelCounts:
        """Return counts of normal and total distinct entity ids.

        Args:
            label_for_group (Callable[[str], int | None]): Lookup that maps each
                entity id to its anomaly label.

        Returns:
            EntityLabelCounts: Normal and total distinct entity counts.
        """
        scanner = self._dataset().scanner(
            columns=[ENTITY_FIELD],
            batch_size=self._DEFAULT_BATCH_SIZE,
        )
        normals = 0
        entities_seen: set[str] = set()

        for batch in scanner.to_batches():
            col = batch.column(0)
            for val in col.to_pylist():
                if val is None:
                    continue
                entity_id = str(val)
                if entity_id in entities_seen:
                    continue
                entities_seen.add(entity_id)
                label = label_for_group(entity_id)
                if is_anomalous_label(label):
                    continue
                normals += 1

        total = len(entities_seen)
        return EntityLabelCounts(normal_entities=normals, total_entities=total)

    def timestamp_bounds(self) -> tuple[int | None, int | None]:
        """Return min and max timestamps present in the dataset.

        Returns:
            tuple[int | None, int | None]: Minimum and maximum timestamps, if any.
        """
        ts_scanner = self._dataset().scanner(
            columns=[TIMESTAMP_FIELD],
            batch_size=self._DEFAULT_BATCH_SIZE,
            batch_readahead=2,
            fragment_readahead=1,
        )

        min_ts: int | None = None
        max_ts: int | None = None
        pc_min_max = getattr(pc, "min_max")  # noqa: B009 - stubs miss this function

        for batch in ts_scanner.to_batches():
            col = batch.column(0)
            if len(col) == 0:
                continue

            stats = pc_min_max(col)
            batch_min = stats["min"]
            batch_max = stats["max"]

            if batch_min.is_valid:
                min_value = batch_min.as_py()
                min_ts = min_value if min_ts is None else min(min_ts, min_value)
            if batch_max.is_valid:
                max_value = batch_max.as_py()
                max_ts = max_value if max_ts is None else max(max_ts, max_value)

        return min_ts, max_ts

    def _projected_columns(
        self,
        columns: Sequence[str] | None,
    ) -> list[str]:
        """Return the projected parquet columns for a row iterator.

        Args:
            columns (Sequence[str] | None): Caller-requested projection, or
                `None` for the full structured schema.

        Returns:
            list[str]: Column names to request from the dataset scanner.
        """
        if columns is not None:
            # Preserve caller order, drop duplicates.
            return list(dict.fromkeys(columns))
        return list(self._structured_columns())

    @staticmethod
    def _structured_columns() -> tuple[str, ...]:
        """Return the canonical structured-row column ordering.

        Returns:
            tuple[str, ...]: Structured row field names in dataclass order.
        """
        return tuple(StructuredLine.__dataclass_fields__.keys())

    def iter_entity_sequences(
        self,
    ) -> Callable[[], Iterator[Collection[StructuredLine]]]:
        """Yield sequences grouped by entity in chronological order.

        Returns:
            Callable[[], Iterator[Collection[StructuredLine]]]: Callable producing
                entity-grouped windows ordered by a materialised chronology
                index, with a row-based fallback for older caches.
        """

        def _iter() -> Iterator[Collection[StructuredLine]]:
            chronology_index = self.load_entity_chronology_index()
            bucket_iters = [
                self._iter_entity_groups_in_bucket(
                    bucket_id,
                    chronology_index=chronology_index,
                )
                for bucket_id in sorted(self._iter_buckets())
            ]

            heap: list[
                tuple[EntityChronologyKey, int, _EntityGroup, Iterator[_EntityGroup]]
            ] = []
            for bucket_index, bucket_iter in enumerate(bucket_iters):
                try:
                    first_group = next(bucket_iter)
                except StopIteration:
                    continue
                heap.append((first_group.order, bucket_index, first_group, bucket_iter))

            heapq.heapify(heap)

            while heap:
                _, bucket_index, group, bucket_iter = heapq.heappop(heap)
                yield group.rows

                try:
                    next_group = next(bucket_iter)
                except StopIteration:
                    continue
                heapq.heappush(
                    heap,
                    (next_group.order, bucket_index, next_group, bucket_iter),
                )

        return _iter

    def _iter_entity_groups_in_bucket(
        self,
        bucket_id: int,
        *,
        chronology_index: dict[str, EntityChronologyKey],
    ) -> Iterator[_EntityGroup]:
        """Yield entity groups from one bucket ordered by first timestamp.

        Args:
            bucket_id (int): Parquet bucket identifier to scan.
            chronology_index (dict[str, EntityChronologyKey]): Materialised
                chronology metadata keyed by entity id.

        Yields:
            _EntityGroup: One entity group from the requested bucket.
        """
        by_entity: dict[str, list[StructuredLine]] = {}
        for row in self.iter_structured_lines(
            filter_expr=(ds.field(ENTITY_BUCKET_FIELD) == bucket_id),
        )():
            if row.entity_id is None:
                continue
            by_entity.setdefault(row.entity_id, []).append(row)

        yield from sorted(
            (
                self._entity_group_for_rows(
                    entity_id,
                    rows,
                    chronology_index=chronology_index,
                )
                for entity_id, rows in by_entity.items()
            ),
            key=lambda group: group.order,
        )

    @staticmethod
    def _entity_group_for_rows(
        entity_id: str,
        rows: Collection[StructuredLine],
        *,
        chronology_index: dict[str, EntityChronologyKey],
    ) -> _EntityGroup:
        """Build chronological metadata for one entity's grouped rows.

        Args:
            entity_id (str): Entity identifier for the grouped rows.
            rows (Collection[StructuredLine]): Structured rows belonging to
                one entity.
            chronology_index (dict[str, EntityChronologyKey]): Materialised
                chronology metadata keyed by entity id.

        Returns:
            _EntityGroup: Entity rows paired with chronological order metadata.
        """
        order = chronology_index.get(entity_id)
        if order is None:
            first_row = next(iter(rows), None)
            if first_row is None:
                order = EntityChronologyKey(1, 0, 0, entity_id)
            else:
                first_timestamp = first_row.timestamp_unix_ms
                order = EntityChronologyKey(
                    first_timestamp_missing=1 if first_timestamp is None else 0,
                    first_timestamp_unix_ms=0
                    if first_timestamp is None
                    else int(first_timestamp),
                    first_line_order=int(first_row.line_order),
                    entity_id=entity_id,
                )
        return _EntityGroup(order=order, rows=tuple(rows))

    def entity_chronology_index_path(self) -> Path:
        """Return the sidecar path storing entity chronology metadata.

        Returns:
            Path: JSONL sidecar path used for entity chronology ordering.
        """
        return (
            self.structured_data_cache(self.dataset_name)
            / ENTITY_CHRONOLOGY_INDEX_FILENAME
        )

    def load_entity_chronology_index(self) -> dict[str, EntityChronologyKey]:
        """Load the materialised chronology sidecar, if it exists.

        Returns:
            dict[str, EntityChronologyKey]: Chronology metadata keyed by entity id.
        """
        index_path = self.entity_chronology_index_path()
        if not index_path.exists():
            return {}

        chronology: dict[str, EntityChronologyKey] = {}
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                payload = json.loads(raw)
                chronology[str(payload["entity_id"])] = EntityChronologyKey(
                    first_timestamp_missing=int(payload["first_timestamp_missing"]),
                    first_timestamp_unix_ms=int(payload["first_timestamp_unix_ms"]),
                    first_line_order=int(payload["first_line_order"]),
                    entity_id=str(payload["entity_id"]),
                )
        return chronology

    def _iter_buckets(self) -> set[int]:
        """Enumerate bucket ids present in the parquet dataset.

        Returns:
            set[int]: Entity bucket identifiers present in the dataset.
        """
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

    @staticmethod
    def _stream_time_windows(
        rows: Iterator[StructuredLine],
        *,
        first_ts: int,
        last_ts: int,
        time_span_ms: int,
        step: int,
    ) -> Iterator[tuple[StructuredLine, ...]]:
        """Slide a time window over rows, yielding buffered windows.

        Args:
            rows (Iterator[StructuredLine]): Timestamp-ordered structured rows.
            first_ts (int): Earliest timestamp present in the dataset.
            last_ts (int): Latest timestamp present in the dataset.
            time_span_ms (int): Width of each emitted time window.
            step (int): Window step size in milliseconds.

        Yields:
            tuple[StructuredLine, ...]: Rows belonging to each emitted time window.
        """
        buffer: deque[StructuredLine] = deque()
        window_start = first_ts
        window_end = window_start + time_span_ms

        # TODO: Handle case where time_span_ms is larger than the total timestamp range

        for row in rows:
            ts = row.timestamp_unix_ms
            if ts is None:
                continue

            while ts >= window_end:
                if buffer:
                    yield tuple(buffer)

                window_start += step
                window_end = window_start + time_span_ms
                while buffer and (buffer[0].timestamp_unix_ms or 0) < window_start:
                    buffer.popleft()

            buffer.append(row)

        while window_start <= last_ts:
            if buffer:
                yield tuple(buffer)
            window_start += step
            window_end = window_start + time_span_ms
            while buffer and (buffer[0].timestamp_unix_ms or 0) < window_start:
                buffer.popleft()

    def iter_time_window_sequences(
        self,
        time_span_ms: int,
        step_span_ms: int | None = None,
    ) -> Callable[[], Iterator[Collection[StructuredLine]]]:
        """Yield sequences grouped by sliding time windows.

        Args:
            time_span_ms (int): Width of each window in milliseconds.
            step_span_ms (int | None): Optional step between successive windows.
                Defaults to `time_span_ms`.

        Returns:
            Callable[[], Iterator[Collection[StructuredLine]]]: Callable producing
                time-window grouped rows.
        """

        def _iter() -> Iterator[Collection[StructuredLine]]:
            first_ts, last_ts = self.timestamp_bounds()
            if first_ts is None or last_ts is None:
                msg = "No timestamps available for time-based windowing."
                raise ValueError(msg)

            step = time_span_ms if step_span_ms is None else step_span_ms
            if time_span_ms <= 0 or step <= 0:
                msg = "time_span_ms and step_span_ms must be positive integers"
                raise ValueError(msg)

            yield from self._stream_time_windows(
                self._iter_structured_lines_by_timestamp()(),
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
        """Yield sequences of fixed window size over ordered rows.

        Args:
            window_size (int): Number of rows in each emitted window.
            step_size (int | None): Optional step between successive windows.
                Defaults to `window_size`.

        Returns:
            Callable[[], Iterator[Collection[StructuredLine]]]: Callable producing
                fixed-size row windows.
        """

        def _iter() -> Iterator[Collection[StructuredLine]]:
            step = window_size if step_size is None else step_size
            if window_size <= 0 or step <= 0:
                msg = "window_size and step_size must be positive integers"
                raise ValueError(msg)

            buffer: deque[StructuredLine] = deque()
            for row in self.iter_structured_lines(
                batch_size=self._DEFAULT_BATCH_SIZE,
            )():
                buffer.append(row)
                if len(buffer) == window_size:
                    yield tuple(buffer)

                    for _ in range(step):
                        if not buffer:
                            break
                        buffer.popleft()

            if buffer:
                yield tuple(buffer)

        return _iter

    def _iter_structured_lines_by_timestamp(
        self,
    ) -> Callable[[], Iterator[StructuredLine]]:
        """Merge bucketed parquet fragments into global timestamp order.

        Returns:
            Callable[[], Iterator[StructuredLine]]: Callable producing globally
                time-ordered structured rows.
        """

        def _iter() -> Iterator[StructuredLine]:
            buckets = sorted(self._iter_buckets())
            buckets_with_null = [*buckets, None]

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
                heap.append((self._timestamp_row_key(first), idx, first, it))

            heapq.heapify(heap)

            while heap:
                _, idx, row, it = heapq.heappop(heap)
                yield row
                try:
                    nxt = next(it)
                except StopIteration:
                    continue
                heapq.heappush(heap, (self._timestamp_row_key(nxt), idx, nxt, it))

        return _iter

    @staticmethod
    def _timestamp_row_key(row: StructuredLine) -> tuple[int, int, int]:
        """Return a deterministic ordering key for timestamp-ordered iteration.

        Args:
            row (StructuredLine): Row to convert into a sortable key.

        Returns:
            tuple[int, int, int]: Sort key placing missing timestamps last and
                preserving deterministic ordering.
        """
        ts = row.timestamp_unix_ms
        # Missing timestamps go after those with timestamps; line_order
        # acts as a deterministic tiebreaker.
        return (1 if ts is None else 0, int(ts or 0), row.line_order)
