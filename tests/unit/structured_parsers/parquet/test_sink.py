"""Tests for `ParquetStructuredSink`."""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar

import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from prefect.logging import disable_run_logger
from typing_extensions import override

from anomalog.cache import CachePathsConfig
from anomalog.parsers.structured.contracts import (
    LINE_FIELD,
    TIMESTAMP_FIELD,
    UNTEMPLATED_FIELD,
    BaseStructuredLine,
    StructuredLine,
    StructuredParser,
)
from anomalog.parsers.structured.parquet import writer_worker
from anomalog.parsers.structured.parquet.sink import ParquetStructuredSink
from anomalog.parsers.structured.parquet.writer_worker import (
    WriterConfig,
    extract_structured_components,
)
from tests.unit.helpers import structured_line


class _Parser(StructuredParser):
    """Test-only parser for compact fixture rows written to the raw log file."""

    name: ClassVar[str] = "test"

    @override
    def parse_line(self, raw_line: str) -> BaseStructuredLine | None:
        timestamp_s, entity_id, message, anomalous_s = raw_line.split("|", maxsplit=3)
        return BaseStructuredLine(
            timestamp_unix_ms=int(timestamp_s) if timestamp_s else None,
            entity_id=entity_id or None,
            untemplated_message_text=message,
            anomalous=int(anomalous_s) if anomalous_s else None,
        )


class _NullParser(StructuredParser):
    """Parser that drops every line for empty-output extraction tests."""

    name: ClassVar[str] = "null"

    @override
    def parse_line(self, raw_line: str) -> BaseStructuredLine | None:
        del raw_line
        return None


class _LenientParser(_Parser):
    """Parser variant that treats malformed lines as skipped rows."""

    @override
    def parse_line(self, raw_line: str) -> BaseStructuredLine | None:
        try:
            return super().parse_line(raw_line)
        except ValueError:
            return None


def _make_sink(tmp_path: Path) -> ParquetStructuredSink:
    """Create a sink rooted entirely inside the per-test temp directory.

    Args:
        tmp_path (Path): Temporary directory backing the sink cache roots.

    Returns:
        ParquetStructuredSink: Sink backed only by the test temp directory.
    """
    cache_paths = CachePathsConfig(
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
    )
    return ParquetStructuredSink(
        dataset_name="demo",
        raw_dataset_path=tmp_path / "raw.log",
        parser=_Parser(),
        cache_paths=cache_paths,
    )


def _write_rows(
    sink: ParquetStructuredSink,
    rows: list[StructuredLine],
) -> None:
    """Persist fixture rows through the real extractor into parquet files."""
    sink.raw_dataset_path.write_text(
        "\n".join(_raw_line(row) for row in rows),
        encoding="utf-8",
    )
    with disable_run_logger():
        extract_structured_components(
            raw_input_path=sink.raw_dataset_path,
            parser=sink.parser,
            parquet_out_dir=sink.structured_data_cache(sink.dataset_name),
            config=WriterConfig(
                buckets=4,
                batch_rows=2,
                max_rows_per_file=8,
                max_rows_per_group=8,
                max_open_files=8,
                log_every_rows=0,
                max_partitions=8,
            ),
        )


def _raw_line(row: StructuredLine) -> str:
    """Serialize a fixture row into the compact format consumed by `_Parser`.

    Args:
        row (StructuredLine): Structured row to serialize.

    Returns:
        str: Pipe-delimited raw line for the test parser.
    """
    return "|".join(
        [
            "" if row.timestamp_unix_ms is None else str(row.timestamp_unix_ms),
            "" if row.entity_id is None else row.entity_id,
            row.untemplated_message_text,
            "" if row.anomalous is None else str(row.anomalous),
        ],
    )


# Distinct fixture sets are kept only where they encode different invariants:
# - `WINDOW_AND_BATCH_ROWS` supports batch defaults and cross-partition ordering.
# - `ENTITY_GROUP_ROWS` focuses on entity counts/labels.
WINDOW_AND_BATCH_ROWS = [
    structured_line(
        line_order=0,
        timestamp_unix_ms=None,
        entity_id=None,
        untemplated_message_text="",
        anomalous=None,
    ),
    structured_line(
        line_order=7,
        timestamp_unix_ms=12,
        entity_id=None,
        untemplated_message_text="hello",
        anomalous=None,
    ),
    structured_line(
        line_order=0,
        timestamp_unix_ms=100,
        entity_id="node-a",
        untemplated_message_text="first",
        anomalous=0,
    ),
    structured_line(
        line_order=1,
        timestamp_unix_ms=None,
        entity_id=None,
        untemplated_message_text="second",
        anomalous=None,
    ),
    structured_line(
        line_order=2,
        timestamp_unix_ms=12,
        entity_id="node-b",
        untemplated_message_text="b2",
        anomalous=0,
    ),
    structured_line(
        line_order=0,
        timestamp_unix_ms=0,
        entity_id="node-a",
        untemplated_message_text="a0",
        anomalous=0,
    ),
    structured_line(
        line_order=3,
        timestamp_unix_ms=18,
        entity_id="node-a",
        untemplated_message_text="a3",
        anomalous=0,
    ),
    structured_line(
        line_order=1,
        timestamp_unix_ms=5,
        entity_id="node-c",
        untemplated_message_text="c1",
        anomalous=0,
    ),
]

ENTITY_GROUP_ROWS = [
    structured_line(
        line_order=0,
        timestamp_unix_ms=100,
        entity_id="node-b",
        untemplated_message_text="b0",
        anomalous=0,
    ),
    structured_line(
        line_order=1,
        timestamp_unix_ms=120,
        entity_id="node-a",
        untemplated_message_text="a1",
        anomalous=0,
    ),
    structured_line(
        line_order=2,
        timestamp_unix_ms=140,
        entity_id="node-a",
        untemplated_message_text="a2",
        anomalous=1,
    ),
]


@pytest.mark.allow_no_new_coverage
def test_rows_from_batch_applies_defaults_for_missing_and_null_columns(
    tmp_path: Path,
) -> None:
    """Row decoding should fall back to current defaults for missing columns."""
    # This locks in the row-decoding defaults for missing projected columns. The
    # remaining uncovered lines in this area belong to parquet scan paths rather
    # than the batch-decoding contract exercised here.
    sink = _make_sink(tmp_path)
    batch = pa.record_batch(
        [
            pa.array([0, WINDOW_AND_BATCH_ROWS[1].line_order], type=pa.int64()),
            pa.array(
                [None, WINDOW_AND_BATCH_ROWS[1].timestamp_unix_ms],
                type=pa.int64(),
            ),
            pa.array([None, "hello"], type=pa.string()),
        ],
        names=[LINE_FIELD, TIMESTAMP_FIELD, UNTEMPLATED_FIELD],
    )

    rows = list(sink._rows_from_batch(batch))  # noqa: SLF001

    assert rows == WINDOW_AND_BATCH_ROWS[:2]


def test_iter_structured_lines_reads_rows_from_real_parquet_dataset(
    tmp_path: Path,
) -> None:
    """Iterating structured lines should round-trip rows written by the extractor."""
    sink = _make_sink(tmp_path)
    _write_rows(
        sink,
        WINDOW_AND_BATCH_ROWS[2:4],
    )

    rows = list(sink.iter_structured_lines()())

    assert rows == WINDOW_AND_BATCH_ROWS[2:4]


def test_iter_structured_lines_projection_applies_defaults_from_real_dataset(
    tmp_path: Path,
) -> None:
    """Projected parquet scans should fill omitted columns with defaults."""
    sink = _make_sink(tmp_path)
    _write_rows(
        sink,
        WINDOW_AND_BATCH_ROWS[2:4],
    )

    rows = list(
        sink.iter_structured_lines(
            columns=[TIMESTAMP_FIELD, LINE_FIELD],
        )(),
    )

    assert rows == [
        structured_line(
            line_order=0,
            timestamp_unix_ms=100,
            entity_id=None,
            untemplated_message_text="",
            anomalous=None,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=None,
            entity_id=None,
            untemplated_message_text="",
            anomalous=None,
        ),
    ]


def test_sink_statistics_and_entity_grouping_use_real_dataset(
    tmp_path: Path,
) -> None:
    """Entity-aware statistics should come from persisted parquet data."""
    sink = _make_sink(tmp_path)
    expected_row_count = 3
    _write_rows(
        sink,
        ENTITY_GROUP_ROWS,
    )

    normal_entities, total_entities = sink.count_entities_by_label(
        lambda entity_id: 1 if entity_id == "node-a" else 0,
    )
    sequences = list(sink.iter_entity_sequences()())

    assert sink.count_rows() == expected_row_count
    assert (normal_entities, total_entities) == (1, 2)
    assert [[row.entity_id for row in rows] for rows in sequences] == [
        ["node-b"],
        ["node-a", "node-a"],
    ]


def test_sink_entity_grouping_skips_null_entity_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entity grouping should skip rows without an entity id."""
    sink = _make_sink(tmp_path)

    def _iter_buckets(_self: ParquetStructuredSink) -> set[int]:
        return {0}

    def _iter_structured_lines(
        _self: ParquetStructuredSink,
        columns: list[str] | None = None,
        *,
        filter_expr: ds.Expression | None = None,
        batch_size: int | None = None,
    ) -> Callable[[], Iterator[StructuredLine]]:
        del columns, filter_expr, batch_size
        return lambda: iter(
            [
                WINDOW_AND_BATCH_ROWS[2],
                structured_line(
                    line_order=99,
                    timestamp_unix_ms=111,
                    entity_id=None,
                    untemplated_message_text="ignored",
                    anomalous=0,
                ),
                WINDOW_AND_BATCH_ROWS[7],
            ],
        )

    monkeypatch.setattr(ParquetStructuredSink, "_iter_buckets", _iter_buckets)
    monkeypatch.setattr(
        ParquetStructuredSink,
        "iter_structured_lines",
        _iter_structured_lines,
    )

    sequences = list(sink.iter_entity_sequences()())

    assert [[row.entity_id for row in rows] for rows in sequences] == [
        ["node-a"],
        ["node-c"],
    ]


@pytest.mark.allow_no_new_coverage
def test_sink_timestamp_bounds_reads_min_and_max_from_real_dataset(
    tmp_path: Path,
) -> None:
    """Timestamp bounds should scan persisted parquet data for min and max."""
    # This guards the externally visible min/max contract. Nearby uncovered
    # branches are empty-batch and null-handling details that do not replace
    # this persisted-dataset regression check.
    sink = _make_sink(tmp_path)
    _write_rows(
        sink,
        [ENTITY_GROUP_ROWS[0], ENTITY_GROUP_ROWS[2]],
    )

    assert sink.timestamp_bounds() == (100, 140)


def test_sink_time_window_iteration_uses_global_timestamp_order_across_partitions(
    tmp_path: Path,
) -> None:
    """Time-window iteration should follow persisted timestamp order."""
    sink = _make_sink(tmp_path)
    _write_rows(
        sink,
        WINDOW_AND_BATCH_ROWS[4:],
    )

    time_windows = list(sink.iter_time_window_sequences(10, step_span_ms=5)())

    assert [[row.timestamp_unix_ms for row in rows] for rows in time_windows] == [
        [0, 5],
        [5, 12],
        [12, 18],
        [18],
    ]


def test_sink_time_window_iteration_includes_null_bucket_rows_in_timestamp_order(
    tmp_path: Path,
) -> None:
    """Timestamp ordering should merge null-entity rows with bucketed rows."""
    sink = _make_sink(tmp_path)
    _write_rows(
        sink,
        [
            WINDOW_AND_BATCH_ROWS[2],
            structured_line(
                line_order=1,
                timestamp_unix_ms=110,
                entity_id=None,
                untemplated_message_text="middle",
                anomalous=0,
            ),
            WINDOW_AND_BATCH_ROWS[0],
            ENTITY_GROUP_ROWS[2],
        ],
    )

    time_windows = list(sink.iter_time_window_sequences(15, step_span_ms=15)())

    assert [[row.timestamp_unix_ms for row in rows] for rows in time_windows] == [
        [100, 110],
        [140],
    ]


def test_sink_fixed_window_iteration_uses_global_order_across_partitions(
    tmp_path: Path,
) -> None:
    """Fixed-size windows should preserve global line order across partitions."""
    sink = _make_sink(tmp_path)
    _write_rows(
        sink,
        WINDOW_AND_BATCH_ROWS[4:],
    )

    fixed_windows = list(
        sink.iter_fixed_window_sequences(window_size=2, step_size=2)(),
    )

    assert [[row.line_order for row in rows] for rows in fixed_windows] == [
        [0, 1],
        [2, 3],
    ]


def test_sink_fixed_window_iteration_yields_partial_tail_when_step_exhausts_buffer(
    tmp_path: Path,
) -> None:
    """Fixed-size windows should emit a trailing partial window when rows remain."""
    sink = _make_sink(tmp_path)
    _write_rows(
        sink,
        WINDOW_AND_BATCH_ROWS[4:7],
    )

    fixed_windows = list(
        sink.iter_fixed_window_sequences(window_size=2, step_size=3)(),
    )

    assert [[row.line_order for row in rows] for rows in fixed_windows] == [
        [0, 1],
        [2],
    ]


def test_sink_time_window_iteration_rejects_non_positive_sizes(
    tmp_path: Path,
) -> None:
    """Time-window iteration should reject non-positive span and step values."""
    sink = _make_sink(tmp_path)
    _write_rows(
        sink,
        WINDOW_AND_BATCH_ROWS[4:],
    )

    with pytest.raises(ValueError, match="must be positive integers"):
        list(sink.iter_time_window_sequences(0)())

    with pytest.raises(ValueError, match="must be positive integers"):
        list(sink.iter_time_window_sequences(10, step_span_ms=0)())


def test_sink_time_window_iteration_requires_at_least_one_timestamp(
    tmp_path: Path,
) -> None:
    """Time-window iteration should fail when the dataset has no timestamps."""
    sink = _make_sink(tmp_path)
    _write_rows(
        sink,
        [WINDOW_AND_BATCH_ROWS[0], WINDOW_AND_BATCH_ROWS[3]],
    )

    with pytest.raises(
        ValueError,
        match=r"No timestamps available for time-based windowing\.",
    ):
        list(sink.iter_time_window_sequences(10)())


def test_sink_fixed_window_iteration_rejects_non_positive_sizes(
    tmp_path: Path,
) -> None:
    """Fixed-window iteration should reject non-positive size and step values."""
    sink = _make_sink(tmp_path)
    _write_rows(
        sink,
        WINDOW_AND_BATCH_ROWS[4:],
    )

    with pytest.raises(ValueError, match="must be positive integers"):
        list(sink.iter_fixed_window_sequences(0)())

    with pytest.raises(ValueError, match="must be positive integers"):
        list(sink.iter_fixed_window_sequences(2, step_size=0)())


def test_iter_record_batches_skips_unparseable_rows_and_populates_null_entity_bucket(
    tmp_path: Path,
) -> None:
    """Batch iteration should skip parser misses and preserve null entity buckets."""
    expected_rows = 2
    raw_input_path = tmp_path / "raw.log"
    raw_input_path.write_text(
        "100|node-a|first|0\ninvalid line\n101||second|",
        encoding="utf-8",
    )

    with disable_run_logger():
        iter_record_batches = vars(writer_worker)["_iter_record_batches"]
        batches = list(
            iter_record_batches(
                raw_input_path,
                _LenientParser(),
                cfg=WriterConfig(batch_rows=10, log_every_rows=1),
            ),
        )

    assert len(batches) == 1
    assert batches[0].num_rows == expected_rows
    assert batches[0].column("entity_bucket").to_pylist()[1] is None


def test_extract_structured_components_rejects_missing_input_path(
    tmp_path: Path,
) -> None:
    """Extraction should fail fast when the raw log file is absent."""
    with (
        disable_run_logger(),
        pytest.raises(FileNotFoundError, match="Input file does not exist"),
    ):
        extract_structured_components(
            raw_input_path=tmp_path / "missing.log",
            parser=_Parser(),
            parquet_out_dir=tmp_path / "out",
        )


def test_extract_structured_components_rejects_empty_parser_output(
    tmp_path: Path,
) -> None:
    """Extraction should fail when parsing produces no structured rows."""
    raw_input_path = tmp_path / "raw.log"
    raw_input_path.write_text("invalid line\n", encoding="utf-8")

    with (
        disable_run_logger(),
        pytest.raises(
            ValueError,
            match="No structured lines produced",
        ),
    ):
        extract_structured_components(
            raw_input_path=raw_input_path,
            parser=_NullParser(),
            parquet_out_dir=tmp_path / "out",
        )


def test_extract_structured_components_tolerates_racing_output_dir_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cleanup should ignore a directory disappearing between exists and rmtree."""
    raw_input_path = tmp_path / "raw.log"
    raw_input_path.write_text("100|node-a|first|0\n", encoding="utf-8")
    parquet_out_dir = tmp_path / "out"
    parquet_out_dir.mkdir()
    (parquet_out_dir / "stale.txt").write_text("stale", encoding="utf-8")

    def _raise_missing(_path: Path) -> None:
        msg = "already gone"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(
        "anomalog.parsers.structured.parquet.writer_worker.shutil.rmtree",
        _raise_missing,
    )

    with disable_run_logger():
        has_anomaly = extract_structured_components(
            raw_input_path=raw_input_path,
            parser=_Parser(),
            parquet_out_dir=parquet_out_dir,
            config=WriterConfig(
                buckets=2,
                batch_rows=1,
                max_rows_per_file=8,
                max_rows_per_group=8,
                max_open_files=8,
                log_every_rows=0,
                max_partitions=8,
            ),
        )

    assert has_anomaly is False
    assert parquet_out_dir.exists()


@pytest.mark.allow_no_new_coverage
def test_projected_columns_preserves_order_while_deduplicating(
    tmp_path: Path,
) -> None:
    """Projected columns should preserve caller order while dropping duplicates."""
    sink = _make_sink(tmp_path)

    # Keep this direct assertion because higher-level projection tests would not
    # clearly pinpoint a regression in the order-preserving deduplication logic.
    assert sink._projected_columns(  # noqa: SLF001
        [TIMESTAMP_FIELD, LINE_FIELD, TIMESTAMP_FIELD],
    ) == [TIMESTAMP_FIELD, LINE_FIELD]
