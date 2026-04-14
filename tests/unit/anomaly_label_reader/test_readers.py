"""Tests for anomaly label reader implementations."""

from pathlib import Path

import pytest
from prefect.logging import disable_run_logger
from typing_extensions import override

from anomalog.cache import CachePathsConfig
from anomalog.labels import CSVReader, InlineReader
from anomalog.parsers.structured.contracts import BaseStructuredLine, StructuredLine
from anomalog.parsers.structured.parquet.sink import ParquetStructuredSink
from anomalog.parsers.structured.parquet.writer_worker import (
    WriterConfig,
    extract_structured_components,
)
from tests.unit.helpers import InMemoryStructuredSink, NullStructuredParser

INLINE_CATEGORY_LABEL = 2


class _ParquetParser(NullStructuredParser):
    """Compact parser for parquet-backed inline-label tests."""

    @override
    def parse_line(self, raw_line: str) -> BaseStructuredLine | None:
        timestamp_s, entity_id, message, anomalous_s = raw_line.split("|", maxsplit=3)
        return BaseStructuredLine(
            timestamp_unix_ms=int(timestamp_s) if timestamp_s else None,
            entity_id=entity_id or None,
            untemplated_message_text=message,
            anomalous=int(anomalous_s) if anomalous_s else None,
        )


def test_csv_reader_loads_group_labels_and_ignores_invalid_rows(tmp_path: Path) -> None:
    """CSVReader returns only valid integer labels for configured entity ids."""
    labels_file = tmp_path / "labels.csv"
    labels_file.write_text(
        "entity_id,anomalous\nnode-a,1\nnode-b,nope\nnode-c,0\n",
        encoding="utf-8",
    )

    lookup = CSVReader(relative_path=Path("labels.csv"), dataset_root=tmp_path).load()

    assert lookup.label_for_group("node-a") == 1
    assert lookup.label_for_group("node-b") is None
    assert lookup.label_for_group("node-c") == 0
    assert lookup.label_for_line(10) is None


def test_csv_reader_with_context_binds_sink_dataset_root(tmp_path: Path) -> None:
    """CSVReader resolves relative_path from the provided dataset root."""
    sink_dataset_root = tmp_path / "data" / "demo"
    sink_dataset_root.mkdir(parents=True)
    (sink_dataset_root / "labels.csv").write_text(
        "entity_id,anomalous\nnode-a,1\n",
        encoding="utf-8",
    )

    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=sink_dataset_root / "demo.log",
        parser=NullStructuredParser(),
        rows=[
            StructuredLine(
                line_order=0,
                timestamp_unix_ms=1_000,
                entity_id="node-a",
                untemplated_message_text="hello",
                anomalous=None,
            ),
        ],
        anomalies_inline=False,
    )

    with disable_run_logger():
        lookup = (
            CSVReader(relative_path=Path("labels.csv"))
            .with_context(
                dataset_root=sink_dataset_root,
                sink=sink,
            )
            .load()
        )

    assert lookup.label_for_group("node-a") == 1
    assert lookup.label_for_group("missing") is None


# TODO: Is this right implementation: If a group has multiple anomaly categories,
# should the first non-zero category is used as the label for all lines in that group?
# Protects the generic sink-backed inline reader contract.
# The parquet-backed fast path has its own dedicated coverage test below.
@pytest.mark.allow_no_new_coverage
def test_inline_reader_loads_sparse_line_and_group_labels(tmp_path: Path) -> None:
    """InlineReader preserves the first non-zero label for each line and group."""
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=tmp_path / "structured.parquet",
        parser=NullStructuredParser(),
        rows=[
            StructuredLine(
                line_order=0,
                timestamp_unix_ms=1_000,
                entity_id="node-a",
                untemplated_message_text="ok",
                anomalous=0,
            ),
            StructuredLine(
                line_order=1,
                timestamp_unix_ms=1_500,
                entity_id="node-b",
                untemplated_message_text="bad",
                anomalous=INLINE_CATEGORY_LABEL,
            ),
            StructuredLine(
                line_order=2,
                timestamp_unix_ms=2_000,
                entity_id="node-b",
                untemplated_message_text="still bad",
                anomalous=1,
            ),
            StructuredLine(
                line_order=3,
                timestamp_unix_ms=2_500,
                entity_id=None,
                untemplated_message_text="missing",
                anomalous=1,
            ),
        ],
    )

    lookup = InlineReader(sink=sink).load()

    assert lookup.label_for_line(0) is None
    assert lookup.label_for_line(1) == INLINE_CATEGORY_LABEL
    assert lookup.label_for_line(2) == 1
    assert lookup.label_for_group("node-a") is None
    assert lookup.label_for_group("node-b") == INLINE_CATEGORY_LABEL


def test_inline_reader_loads_sparse_labels_from_parquet_sink(tmp_path: Path) -> None:
    """InlineReader uses the parquet-backed fast path without changing results."""
    cache_paths = CachePathsConfig(
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
    )
    sink = ParquetStructuredSink(
        dataset_name="demo",
        raw_dataset_path=tmp_path / "raw.log",
        parser=_ParquetParser(),
        cache_paths=cache_paths,
    )
    sink.raw_dataset_path.write_text(
        (
            "1000|node-a|ok|0\n"
            "1500|node-b|bad|2\n"
            "2000|node-b|still bad|1\n"
            "2500||missing|1"
        ),
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

    lookup = InlineReader(sink=sink).load()

    assert lookup.label_for_line(0) is None
    assert lookup.label_for_line(1) == INLINE_CATEGORY_LABEL
    assert lookup.label_for_line(2) == 1
    assert lookup.label_for_group("node-a") is None
    assert lookup.label_for_group("node-b") == INLINE_CATEGORY_LABEL
