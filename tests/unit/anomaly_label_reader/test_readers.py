"""Tests for anomaly label reader implementations."""

from pathlib import Path

from prefect.logging import disable_run_logger

from anomalog.anomaly_label_reader import CSVReader, InlineReader
from anomalog.cache import CachePathsConfig
from anomalog.sources.contracts import DatasetSource
from anomalog.sources.raw_dataset import RawDataset
from anomalog.structured_parsers.contracts import StructuredLine
from tests.unit.helpers import InMemoryStructuredSink, NullStructuredParser

INLINE_CATEGORY_LABEL = 2


class _UnusedSource(DatasetSource):
    def materialise(self, dst_dir: Path) -> Path:
        return dst_dir


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


def test_raw_dataset_binds_csv_reader_to_sink_dataset_root(tmp_path: Path) -> None:
    """RawDataset resolves `CSVReader.relative_path` from the sink dataset root."""
    sink_dataset_root = tmp_path / "data" / "demo"
    sink_dataset_root.mkdir(parents=True)
    (sink_dataset_root / "labels.csv").write_text(
        "entity_id,anomalous\nnode-a,1\n",
        encoding="utf-8",
    )

    dataset = RawDataset(
        dataset_name="demo",
        source=_UnusedSource(),
        structured_parser=NullStructuredParser(),
        cache_paths=CachePathsConfig(
            data_root=tmp_path / "data",
            cache_root=tmp_path / "cache",
        ),
        anomaly_label_reader=CSVReader(relative_path=Path("labels.csv")),
    )
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=dataset.raw_logs_path,
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
        structured = dataset.extract_structured_components(sink=sink)

    assert structured.anomaly_labels.label_for_group("node-a") == 1
    assert structured.anomaly_labels.label_for_group("missing") is None


# TODO: Is this right implementation: If a group has multiple anomaly categories,
# should the first non-zero category is used as the label for all lines in that group?
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
