"""Tests for `RawDataset` orchestration."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from prefect.logging import disable_run_logger

from anomalog.anomaly_label_reader import (
    AnomalyLabelLookup,
    AnomalyLabelReader,
)
from anomalog.cache import CachePathsConfig
from anomalog.sources.contracts import DatasetSource
from anomalog.sources.raw_dataset import RawDataset
from anomalog.structured_parsers.contracts import StructuredSink
from tests.unit.helpers import (
    InMemoryStructuredSink,
    NullStructuredParser,
    label_lookup,
    structured_line,
)


@dataclass(frozen=True)
class _Source(DatasetSource):
    materialised_to: Path | None = None

    def materialise(self, dst_dir: Path) -> Path:
        object.__setattr__(self, "materialised_to", dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        return dst_dir


@dataclass(frozen=True)
class _LabelReader(AnomalyLabelReader):
    lookup: AnomalyLabelLookup
    bound_root: Path | None = None
    bound_sink: StructuredSink | None = None

    def load(self) -> AnomalyLabelLookup:
        if self.bound_root is None or self.bound_sink is None:
            return self.lookup
        sink = self.bound_sink

        def _label_for_group(entity_id: str) -> int | None:
            # This derived lookup only returns a label when `with_context`
            # supplied the sink and dataset root that `RawDataset` is expected
            # to bind before calling `load`.
            if entity_id != "node-a":
                return None
            if self.bound_root != sink.raw_dataset_path.parent:
                return None
            if sink.dataset_name != "demo":
                return None
            return 1

        return label_lookup(label_for_group=_label_for_group)

    def with_context(
        self,
        *,
        dataset_root: Path,
        sink: StructuredSink,
    ) -> "_LabelReader":
        return _LabelReader(
            lookup=self.lookup,
            bound_root=dataset_root,
            bound_sink=sink,
        )


def _dataset(
    tmp_path: Path,
    *,
    anomaly_label_reader: _LabelReader | None = None,
) -> RawDataset:
    return RawDataset(
        dataset_name="demo",
        source=_Source(),
        structured_parser=NullStructuredParser(),
        cache_paths=CachePathsConfig(
            data_root=tmp_path / "data",
            cache_root=tmp_path / "cache",
        ),
        anomaly_label_reader=anomaly_label_reader,
    )


def test_raw_dataset_iter_lines_strips_trailing_newlines(tmp_path: Path) -> None:
    """iter_lines returns raw lines without trailing newline characters."""
    dataset = _dataset(tmp_path)
    dataset.raw_logs_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.raw_logs_path.write_text("a\nb\n", encoding="utf-8")

    assert list(dataset.iter_lines()) == ["a", "b"]


def test_extract_structured_components_uses_inline_labels_when_present(
    tmp_path: Path,
) -> None:
    """Inline anomaly labels bypass the external reader branch."""
    dataset = _dataset(tmp_path)
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=dataset.raw_logs_path,
        parser=NullStructuredParser(),
        rows=[
            structured_line(
                line_order=0,
                timestamp_unix_ms=None,
                entity_id="node-a",
                untemplated_message_text="hello",
                anomalous=1,
            ),
        ],
        anomalies_inline=True,
    )

    with disable_run_logger():
        structured = dataset.extract_structured_components(
            sink=sink,
        )

    assert structured.anomaly_labels.label_for_line(0) == 1
    assert structured.anomaly_labels.label_for_group("node-a") == 1


@pytest.mark.allow_no_new_coverage
def test_extract_structured_components_uses_contextual_label_reader_when_needed(
    tmp_path: Path,
) -> None:
    """External label readers are bound to dataset root and sink."""
    # This protects the orchestration contract that RawDataset passes the sink
    # and dataset root into `with_context` before loading labels. A nearby
    # uncovered branch would not check that those exact values are propagated.
    reader = _LabelReader(lookup=label_lookup())
    dataset = _dataset(tmp_path, anomaly_label_reader=reader)
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=dataset.raw_logs_path,
        parser=NullStructuredParser(),
        rows=[
            structured_line(
                line_order=0,
                timestamp_unix_ms=None,
                entity_id="node-a",
                untemplated_message_text="hello",
                anomalous=None,
            ),
        ],
        anomalies_inline=False,
    )

    with disable_run_logger():
        structured = dataset.extract_structured_components(
            sink=sink,
        )

    assert structured.anomaly_labels.label_for_group("node-a") == 1
    assert structured.anomaly_labels.label_for_group("missing") is None


def test_extract_structured_components_requires_label_reader_when_no_inline_labels(
    tmp_path: Path,
) -> None:
    """Datasets without inline or external labels raise a clear error."""
    dataset = _dataset(tmp_path)
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=dataset.raw_logs_path,
        parser=NullStructuredParser(),
        rows=[],
        anomalies_inline=False,
    )

    with (
        disable_run_logger(),
        pytest.raises(ValueError, match="no inline anomaly labels"),
    ):
        dataset.extract_structured_components(
            sink=sink,
        )
