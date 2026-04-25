"""Tests for the public fluent dataset builder API."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest
from prefect.logging import disable_run_logger
from typing_extensions import override

from anomalog import DatasetSpec
from anomalog.cache import CachePathsConfig
from anomalog.labels import CSVReader
from anomalog.parsers import BGLParser, Drain3Parser, ParquetStructuredSink
from anomalog.parsers.structured import (
    BaseStructuredLine,
    StructuredParser,
    StructuredSink,
)
from anomalog.parsers.structured.contracts import EntityLabelCounts, StructuredLine
from anomalog.parsers.template import ExtractedParameters, LogTemplate, TemplateParser
from anomalog.presets import bgl, hdfs_v1, preset_names, resolve_preset
from anomalog.sources import DatasetSource
from anomalog.sources.local import LocalZipSource
from tests.unit.helpers import InMemoryStructuredSink, structured_line


@dataclass(frozen=True)
class _StubSource(DatasetSource):
    name: ClassVar[str] = "stub"
    dataset_root: Path
    raw_logs_file: Path

    def materialise(
        self,
        *,
        dst_dir: Path,
    ) -> Path:
        del dst_dir
        return self.dataset_root

    def raw_logs_path(
        self,
        *,
        dataset_name: str,
        dataset_root: Path,
    ) -> Path:
        del dataset_name
        assert dataset_root == self.dataset_root
        return self.raw_logs_file


@dataclass(frozen=True)
class _NullParser(StructuredParser):
    name: ClassVar[str] = "null"

    @override
    def parse_line(self, raw_line: str) -> BaseStructuredLine | None:
        del raw_line
        return None


class _RecordingTemplateParser(TemplateParser):
    name = "recording"
    seen_lines: ClassVar[list[str]] = []

    def __init__(self, dataset_name: str | None = None) -> None:
        self.dataset_name = dataset_name

    @override
    def inference(
        self,
        unstructured_text: str,
    ) -> tuple[LogTemplate, ExtractedParameters]:
        return unstructured_text, []

    def train(
        self,
        untemplated_text_iterator: Callable[[], Iterator[str]],
    ) -> None:
        type(self).seen_lines.extend(list(untemplated_text_iterator()))


class _RecordingSink(StructuredSink):
    rows: ClassVar[list[StructuredLine]] = []
    seen_raw_paths: ClassVar[list[Path]] = []

    def __init__(
        self,
        *,
        dataset_name: str,
        raw_dataset_path: Path,
        parser: StructuredParser,
        cache_paths: CachePathsConfig,
    ) -> None:
        del cache_paths
        type(self).seen_raw_paths.append(raw_dataset_path)
        self._sink = InMemoryStructuredSink(
            dataset_name=dataset_name,
            raw_dataset_path=raw_dataset_path,
            parser=parser,
            rows=type(self).rows,
            anomalies_inline=True,
        )

    @property
    def dataset_name(self) -> str:
        return self._sink.dataset_name

    @property
    def raw_dataset_path(self) -> Path:
        return self._sink.raw_dataset_path

    @property
    def parser(self) -> StructuredParser:
        return self._sink.parser

    def write_structured_lines(self) -> bool:
        return self._sink.write_structured_lines()

    def iter_structured_lines(
        self,
        columns: Sequence[str] | None = None,
    ) -> Callable[[], Iterator[StructuredLine]]:
        return self._sink.iter_structured_lines(columns=columns)

    def load_inline_label_cache(self) -> tuple[dict[int, int], dict[str, int]]:
        return self._sink.load_inline_label_cache()

    def count_rows(self) -> int:
        return self._sink.count_rows()

    def count_entities_by_label(
        self,
        label_for_group: Callable[[str], int | None],
    ) -> EntityLabelCounts:
        return self._sink.count_entities_by_label(label_for_group)

    def timestamp_bounds(self) -> tuple[int | None, int | None]:
        return self._sink.timestamp_bounds()

    def iter_entity_sequences(self) -> Callable[[], Iterator[Sequence[StructuredLine]]]:
        return self._sink.iter_entity_sequences()

    def iter_fixed_window_sequences(
        self,
        window_size: int,
        step_size: int | None = None,
    ) -> Callable[[], Iterator[Sequence[StructuredLine]]]:
        return self._sink.iter_fixed_window_sequences(window_size, step_size)

    def iter_time_window_sequences(
        self,
        time_span_ms: int,
        step_span_ms: int | None = None,
    ) -> Callable[[], Iterator[Sequence[StructuredLine]]]:
        return self._sink.iter_time_window_sequences(time_span_ms, step_span_ms)


def test_dataset_spec_builder_methods_are_pure() -> None:
    """Fluent builder methods return new configured specs without mutation."""
    base = DatasetSpec("demo")
    source = LocalZipSource(Path("demo.zip"), raw_logs_relpath=Path("BGL.log"))
    labels = CSVReader(relative_path=Path("labels.csv"))
    templated = (
        base.from_source(source)
        .parse_with(BGLParser())
        .label_with(labels)
        .template_with(Drain3Parser)
        .with_cache_paths(
            CachePathsConfig(
                data_root=Path("demo-data"),
                cache_root=Path("demo-cache"),
            ),
        )
    )

    assert base.source is None
    assert base.structured_parser is None
    assert base.template_parser is Drain3Parser
    assert base.structured_sink is ParquetStructuredSink
    assert templated.source is source
    assert isinstance(templated.structured_parser, BGLParser)
    assert templated.anomaly_label_reader is labels
    assert templated.template_parser is Drain3Parser
    assert templated.source.raw_logs_relpath == Path("BGL.log")
    assert templated.cache_paths.data_root == Path("demo-data")


def test_dataset_spec_build_requires_source_and_structured_parser() -> None:
    """Build validates the required source and structured parser stages."""
    with pytest.raises(ValueError, match="requires a source"):
        DatasetSpec("demo").build()

    with pytest.raises(ValueError, match="requires a structured parser"):
        DatasetSpec("demo").from_source(
            LocalZipSource(Path("demo.zip")),
        ).build()


# Protects the default-Drain3 contract.
# The nearby uncovered branch is an internal impossible-state guard.
@pytest.mark.allow_no_new_coverage
def test_dataset_spec_defaults_to_drain3_template_parser() -> None:
    """Dataset specs default to Drain3 unless explicitly overridden."""
    spec = (
        DatasetSpec("demo")
        .from_source(LocalZipSource(Path("demo.zip")))
        .parse_with(
            BGLParser(),
        )
    )

    assert spec.template_parser is Drain3Parser
    assert bgl.template_parser is Drain3Parser
    assert hdfs_v1.template_parser is Drain3Parser


def test_builtin_presets_register_and_resolve_by_name() -> None:
    """Built-in presets should be available through the public registry."""
    assert resolve_preset("bgl") is bgl
    assert resolve_preset("hdfs_v1") is hdfs_v1
    assert set(preset_names()) >= {"bgl", "hdfs_v1"}


def test_builtin_presets_reject_unknown_names() -> None:
    """Unknown preset names raise a descriptive KeyError."""
    with pytest.raises(KeyError, match="Unsupported preset: 'missing'"):
        resolve_preset("missing")


def test_dataset_spec_build_requires_non_empty_dataset_name() -> None:
    """Compilation validates dataset names before running the flow."""
    spec = (
        DatasetSpec("")
        .from_source(LocalZipSource(Path("demo.zip")))
        .parse_with(BGLParser())
        .template_with(Drain3Parser)
    )

    with pytest.raises(ValueError, match="non-empty dataset name"):
        spec.build()


def test_dataset_spec_clear_cache_removes_dataset_roots(tmp_path: Path) -> None:
    """Cache clearing should remove only the targeted dataset roots.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for data and cache roots.
    """
    cache_paths = CachePathsConfig(
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
    )
    targeted_data = cache_paths.data_root / "demo"
    targeted_cache = cache_paths.cache_root / "demo"
    other_data = cache_paths.data_root / "other"
    other_cache = cache_paths.cache_root / "other"

    targeted_data.mkdir(parents=True)
    targeted_cache.mkdir(parents=True)
    other_data.mkdir(parents=True)
    other_cache.mkdir(parents=True)

    (targeted_data / "raw.log").write_text("hello\n", encoding="utf-8")
    (targeted_cache / "artifact.txt").write_text("cached\n", encoding="utf-8")
    (other_data / "raw.log").write_text("keep\n", encoding="utf-8")
    (other_cache / "artifact.txt").write_text("keep\n", encoding="utf-8")

    DatasetSpec("demo").with_cache_paths(cache_paths).clear_cache()

    assert not targeted_data.exists()
    assert not targeted_cache.exists()
    assert other_data.exists()
    assert other_cache.exists()


def test_dataset_spec_clear_cache_requires_non_empty_dataset_name() -> None:
    """Cache clearing validates dataset names before deleting anything."""
    with pytest.raises(ValueError, match="non-empty dataset name"):
        DatasetSpec("").clear_cache()


def test_dataset_spec_clear_cache_ignores_missing_dataset_roots(tmp_path: Path) -> None:
    """Clearing a dataset with no local cache should succeed silently.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for data and cache roots.
    """
    DatasetSpec("demo").with_cache_paths(
        CachePathsConfig(
            data_root=tmp_path / "data",
            cache_root=tmp_path / "cache",
        ),
    ).clear_cache()


def test_dataset_spec_build_uses_configured_sink_type(tmp_path: Path) -> None:
    """Build should use the configured sink type instead of hardcoding parquet.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for the fake dataset tree.
    """
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    raw_logs_path = dataset_root / "demo.log"
    raw_logs_path.write_text("ignored\n", encoding="utf-8")
    _RecordingSink.rows = [
        structured_line(
            line_order=0,
            timestamp_unix_ms=1_000,
            entity_id="node-a",
            untemplated_message_text="hello",
            anomalous=1,
        ),
    ]
    _RecordingSink.seen_raw_paths = []
    _RecordingTemplateParser.seen_lines = []

    spec = (
        DatasetSpec("demo")
        .from_source(
            _StubSource(dataset_root=dataset_root, raw_logs_file=raw_logs_path),
        )
        .parse_with(_NullParser())
        .store_with(_RecordingSink)
        .template_with(_RecordingTemplateParser)
    )

    with disable_run_logger():
        templated = spec.build()

    assert _RecordingSink.seen_raw_paths == [raw_logs_path]
    assert isinstance(templated.sink, _RecordingSink)
    assert _RecordingTemplateParser.seen_lines == ["hello"]
    assert isinstance(templated.template_parser, _RecordingTemplateParser)
    assert templated.template_parser.dataset_name == "demo"
