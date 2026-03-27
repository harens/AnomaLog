"""Tests for structured and templated dataset helpers."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from anomalog.anomaly_label_reader import AnomalyLabelLookup
from anomalog.cache import CachePathsConfig
from anomalog.sequences import GroupingMode
from anomalog.structured_parsers.structured_dataset import StructuredDataset
from anomalog.template_parsers.templated_dataset import (
    ExtractedParameters,
    LogTemplate,
    TemplateParser,
)
from tests.unit.helpers import (
    InMemoryStructuredSink,
    NullStructuredParser,
    label_lookup,
    structured_line,
)

WINDOW_SIZE = 5
WINDOW_STEP = 2
TIME_SPAN_MS = 1_000
TIME_STEP_MS = 250


@dataclass(frozen=True)
class _RecordingTemplateParser(TemplateParser):
    dataset_name: str
    seen_lines: list[str]

    def inference(
        self,
        unstructured_text: str,
    ) -> tuple[LogTemplate, ExtractedParameters]:
        return unstructured_text.upper(), ()

    def train(
        self,
        untemplated_text_iterator: Callable[[], Iterator[str]],
    ) -> None:
        self.seen_lines.extend(list(untemplated_text_iterator()))


def _labels() -> AnomalyLabelLookup:
    return label_lookup()


def _cache_paths() -> CachePathsConfig:
    base = Path.cwd() / "tests" / "_artifacts" / "structured_dataset"
    return CachePathsConfig(
        data_root=base / "data",
        cache_root=base / "cache",
    )


def test_structured_dataset_mine_templates_trains_parser_from_sink_rows() -> None:
    """mine_templates_with feeds untemplated messages into the parser."""
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=Path("raw.log"),
        parser=NullStructuredParser(),
        rows=[
            structured_line(
                line_order=0,
                timestamp_unix_ms=1_000,
                entity_id="node-a",
                untemplated_message_text="first",
                anomalous=None,
            ),
            structured_line(
                line_order=1,
                timestamp_unix_ms=1_200,
                entity_id="node-a",
                untemplated_message_text="second",
                anomalous=None,
            ),
        ],
    )
    parser = _RecordingTemplateParser(dataset_name="demo", seen_lines=[])
    dataset = StructuredDataset(
        sink=sink,
        cache_paths=_cache_paths(),
        anomaly_labels=_labels(),
    )

    templated = dataset.mine_templates_with(parser)

    assert parser.seen_lines == ["first", "second"]
    assert templated.sink is sink
    assert templated.template_parser is parser


def test_templated_dataset_grouping_helpers_configure_sequence_builder() -> None:
    """Grouping helpers delegate to the expected SequenceBuilder modes."""
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=Path("raw.log"),
        parser=NullStructuredParser(),
        rows=[],
    )
    parser = _RecordingTemplateParser(dataset_name="demo", seen_lines=[])
    templated = StructuredDataset(
        sink=sink,
        cache_paths=_cache_paths(),
        anomaly_labels=_labels(),
    ).mine_templates_with(parser)

    assert templated.group_by_entity().mode is GroupingMode.ENTITY
    assert (
        templated.group_by_fixed_window(WINDOW_SIZE, step_size=WINDOW_STEP).mode
        is GroupingMode.FIXED
    )
    assert (
        templated.group_by_fixed_window(WINDOW_SIZE, step_size=WINDOW_STEP).window_size
        == WINDOW_SIZE
    )
    assert (
        templated.group_by_fixed_window(WINDOW_SIZE, step_size=WINDOW_STEP).step
        == WINDOW_STEP
    )
    assert (
        templated.group_by_time_window(TIME_SPAN_MS, step_span_ms=TIME_STEP_MS).mode
        is GroupingMode.TIME
    )
    assert (
        templated.group_by_time_window(
            TIME_SPAN_MS,
            step_span_ms=TIME_STEP_MS,
        ).time_span_ms
        == TIME_SPAN_MS
    )
    assert (
        templated.group_by_time_window(TIME_SPAN_MS, step_span_ms=TIME_STEP_MS).step
        == TIME_STEP_MS
    )
