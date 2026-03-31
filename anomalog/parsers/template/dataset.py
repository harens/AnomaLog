"""Abstractions for templated datasets and template parser protocols."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias

from anomalog.cache import CachePathsConfig
from anomalog.parsers.structured.contracts import StructuredSink
from anomalog.sequences import SequenceBuilder

if TYPE_CHECKING:
    from anomalog.labels import AnomalyLabelLookup

UntemplatedText: TypeAlias = str
LogTemplate: TypeAlias = str
ExtractedParameters: TypeAlias = Iterable[str]


# TODO: Add visualisation methods
class TemplateParser(Protocol):
    """Protocol describing template mining and inference behaviour."""

    dataset_name: str | None

    def __init__(self, dataset_name: str | None = None) -> None:
        """Initialise the parser with an optional bound dataset name."""

    def inference(
        self,
        unstructured_text: UntemplatedText,
    ) -> tuple[LogTemplate, ExtractedParameters]:
        """Return (template, parameters) for an unstructured log line."""

    def train(
        self,
        untemplated_text_iterator: Callable[[], Iterator[UntemplatedText]],
    ) -> None:
        """Train the parser using an iterator over untemplated text."""


@dataclass(slots=True, frozen=True)
class TemplatedDataset:
    """Structured dataset paired with a trained template parser and labels."""

    sink: StructuredSink
    cache_paths: CachePathsConfig
    template_parser: TemplateParser
    anomaly_labels: "AnomalyLabelLookup"

    @property
    def sequence_builder(self) -> SequenceBuilder:
        """Return a SequenceBuilder configured from this dataset."""
        return SequenceBuilder.from_dataset(self)

    def group_by_entity(self) -> SequenceBuilder:
        """Group sequences by entity id."""
        return self.sequence_builder.entity()

    def group_by_fixed_window(
        self,
        window_size: int,
        step_size: int | None = None,
    ) -> SequenceBuilder:
        """Group sequences in fixed-size windows."""
        return self.sequence_builder.fixed(size=window_size, step=step_size)

    def group_by_time_window(
        self,
        time_span_ms: int,
        step_span_ms: int | None = None,
    ) -> SequenceBuilder:
        """Group sequences using time-based sliding windows."""
        return self.sequence_builder.time(
            span_ms=time_span_ms,
            step_ms=step_span_ms,
        )
