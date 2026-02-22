from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

from anomalog.anomaly_label_reader import AnomalyLabelLookup
from anomalog.cache import CachePathsConfig
from anomalog.models.sequences import SequenceBuilder
from anomalog.structured_parsers.contracts import StructuredSink

UntemplatedText: TypeAlias = str
LogTemplate: TypeAlias = str
ExtractedParameters: TypeAlias = Iterable[str]


# TODO: Add visualisation methods
@runtime_checkable
class TemplateParser(Protocol):
    dataset_name: str

    def inference(
        self,
        unstructured_text: UntemplatedText,
    ) -> tuple[LogTemplate, ExtractedParameters]: ...

    def train(
        self,
        untemplated_text_iterator: Callable[[], Iterator[UntemplatedText]],
    ) -> None: ...


@dataclass(slots=True, frozen=True)
class TemplatedDataset:
    sink: StructuredSink
    cache_paths: CachePathsConfig
    template_parser: TemplateParser
    anomaly_labels: AnomalyLabelLookup

    @property
    def sequence_builder(self) -> SequenceBuilder:
        return SequenceBuilder.from_dataset(self)

    def group_entity(self) -> SequenceBuilder:
        return self.sequence_builder.entity()

    def group_fixed_window(
        self,
        window_size: int,
        step_size: int | None = None,
    ) -> SequenceBuilder:
        return self.sequence_builder.fixed(size=window_size, step=step_size)

    def group_time_window(
        self,
        time_span_ms: int,
        step_span_ms: int | None = None,
    ) -> SequenceBuilder:
        return self.sequence_builder.time(
            span_ms=time_span_ms,
            step_ms=step_span_ms,
        )
