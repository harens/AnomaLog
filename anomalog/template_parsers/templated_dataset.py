from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

from anomalog.anomaly_label_reader import AnomalyLabelLookup
from anomalog.cache import CachePathsConfig
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
