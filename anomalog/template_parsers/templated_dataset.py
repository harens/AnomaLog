from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from anomalog.anomaly_label_reader import AnomalyLabelLookup
from anomalog.cache import CachePathsConfig
from anomalog.models.naive_bayes import NBConfig, run_naive_bayes
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

    def group_entity(self) -> "GroupedDatasetView":
        return GroupedDatasetView(self, mode="entity")

    def group_fixed_window(
        self,
        window_size: int,
        step_size: int | None = None,
    ) -> "GroupedDatasetView":
        return GroupedDatasetView(
            self,
            mode="fixed",
            window_size=window_size,
            step=step_size,
        )

    def group_time_window(
        self,
        time_span_ms: int,
        step_span_ms: int | None = None,
    ) -> "GroupedDatasetView":
        return GroupedDatasetView(
            self,
            mode="time",
            time_span_ms=time_span_ms,
            step=step_span_ms,
        )


@dataclass(slots=True, frozen=True)
class GroupedDatasetView:
    dataset: TemplatedDataset
    mode: Literal["entity", "fixed", "time"]
    window_size: int | None = None
    time_span_ms: int | None = None
    step: int | None = None

    @property
    def sequence_builder(self) -> SequenceBuilder:
        return SequenceBuilder(
            sink=self.dataset.sink,
            infer=self.dataset.template_parser.inference,
            label_for_group=self.dataset.anomaly_labels.label_for_group,
            mode=self.mode,
            window_size=self.window_size,
            time_span_ms=self.time_span_ms,
            step=self.step,
        )

    def naive_bayes(self, config: NBConfig | None = None) -> dict:
        cfg = config or NBConfig()
        cfg = replace(cfg, mode=self.mode)

        return run_naive_bayes(self.sequence_builder, cfg)
