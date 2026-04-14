"""Abstractions for templated datasets and template parser contracts."""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeAlias, runtime_checkable

from anomalog.cache import CachePathsConfig
from anomalog.parsers.structured.contracts import StructuredSink
from anomalog.sequences import (
    EntitySequenceBuilder,
    FixedSequenceBuilder,
    TimeSequenceBuilder,
)

if TYPE_CHECKING:
    from anomalog.labels import AnomalyLabelLookup

UntemplatedText: TypeAlias = str
LogTemplate: TypeAlias = str
ExtractedParameters: TypeAlias = Iterable[str]


# TODO: Add visualisation methods
@runtime_checkable
class TemplateParser(Protocol):
    """Interface for template mining implementations.

    Implementations are initialised with an optional dataset name so runtime
    caches can be scoped per dataset when needed.
    """

    name: ClassVar[str]
    dataset_name: str | None

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

    def sequences(self) -> EntitySequenceBuilder:
        """Return the default entity-grouped sequence builder.

        Returns:
            EntitySequenceBuilder: Default entity-grouped sequence view.
        """
        return EntitySequenceBuilder.from_dataset(self)

    def group_by_entity(self) -> EntitySequenceBuilder:
        """Group sequences by entity id.

        Returns:
            EntitySequenceBuilder: Entity-grouped sequence view.
        """
        return self.sequences()

    def group_by_fixed_window(
        self,
        window_size: int,
        step_size: int | None = None,
    ) -> FixedSequenceBuilder:
        """Group sequences in fixed-size windows.

        Args:
            window_size (int): Number of rows in each emitted window.
            step_size (int | None): Optional step between successive windows.
                Defaults to `window_size`.

        Returns:
            FixedSequenceBuilder: Fixed-window sequence view.
        """
        return FixedSequenceBuilder(
            sink=self.sink,
            infer_template=self.template_parser.inference,
            label_for_group=self.anomaly_labels.label_for_group,
            window_size=window_size,
            step=step_size,
        )

    def group_by_time_window(
        self,
        time_span_ms: int,
        step_span_ms: int | None = None,
    ) -> TimeSequenceBuilder:
        """Group sequences using time-based sliding windows.

        Args:
            time_span_ms (int): Width of each emitted time window in
                milliseconds.
            step_span_ms (int | None): Optional step between successive windows.
                Defaults to `time_span_ms`.

        Returns:
            TimeSequenceBuilder: Time-window sequence view.
        """
        return TimeSequenceBuilder(
            sink=self.sink,
            infer_template=self.template_parser.inference,
            label_for_group=self.anomaly_labels.label_for_group,
            time_span_ms=time_span_ms,
            step=step_span_ms,
        )
