from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

# Shared field names to avoid magic strings elsewhere.
LINE_FIELD = "line_order"
TIMESTAMP_FIELD = "timestamp_unix_ms"
ENTITY_FIELD = "entity_id"
UNTEMPLATED_FIELD = "untemplated_message_text"
ANOMALOUS_FIELD = "anomalous"


@dataclass(frozen=True, slots=True)
class BaseStructuredLine:
    timestamp_unix_ms: int | None
    entity_id: str | None
    untemplated_message_text: str
    anomalous: int | None  # 0/1/None (but also different anomalous categories)


@dataclass(frozen=True, slots=True)
class StructuredLine(BaseStructuredLine):
    line_order: int

    @classmethod
    def with_line_order(
        cls,
        *,
        line_order: int,
        base: BaseStructuredLine,
    ) -> "StructuredLine":
        return cls(
            timestamp_unix_ms=base.timestamp_unix_ms,
            entity_id=base.entity_id,
            untemplated_message_text=base.untemplated_message_text,
            anomalous=base.anomalous,
            line_order=line_order,
        )


@runtime_checkable
class StructuredParser(Protocol):
    def parse_line(
        self,
        raw_line: str,
    ) -> BaseStructuredLine | None: ...


# TODO: Add visualisation methods
@runtime_checkable
class StructuredSink(Protocol):
    dataset_name: str
    raw_dataset_path: Path
    parser: StructuredParser

    # Returns whether any of the lines have anomalous label 1 (as opposed to 0 or None).
    def write_structured_lines(self) -> bool: ...

    # Batched access to structured rows, returned as StructuredLine instances.
    def iter_structured_lines(
        self,
        columns: Sequence[str] | None = None,
    ) -> Callable[[], Iterator[StructuredLine]]: ...

    # Log Grouping Strategies
    def iter_entity_sequences(
        self,
    ) -> Callable[[], Iterator[Collection[StructuredLine]]]: ...

    def iter_fixed_window_sequences(
        self,
        window_size: int,
        step_size: int | None = None,  # defaults to window_size (non-overlapping)
    ) -> Callable[[], Iterator[Collection[StructuredLine]]]: ...

    def iter_time_window_sequences(
        self,
        time_span_ms: int,
        step_span_ms: int | None = None,  # defaults to time_span_ms (non-overlapping)
    ) -> Callable[[], Iterator[Collection[StructuredLine]]]: ...
