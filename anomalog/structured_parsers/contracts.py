from collections.abc import Callable, Iterator, Sequence
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
class StructuredLine:
    line_order: int
    timestamp_unix_ms: int | None
    entity_id: str | None
    untemplated_message_text: str
    anomalous: int | None  # 0/1/None (but also different anomalous categories)


@runtime_checkable
class StructuredParser(Protocol):
    # line_order: e.g. byte offset or line number
    def parse_line(self, raw_line: str, line_order: int) -> StructuredLine | None: ...


# TODO: Add visualisation methods
@runtime_checkable
class StructuredSink(Protocol):
    dataset_name: str
    raw_dataset_path: Path
    parser: StructuredParser

    # Returns whether any of the lines have anomalous label 1 (as opposed to 0 or None).
    def write_structured_lines(self) -> bool: ...

    def read_unstructured_free_text(self) -> Callable[[], Iterator[str]]: ...

    # Batched access to structured rows, returned as StructuredLine instances.
    def iter_structured_lines(
        self,
        columns: Sequence[str] | None = None,
    ) -> Callable[[], Iterator[StructuredLine]]: ...

    # Optional fast-path lookups for anomaly labels. Return None if unknown.
    def label_for_line(self, line_order: int) -> int | None: ...
    def label_for_group(self, entity_id: str) -> int | None: ...
