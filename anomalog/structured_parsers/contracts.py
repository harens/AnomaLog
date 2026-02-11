from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class StructuredLine:
    line_order: int
    timestamp_unix_ms: int | None
    entity_id: str | None
    untemplated_message_text: str
    anomalous: int | None  # 0/1/None


@runtime_checkable
class StructuredParser(Protocol):
    # line_order: e.g. byte offset or line number
    def parse_line(self, raw_line: str, line_order: int) -> StructuredLine | None: ...


@runtime_checkable
class StructuredSink(Protocol):
    dataset_name: str
    raw_dataset_path: Path
    parser: StructuredParser

    # Returns whether any of the lines have anomalous label 1 (as opposed to 0 or None).
    def write_structured_lines(self) -> bool: ...

    def read_unstructured_free_text(self) -> Callable[[], Iterator[str]]: ...
