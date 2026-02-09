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

    def write_structured_lines(self) -> None: ...

    def read_unstructured_free_text(self) -> Callable[[], Iterator[str]]: ...
