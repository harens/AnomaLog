"""Contracts for converting raw log lines into structured records."""

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
    """Minimal structured representation of a parsed log line."""

    timestamp_unix_ms: int | None
    entity_id: str | None
    untemplated_message_text: str
    anomalous: int | None  # 0/1/None (but also different anomalous categories)


@dataclass(frozen=True, slots=True)
class StructuredLine(BaseStructuredLine):
    """Structured line with a deterministic ordering attribute."""

    line_order: int

    @classmethod
    def with_line_order(
        cls,
        *,
        line_order: int,
        base: BaseStructuredLine,
    ) -> "StructuredLine":
        """Create a StructuredLine by adding line_order to a base record.

        >>> base = BaseStructuredLine(None, "node1", "msg", None)
        >>> StructuredLine.with_line_order(line_order=5, base=base).line_order
        5
        """
        return cls(
            timestamp_unix_ms=base.timestamp_unix_ms,
            entity_id=base.entity_id,
            untemplated_message_text=base.untemplated_message_text,
            anomalous=base.anomalous,
            line_order=line_order,
        )


@runtime_checkable
class StructuredParser(Protocol):
    """Interface for parsing raw log lines into structured records."""

    def parse_line(
        self,
        raw_line: str,
    ) -> BaseStructuredLine | None:
        """Parse raw_line into a BaseStructuredLine or return None to skip."""


# TODO: Add visualisation methods
@runtime_checkable
class StructuredSink(Protocol):
    """Interface for storing and iterating structured log records."""

    dataset_name: str
    raw_dataset_path: Path
    parser: StructuredParser

    # Returns whether any of the lines have anomalous label 1 (as opposed to 0 or None).
    def write_structured_lines(self) -> bool:
        """Persist structured lines and return True if anomalies were found."""

    # Batched access to structured rows, returned as StructuredLine instances.
    def iter_structured_lines(
        self,
        columns: Sequence[str] | None = None,
    ) -> Callable[[], Iterator[StructuredLine]]:
        """Return a callable yielding StructuredLine rows, optionally projected."""

    # Dataset statistics / bounds
    def count_rows(self) -> int:
        """Count total rows in the dataset."""

    def count_entities_by_label(
        self,
        label_for_group: Callable[[str], int | None],
    ) -> tuple[int, int]:
        """Return counts of (normal, total) distinct entity IDs."""

    def timestamp_bounds(self) -> tuple[int | None, int | None]:
        """Return (min_ts, max_ts) in milliseconds if available."""

    # Log Grouping Strategies
    def iter_entity_sequences(
        self,
    ) -> Callable[[], Iterator[Collection[StructuredLine]]]:
        """Yield sequences grouped by entity."""

    def iter_fixed_window_sequences(
        self,
        window_size: int,
        step_size: int | None = None,  # defaults to window_size (non-overlapping)
    ) -> Callable[[], Iterator[Collection[StructuredLine]]]:
        """Yield sequences grouped by fixed window sizes."""

    def iter_time_window_sequences(
        self,
        time_span_ms: int,
        step_span_ms: int | None = None,  # defaults to time_span_ms (non-overlapping)
    ) -> Callable[[], Iterator[Collection[StructuredLine]]]:
        """Yield sequences grouped by sliding time windows."""
