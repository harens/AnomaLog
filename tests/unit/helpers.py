"""Shared test helpers for unit tests."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import create_autospec

from prefect.context import TaskRunContext

from anomalog.labels import AnomalyLabelLookup
from anomalog.parsers.structured.contracts import (
    BaseStructuredLine,
    EntityLabelCounts,
    StructuredLine,
    StructuredParser,
    StructuredSink,
    is_anomalous_label,
)


def structured_line(
    *,
    line_order: int,
    timestamp_unix_ms: int | None,
    entity_id: str | None,
    untemplated_message_text: str,
    anomalous: int | None,
) -> StructuredLine:
    """Build a concise StructuredLine fixture."""
    return StructuredLine(
        line_order=line_order,
        timestamp_unix_ms=timestamp_unix_ms,
        entity_id=entity_id,
        untemplated_message_text=untemplated_message_text,
        anomalous=anomalous,
    )


def task_run_context() -> TaskRunContext:
    """TaskRunContext stub used to satisfy Prefect cache-policy method signatures."""
    return create_autospec(TaskRunContext, instance=True)


def label_lookup(
    *,
    label_for_line: Callable[[int], int | None] | None = None,
    label_for_group: Callable[[str], int | None] | None = None,
) -> AnomalyLabelLookup:
    """Build an AnomalyLabelLookup with overrideable callbacks."""
    return AnomalyLabelLookup(
        label_for_line=(lambda _: None) if label_for_line is None else label_for_line,
        label_for_group=((lambda _: 0) if label_for_group is None else label_for_group),
    )


@dataclass(frozen=True)
class NullStructuredParser(StructuredParser):
    """Minimal parser double for tests that only need sink wiring."""

    def parse_line(self, raw_line: str) -> BaseStructuredLine | None:
        """Discard all input lines."""
        del raw_line
        return None


@dataclass(frozen=True)
class InMemoryStructuredSink(StructuredSink):
    """StructuredSink backed by an in-memory list of rows for unit tests."""

    dataset_name: str
    raw_dataset_path: Path
    parser: StructuredParser
    rows: list[StructuredLine]
    anomalies_inline: bool | None = None

    def write_structured_lines(self) -> bool:
        """Report whether the sink should be treated as having inline labels."""
        if self.anomalies_inline is not None:
            return self.anomalies_inline
        return any(is_anomalous_label(row.anomalous) for row in self.rows)

    def iter_structured_lines(
        self,
        columns: Sequence[str] | None = None,
    ) -> Callable[[], Iterator[StructuredLine]]:
        """Yield stored rows, ignoring column projection."""
        del columns

        def _iter() -> Iterator[StructuredLine]:
            yield from self.rows

        return _iter

    def load_inline_label_cache(self) -> tuple[dict[int, int], dict[str, int]]:
        """Build sparse inline label lookups from the stored rows."""
        line_labels: dict[int, int] = {}
        group_labels: dict[str, int] = {}

        for row in self.rows:
            if not is_anomalous_label(row.anomalous):
                continue
            anomalous_label = row.anomalous
            if anomalous_label is None:
                continue
            line_labels[row.line_order] = anomalous_label
            if row.entity_id is not None:
                group_labels.setdefault(row.entity_id, anomalous_label)

        return line_labels, group_labels

    def count_rows(self) -> int:
        """Return the number of stored rows."""
        return len(self.rows)

    def count_entities_by_label(
        self,
        label_for_group: Callable[[str], int | None],
    ) -> EntityLabelCounts:
        """Count normal and total entities using the provided label lookup."""
        entity_ids = {row.entity_id for row in self.rows if row.entity_id is not None}
        normal = sum(
            not is_anomalous_label(label_for_group(entity_id))
            for entity_id in entity_ids
        )
        return EntityLabelCounts(
            normal_entities=normal,
            total_entities=len(entity_ids),
        )

    def timestamp_bounds(self) -> tuple[int | None, int | None]:
        """Return the minimum and maximum non-null timestamps."""
        timestamps = [
            row.timestamp_unix_ms
            for row in self.rows
            if row.timestamp_unix_ms is not None
        ]
        if not timestamps:
            return None, None
        return min(timestamps), max(timestamps)

    def iter_entity_sequences(
        self,
    ) -> Callable[[], Iterator[Sequence[StructuredLine]]]:
        """Group stored rows by entity id."""
        groups: dict[str, list[StructuredLine]] = {}
        for row in self.rows:
            if row.entity_id is not None:
                groups.setdefault(row.entity_id, []).append(row)

        def _iter() -> Iterator[Sequence[StructuredLine]]:
            yield from groups.values()

        return _iter

    def iter_fixed_window_sequences(
        self,
        window_size: int,
        step_size: int | None = None,
    ) -> Callable[[], Iterator[Sequence[StructuredLine]]]:
        """Yield sliding row windows over the stored rows."""
        step = window_size if step_size is None else step_size

        def _iter() -> Iterator[Sequence[StructuredLine]]:
            for start in range(0, len(self.rows), step):
                window = self.rows[start : start + window_size]
                if window:
                    yield window

        return _iter

    def iter_time_window_sequences(
        self,
        time_span_ms: int,
        step_span_ms: int | None = None,
    ) -> Callable[[], Iterator[Sequence[StructuredLine]]]:
        """Yield a single window containing all rows."""
        del time_span_ms, step_span_ms

        def _iter() -> Iterator[Sequence[StructuredLine]]:
            if self.rows:
                yield self.rows

        return _iter
