"""Shared test helpers for unit tests."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, TypedDict
from unittest.mock import create_autospec

import pyarrow.dataset as ds
from prefect.context import TaskRunContext
from typing_extensions import NotRequired, Unpack, override

from anomalog.labels import AnomalyLabelLookup
from anomalog.parsers.structured.contracts import (
    BaseStructuredLine,
    EntityLabelCounts,
    StructuredLine,
    StructuredParser,
    StructuredSink,
    is_anomalous_label,
)
from anomalog.parsers.structured.parquet.writer_worker import EntityChronologyKey


class _StructuredLineFields(TypedDict):
    timestamp_unix_ms: int | None
    entity_id: str | None
    untemplated_message_text: str
    anomalous: int | None
    raw_parameters: NotRequired[list[str] | None]


def structured_line(
    *,
    line_order: int,
    **fields: Unpack[_StructuredLineFields],
) -> StructuredLine:
    """Build a concise StructuredLine fixture.

    Args:
        line_order (int): Stable line order to assign to the fixture row.
        **fields (Unpack[_StructuredLineFields]): Field values used to
            construct the fixture row. Expected keys are `timestamp_unix_ms`,
            `entity_id`,
            `untemplated_message_text`, `anomalous`, and optionally
            `raw_parameters`.

    Returns:
        StructuredLine: Structured row with the supplied field values.
    """
    return StructuredLine(
        line_order=line_order,
        timestamp_unix_ms=fields["timestamp_unix_ms"],
        entity_id=fields["entity_id"],
        untemplated_message_text=fields["untemplated_message_text"],
        anomalous=fields["anomalous"],
        raw_parameters=fields.get("raw_parameters"),
    )


def task_run_context() -> TaskRunContext:
    """TaskRunContext stub used to satisfy Prefect cache-policy method signatures.

    Returns:
        TaskRunContext: Autospecced task-run context double.
    """
    context = create_autospec(TaskRunContext, instance=True)
    context.task = SimpleNamespace(source_code="task()")
    return context


def label_lookup(
    *,
    label_for_line: Callable[[int], int | None] | None = None,
    label_for_group: Callable[[str], int | None] | None = None,
) -> AnomalyLabelLookup:
    """Build an AnomalyLabelLookup with overrideable callbacks.

    Args:
        label_for_line (Callable[[int], int | None] | None): Optional callback
            for line-level labels.
        label_for_group (Callable[[str], int | None] | None): Optional callback
            for group-level labels.

    Returns:
        AnomalyLabelLookup: Lookup with overridable line and group callbacks.
    """
    return AnomalyLabelLookup(
        label_for_line=(lambda _: None) if label_for_line is None else label_for_line,
        label_for_group=((lambda _: 0) if label_for_group is None else label_for_group),
    )


@dataclass(frozen=True)
class NullStructuredParser(StructuredParser):
    """Minimal parser double for tests that only need sink wiring.

    Attributes:
        name (ClassVar[str]): Stable parser name for registry-compatible tests.
    """

    name: ClassVar[str] = "null"

    @override
    def parse_line(self, raw_line: str) -> BaseStructuredLine | None:
        """Discard all input lines.

        Args:
            raw_line (str): Raw line to discard.

        Returns:
            BaseStructuredLine | None: Always `None`.
        """
        del raw_line
        return None


@dataclass(frozen=True)
class _EntityGroup:
    """Test helper pairing an entity's rows with its order key.

    Attributes:
        order (EntityChronologyKey): Deterministic sort key for the entity.
        rows (tuple[StructuredLine, ...]): Structured rows belonging to the
            entity group.
    """

    order: EntityChronologyKey
    rows: tuple[StructuredLine, ...]


@dataclass(frozen=True)
class InMemoryStructuredSink(StructuredSink):
    """StructuredSink backed by an in-memory list of rows for unit tests.

    Attributes:
        dataset_name (str): Dataset identifier exposed through the sink contract.
        raw_dataset_path (Path): Synthetic raw dataset path carried for contract
            compatibility.
        parser (StructuredParser): Parser instance associated with the sink.
        rows (list[StructuredLine]): Stored structured rows returned by the sink.
        anomalies_inline (bool | None): Optional override for whether the sink
            should report inline anomalies without recalculating from rows.
    """

    dataset_name: str
    raw_dataset_path: Path
    parser: StructuredParser
    rows: list[StructuredLine]
    _structured_cache_path: Path | None = None
    anomalies_inline: bool | None = None

    def structured_cache_path(self) -> Path | None:
        """Return the optional synthetic structured cache path for tests.

        Returns:
            Path | None: Synthetic structured cache path supplied by the test,
                or `None` when the helper should behave like an in-memory sink.
        """
        return self._structured_cache_path

    def write_structured_lines(self) -> bool:
        """Report whether the sink should be treated as having inline labels.

        Returns:
            bool: Whether any stored row carries an inline anomaly label field.
        """
        if self.anomalies_inline is not None:
            return self.anomalies_inline
        return any(row.anomalous is not None for row in self.rows)

    def iter_structured_lines(
        self,
        columns: Sequence[str] | None = None,
    ) -> Callable[[], Iterator[StructuredLine]]:
        """Yield stored rows, ignoring column projection.

        Args:
            columns (Sequence[str] | None): Ignored projected columns request.

        Returns:
            Callable[[], Iterator[StructuredLine]]: Callable yielding stored rows.
        """
        del columns

        def _iter() -> Iterator[StructuredLine]:
            yield from self.rows

        return _iter

    def iter_structured_lines_in_source_order(
        self,
        filter_expr: ds.Expression | None = None,
    ) -> Callable[[], Iterator[StructuredLine]]:
        """Yield stored rows ordered by their raw line order.

        Args:
            filter_expr (ds.Expression | None): Ignored filter expression used to keep
                the test helper signature aligned with sink implementations.

        Returns:
            Callable[[], Iterator[StructuredLine]]: Zero-argument callable that
                yields stored rows ordered by `line_order`.
        """
        del filter_expr

        def _iter() -> Iterator[StructuredLine]:
            yield from sorted(self.rows, key=lambda row: row.line_order)

        return _iter

    def iter_entity_sequences_from_line_order(
        self,
        min_line_order: int,
    ) -> Callable[[], Iterator[Sequence[StructuredLine]]]:
        """Yield suffix groups from stored rows at or after a line-order cut.

        Args:
            min_line_order (int): Inclusive lower bound on the raw line order
                used to build the suffix grouping.

        Returns:
            Callable[[], Iterator[Sequence[StructuredLine]]]: Zero-argument
                callable yielding grouped suffix rows by entity.
        """
        groups: dict[str, list[StructuredLine]] = {}
        for row in self.rows:
            if row.line_order < min_line_order or row.entity_id is None:
                continue
            groups.setdefault(row.entity_id, []).append(row)

        def _iter() -> Iterator[Sequence[StructuredLine]]:
            for rows in groups.values():
                yield tuple(rows)

        return _iter

    def load_inline_label_cache(self) -> tuple[dict[int, int], dict[str, int]]:
        """Build sparse inline label lookups from the stored rows.

        Returns:
            tuple[dict[int, int], dict[str, int]]: Sparse per-line and per-group
                anomaly labels.
        """
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
        """Return the number of stored rows.

        Returns:
            int: Number of rows currently stored in the sink.
        """
        return len(self.rows)

    def count_entities_by_label(
        self,
        label_for_group: Callable[[str], int | None],
    ) -> EntityLabelCounts:
        """Count normal and total entities using the provided label lookup.

        Args:
            label_for_group (Callable[[str], int | None]): Entity label lookup.

        Returns:
            EntityLabelCounts: Distinct normal and total entity counts.
        """
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
        """Return the minimum and maximum non-null timestamps.

        Returns:
            tuple[int | None, int | None]: Minimum and maximum timestamp across
                stored rows, or `(None, None)` when no timestamps are present.
        """
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
        """Group stored rows by entity id in chronological order.

        Returns:
            Callable[[], Iterator[Sequence[StructuredLine]]]: Callable yielding
                entity-grouped rows ordered by each entity's first timestamp.
        """
        groups: dict[str, list[StructuredLine]] = {}
        for row in self.rows:
            if row.entity_id is not None:
                groups.setdefault(row.entity_id, []).append(row)

        def _iter() -> Iterator[Sequence[StructuredLine]]:
            ordered_groups = sorted(
                (
                    _EntityGroup(
                        order=_entity_group_order(entity_id, rows),
                        rows=tuple(rows),
                    )
                    for entity_id, rows in groups.items()
                ),
                key=lambda group: group.order,
            )
            for group in ordered_groups:
                yield group.rows

        return _iter

    def iter_fixed_window_sequences(
        self,
        window_size: int,
        step_size: int | None = None,
    ) -> Callable[[], Iterator[Sequence[StructuredLine]]]:
        """Yield sliding row windows over the stored rows.

        Args:
            window_size (int): Number of rows in each emitted window.
            step_size (int | None): Optional step between successive windows.

        Returns:
            Callable[[], Iterator[Sequence[StructuredLine]]]: Callable yielding
                fixed-size windows over stored rows.
        """
        step = window_size if step_size is None else step_size

        def _iter() -> Iterator[Sequence[StructuredLine]]:
            if window_size <= 0 or step <= 0:
                return
            if len(self.rows) < window_size:
                return
            for start in range(0, len(self.rows) - window_size + 1, step):
                window = self.rows[start : start + window_size]
                if len(window) == window_size:
                    yield window

        return _iter

    def iter_time_window_sequences(
        self,
        time_span_ms: int,
        step_span_ms: int | None = None,
    ) -> Callable[[], Iterator[Sequence[StructuredLine]]]:
        """Yield a single window containing all rows.

        Args:
            time_span_ms (int): Requested time span in milliseconds. Ignored.
            step_span_ms (int | None): Requested step span in milliseconds. Ignored.

        Returns:
            Callable[[], Iterator[Sequence[StructuredLine]]]: Callable yielding a
                single all-rows window when rows exist.
        """
        del time_span_ms, step_span_ms

        def _iter() -> Iterator[Sequence[StructuredLine]]:
            if self.rows:
                yield self.rows

        return _iter


def _entity_group_order(
    entity_id: str,
    rows: Sequence[StructuredLine],
) -> EntityChronologyKey:
    """Build the deterministic order key used for entity-grouped tests.

    Args:
        entity_id (str): Entity identifier for the grouped rows.
        rows (Sequence[StructuredLine]): Structured rows belonging to one
            entity.

    Returns:
        EntityChronologyKey: Chronological ordering metadata for one entity.
    """
    first_row = rows[0]
    first_timestamp = next(
        (row.timestamp_unix_ms for row in rows if row.timestamp_unix_ms is not None),
        None,
    )
    return EntityChronologyKey(
        first_timestamp_missing=1 if first_timestamp is None else 0,
        first_timestamp_unix_ms=0 if first_timestamp is None else int(first_timestamp),
        first_line_order=int(first_row.line_order),
        entity_id=entity_id,
    )
