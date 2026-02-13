"""Helpers for loading anomaly labels from different sources.

We need to support:
    - Inline labels: already present in the structured sink (parser emitted them).
    - External labels: e.g. CSV files shipped alongside the dataset.

Two granularities are supported:
    - Per-line labels keyed by line_order (byte offset or sequential number).
    - Per-group labels keyed by entity_id (component/block/node/etc).
"""

import csv
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, runtime_checkable

from anomalog.structured_parsers.contracts import (
    ANOMALOUS_FIELD,
    ENTITY_FIELD,
    LINE_FIELD,
    StructuredSink,
)


@dataclass(frozen=True, slots=True)
class AnomalyLabelLookup:
    """Lookup accessors for anomaly labels."""

    label_for_line: Callable[[int], int | None]
    label_for_group: Callable[[str], int | None]


@runtime_checkable
class AnomalyLabelReader(Protocol):
    """Loads anomaly label lookups."""

    def load(self) -> AnomalyLabelLookup: ...

    def with_context(
        self,
        *,
        dataset_root: Path,
        sink: StructuredSink,
    ) -> "AnomalyLabelReader": ...


@dataclass(frozen=True, slots=True)
class CSVReader(AnomalyLabelReader):
    """Reads anomaly labels from a CSV file (group/entity level only).

    Column names are configurable: entity_column, label_column.
    """

    relative_path: Path
    dataset_root: Path | None = None
    entity_column: str = ENTITY_FIELD
    label_column: str = ANOMALOUS_FIELD

    def load(self) -> AnomalyLabelLookup:
        if self.dataset_root is None:
            msg = "CSVReader.dataset_root was not set."
            raise ValueError(msg)

        path = self.dataset_root / self.relative_path
        entity_col = self.entity_column
        label_col = self.label_column

        group_labels: dict[str, int] = {}
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or entity_col not in reader.fieldnames:
                msg = f"CSV {path} must contain '{entity_col}' column for group labels."
                raise ValueError(msg)
            if label_col not in reader.fieldnames:
                msg = f"CSV {path} must contain '{label_col}' column for labels."
                raise ValueError(msg)
            for row in reader:
                entity_raw = row.get(entity_col)
                label_raw = row.get(label_col)
                if entity_raw is None or label_raw is None:
                    continue
                try:
                    group_labels[str(entity_raw)] = int(label_raw)
                except (TypeError, ValueError):
                    continue

        def _label_for_group(entity_id: str) -> int | None:
            return group_labels.get(entity_id)

        def _label_for_line(line_order: int) -> int | None:  # noqa: ARG001
            return None  # CSVReader does not supply per-line labels.

        return AnomalyLabelLookup(
            label_for_line=_label_for_line,
            label_for_group=_label_for_group,
        )

    def with_context(
        self,
        *,
        dataset_root: Path,
        sink: StructuredSink,
    ) -> "CSVReader":
        _ = sink  # sink not used for CSV-based labels
        if self.dataset_root is not None:
            return self
        return replace(self, dataset_root=dataset_root)


@dataclass(frozen=True, slots=True)
class InlineReader(AnomalyLabelReader):
    """Derives labels directly from the structured sink (anomalous column)."""

    sink: StructuredSink | None = None

    def load(self) -> AnomalyLabelLookup:
        if self.sink is None:
            msg = "InlineReader requires a StructuredSink to read labels."
            raise ValueError(msg)
        sink: StructuredSink = self.sink

        line_labels, group_labels = _load_label_cache(sink)

        def _label_for_line(line_order: int) -> int | None:
            return line_labels.get(line_order)

        def _label_for_group(entity_id: str) -> int | None:
            return group_labels.get(entity_id)

        return AnomalyLabelLookup(
            label_for_line=_label_for_line,
            label_for_group=_label_for_group,
        )

    def with_context(
        self,
        *,
        dataset_root: Path,  # noqa: ARG002 - not needed for inline reader
        sink: StructuredSink,
    ) -> "InlineReader":
        if self.sink is not None:
            return self
        return replace(self, sink=sink)


def _load_label_cache(
    sink: StructuredSink,
) -> tuple[dict[int, int], dict[str, int]]:
    """Materialise label lookups from the sink, staying sparse.

    Strategy:
    - Scan the dataset once with column projection to id + label fields.
    - Store only rows where label is non-zero and not None (anomalies are rare).
    - No attachment to the sink; callers capture the dicts in AnomalyLabelLookup.
    """

    line_labels: dict[int, int] = {}
    group_labels: dict[str, int] = {}

    try:
        iter_rows = sink.iter_structured_lines(
            columns=[LINE_FIELD, ENTITY_FIELD, ANOMALOUS_FIELD],
        )()
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"Failed to iterate structured lines for labels: {exc}"
        raise RuntimeError(msg) from exc

    for row in iter_rows:
        raw_label = getattr(row, ANOMALOUS_FIELD, None)
        if raw_label is None:
            continue
        try:
            label = int(raw_label)
        except (TypeError, ValueError):
            continue
        if label == 0:
            continue  # keep cache sparse; missing -> treated as non-anomalous

        line_order = getattr(row, LINE_FIELD, None)
        if line_order is not None:
            line_labels[int(line_order)] = label

        entity_id = getattr(row, ENTITY_FIELD, None)
        if entity_id is not None and entity_id not in group_labels:
            group_labels[str(entity_id)] = label

    return line_labels, group_labels
