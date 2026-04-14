"""Helpers for loading anomaly labels from different sources."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from anomalog.parsers.structured.contracts import (
    ANOMALOUS_FIELD,
    ENTITY_FIELD,
    StructuredSink,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class AnomalyLabelLookup:
    """Lookup accessors for anomaly labels."""

    label_for_line: Callable[[int], int | None]
    label_for_group: Callable[[str], int | None]


@runtime_checkable
class AnomalyLabelReader(Protocol):
    """Loads anomaly label lookups."""

    def load(self) -> AnomalyLabelLookup:
        """Return callables that map line or group identifiers to labels."""

    def with_context(
        self,
        *,
        dataset_root: Path,
        sink: StructuredSink,
    ) -> AnomalyLabelReader:
        """Bind dataset context and return a configured reader instance."""


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
        """Load labels from the configured CSV file into lookup callables.

        Returns:
            AnomalyLabelLookup: Lookup functions backed by the configured CSV.

        Raises:
            ValueError: If dataset context is missing or the CSV schema is invalid.
        """
        if self.dataset_root is None:
            msg = "CSVReader.dataset_root was not set."
            raise ValueError(msg)

        path = self.dataset_root / self.relative_path
        group_labels: dict[str, int] = {}
        with path.open(newline="", encoding="utf-8") as file_obj:
            reader = csv.DictReader(file_obj)
            if not reader.fieldnames or self.entity_column not in reader.fieldnames:
                msg = (
                    f"CSV {path} must contain '{self.entity_column}' column for "
                    "group labels."
                )
                raise ValueError(msg)
            if self.label_column not in reader.fieldnames:
                msg = (
                    f"CSV {path} must contain '{self.label_column}' column for labels."
                )
                raise ValueError(msg)
            for row in reader:
                entity_raw = row.get(self.entity_column)
                label_raw = row.get(self.label_column)
                if entity_raw is None or label_raw is None:
                    continue
                try:
                    group_labels[str(entity_raw)] = int(label_raw)
                except (TypeError, ValueError):
                    continue

        def _label_for_group(entity_id: str) -> int | None:
            return group_labels.get(entity_id)

        return AnomalyLabelLookup(
            label_for_line=lambda _line_order: (
                None
            ),  # CSVReader does not supply per-line labels.
            label_for_group=_label_for_group,
        )

    def with_context(
        self,
        *,
        dataset_root: Path,
        sink: StructuredSink,
    ) -> CSVReader:
        """Attach dataset context when missing and return a new reader.

        Args:
            dataset_root (Path): Dataset root used to resolve the CSV path.
            sink (StructuredSink): Structured sink for the dataset. Unused for
                CSV-backed labels.

        Returns:
            CSVReader: Reader bound to the supplied dataset root.
        """
        del sink  # sink not used for CSV-based labels
        if self.dataset_root is not None:
            return self
        return replace(self, dataset_root=dataset_root)


@dataclass(frozen=True, slots=True)
class InlineReader(AnomalyLabelReader):
    """Derives labels directly from the structured sink."""

    sink: StructuredSink | None = None

    def load(self) -> AnomalyLabelLookup:
        """Collect inline labels from the sink and return lookup callables.

        Returns:
            AnomalyLabelLookup: Lookup functions backed by the structured sink.

        Raises:
            ValueError: If no structured sink has been attached.
            RuntimeError: If the sink fails while loading inline labels.
        """
        if self.sink is None:
            msg = "InlineReader requires a StructuredSink to read labels."
            raise ValueError(msg)

        try:
            line_labels, group_labels = self.sink.load_inline_label_cache()
        except Exception as exc:  # pragma: no cover
            msg = f"Failed to load inline labels from structured sink: {exc}"
            raise RuntimeError(msg) from exc

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
        dataset_root: Path,
        sink: StructuredSink,
    ) -> InlineReader:
        """Attach sink context when missing and return a new reader.

        Args:
            dataset_root (Path): Dataset root for the current build. Unused for
                inline labels.
            sink (StructuredSink): Structured sink that provides inline labels.

        Returns:
            InlineReader: Reader bound to the supplied structured sink.
        """
        del dataset_root  # not needed for inline reader
        if self.sink is not None:
            return self
        return replace(self, sink=sink)
