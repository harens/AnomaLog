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
    """Normalised access to anomaly labels.

    Both lookup functions return an integer label when one is available, or
    ``None`` when the source has no label for the requested row or group.

    Attributes:
        label_for_line (Callable[[int], int | None]): Returns the anomaly label
            for a structured row's stable `line_order`, or `None` when absent.
        label_for_group (Callable[[str], int | None]): Returns the anomaly label
            for a grouped entity identifier, or `None` when absent.
    """

    label_for_line: Callable[[int], int | None]
    label_for_group: Callable[[str], int | None]


@runtime_checkable
class AnomalyLabelReader(Protocol):
    """Protocol for sources that provide anomaly labels.

    Readers may be configured ahead of time, then bound to dataset-specific
    resources with ``with_context`` immediately before loading.
    """

    def load(self) -> AnomalyLabelLookup:
        """Materialise a normalised label lookup for the current dataset.

        Returns:
            AnomalyLabelLookup: Callables that expose label lookup by stable line
                order and by grouped entity identifier.
        """

    def with_context(
        self,
        *,
        dataset_root: Path,
        sink: StructuredSink,
    ) -> AnomalyLabelReader:
        """Bind dataset-specific runtime context to the reader.

        Args:
            dataset_root (Path): Materialised dataset root for path-relative label
                sources.
            sink (StructuredSink): Structured sink for readers that need direct
                access to parsed rows or sink-owned caches.

        Returns:
            AnomalyLabelReader: Reader instance ready to load labels for this
                concrete dataset build.
        """


@dataclass(frozen=True, slots=True)
class CSVReader(AnomalyLabelReader):
    """Reads anomaly labels from a CSV file (group/entity level only).

    CSV labels are intentionally group-scoped: they annotate entities/blocks
    rather than individual structured rows. Invalid or non-integer label values
    are skipped so malformed rows do not abort the whole dataset build.

    Attributes:
        relative_path (Path): CSV path relative to the materialised dataset root.
        dataset_root (Path | None): Bound dataset root used to resolve
            `relative_path` at runtime.
        entity_column (str): CSV column containing the group/entity identifier.
        label_column (str): CSV column containing the integer anomaly label.
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
    """Derives labels directly from the structured sink.

    This reader exists for datasets whose parser already exposes anomaly labels
    inline. It delegates to the sink so sink implementations can use efficient
    projected scans instead of forcing full row materialisation.

    Attributes:
        sink (StructuredSink | None): Bound sink that can supply sparse inline
            label lookups.
    """

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
