"""Contracts for converting raw log lines into structured records."""

from collections.abc import Callable, Collection, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, NamedTuple, Protocol, runtime_checkable

# Shared field names to avoid magic strings elsewhere.
LINE_FIELD = "line_order"
TIMESTAMP_FIELD = "timestamp_unix_ms"
ENTITY_FIELD = "entity_id"
UNTEMPLATED_FIELD = "untemplated_message_text"
ANOMALOUS_FIELD = "anomalous"


def is_anomalous_label(label: int | None) -> bool:
    """Return whether a label should be treated as anomalous.

    AnomaLog treats `0` as normal and any non-zero integer as anomalous.
    `None` means the label is absent.

    Args:
        label (int | None): Dataset-provided anomaly label.

    Returns:
        bool: `True` when the label represents an anomaly under AnomaLog's shared
            label convention.
    """
    return label is not None and label != 0


class EntityLabelCounts(NamedTuple):
    """Distinct entity counts partitioned by anomaly label.

    Attributes:
        normal_entities (int): Number of distinct entities whose resolved group
            label is normal or missing.
        total_entities (int): Total number of distinct entities considered.
    """

    normal_entities: int
    total_entities: int


@dataclass(frozen=True, slots=True)
class BaseStructuredLine:
    """Minimal structured representation of a parsed log line.

    These fields form the parser-to-sink contract. Every downstream sequence,
    label, and template stage assumes `untemplated_message_text` is preserved
    exactly as the template parser should see it, while timestamps and entity
    ids may be absent for datasets that do not provide them.

    Attributes:
        timestamp_unix_ms (int | None): Event timestamp in Unix milliseconds, or
            `None` when the source does not provide a reliable timestamp.
        entity_id (str | None): Entity or session identifier for entity-based
            grouping, or `None` when unavailable.
        untemplated_message_text (str): Message text forwarded to template
            mining without template substitution applied.
        anomalous (int | None): Dataset-provided anomaly label. `0` and `None`
            are treated as normal-or-missing; any non-zero value is anomalous.
    """

    timestamp_unix_ms: int | None
    entity_id: str | None
    untemplated_message_text: str
    anomalous: int | None  # 0/None normal-or-missing, any non-zero value anomalous


@dataclass(frozen=True, slots=True)
class StructuredLine(BaseStructuredLine):
    """Structured line with a deterministic ordering attribute.

    `line_order` preserves source-file order even when sinks repartition data.
    Downstream label lookups and deterministic grouping rely on it staying
    stable for a given materialised dataset.

    Attributes:
        line_order (int): Stable zero-based ordering key for the parsed record.
    """

    line_order: int

    @classmethod
    def with_line_order(
        cls,
        *,
        line_order: int,
        base: BaseStructuredLine,
    ) -> "StructuredLine":
        """Create a StructuredLine by adding line_order to a base record.

        Args:
            line_order (int): Stable line number to attach to the record.
            base (BaseStructuredLine): Parsed structured record without ordering.

        Examples:
            >>> base = BaseStructuredLine(None, "node1", "msg", None)
            >>> StructuredLine.with_line_order(line_order=5, base=base).line_order
            5

        Returns:
            StructuredLine: Structured record with the supplied `line_order`.
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
    """Interface for parsing raw log lines into structured records.

    Parsers define the dataset-specific semantics of a raw log line. Returning
    `None` is the supported way to skip malformed or irrelevant input without
    aborting the full parse.

    Attributes:
        name (ClassVar[str]): Stable registry/config name for the parser.
    """

    name: ClassVar[str]

    def parse_line(
        self,
        raw_line: str,
    ) -> BaseStructuredLine | None:
        """Parse one raw log line into the shared structured representation.

        Args:
            raw_line (str): Original line text from the raw dataset file.

        Returns:
            BaseStructuredLine | None: Parsed structured record, or `None` when
                the line should be skipped.
        """


# TODO: Add visualisation methods
@runtime_checkable
class StructuredSink(Protocol):
    """Interface for storing and iterating structured log records.

    Sinks own the durable structured representation and the canonical grouping
    operations built on top of it. This keeps dataset parsing separate from the
    storage/query strategy used later by sequence builders and label readers.

    Attributes:
        dataset_name (str): Dataset identifier used to scope sink-owned cache
            paths and artifacts.
        raw_dataset_path (Path): Materialsed raw log file parsed by the sink.
        parser (StructuredParser): Parser used to produce structured records.
    """

    dataset_name: str
    raw_dataset_path: Path
    parser: StructuredParser

    # Returns whether any line has a non-zero anomalous label.
    def write_structured_lines(self) -> bool:
        """Persist structured lines and report whether inline anomalies exist.

        Returns:
            bool: `True` when at least one persisted row carries a non-zero
                inline anomaly label.
        """

    # Batched access to structured rows, returned as StructuredLine instances.
    def iter_structured_lines(
        self,
        columns: Sequence[str] | None = None,
    ) -> Callable[[], Iterator[StructuredLine]]:
        """Return a lazy iterator factory over structured rows.

        Args:
            columns (Sequence[str] | None): Optional projected column subset. Sink
                implementations may use this to avoid reading unused fields while
                still returning full `StructuredLine` objects with defaults.

        Returns:
            Callable[[], Iterator[StructuredLine]]: Zero-argument callable that
                yields structured rows when invoked.
        """

    def iter_structured_lines_in_source_order(
        self,
    ) -> Callable[[], Iterator[StructuredLine]]:
        """Return structured rows in the original raw-entry order.

        Returns:
            Callable[[], Iterator[StructuredLine]]: Zero-argument callable that
                yields structured rows ordered by `line_order`.
        """

    def load_inline_label_cache(self) -> tuple[dict[int, int], dict[str, int]]:
        """Return sparse per-line and per-group inline anomaly labels.

        This exists alongside `iter_structured_lines()` because some sinks can
        build these sparse maps much more efficiently from projected columns or
        batches than by materialising full `StructuredLine` objects row-by-row.

        Returns:
            tuple[dict[int, int], dict[str, int]]: Sparse label maps keyed by
                `line_order` and `entity_id`, respectively.
        """

    # Dataset statistics / bounds
    def count_rows(self) -> int:
        """Count total rows in the dataset.

        Returns:
            int: Number of persisted structured rows.
        """

    def count_entities_by_label(
        self,
        label_for_group: Callable[[str], int | None],
    ) -> EntityLabelCounts:
        """Count distinct entities under the caller's label semantics.

        Args:
            label_for_group (Callable[[str], int | None]): Callback that resolves
                the anomaly label for an entity/group identifier.

        Returns:
            EntityLabelCounts: Normal and total distinct entity counts.
        """

    def timestamp_bounds(self) -> tuple[int | None, int | None]:
        """Return global timestamp bounds for the dataset.

        Returns:
            tuple[int | None, int | None]: Minimum and maximum timestamp in Unix
                milliseconds, or `(None, None)` when no timestamps exist.
        """

    # Log Grouping Strategies
    def iter_entity_sequences(
        self,
    ) -> Callable[[], Iterator[Collection[StructuredLine]]]:
        """Return a lazy iterator factory for chronological entity groups.

        Returns:
            Callable[[], Iterator[Collection[StructuredLine]]]: Callable yielding
                per-entity groups of structured rows ordered by each entity's
                first timestamp, with deterministic tie-breakers.
        """

    def iter_fixed_window_sequences(
        self,
        window_size: int,
        step_size: int | None = None,  # defaults to window_size (non-overlapping)
    ) -> Callable[[], Iterator[Collection[StructuredLine]]]:
        """Return a lazy iterator factory for fixed-size row windows.

        Args:
            window_size (int): Number of rows per emitted window.
            step_size (int | None): Window advance in rows. `None` means
                non-overlapping windows of size `window_size`.

        Returns:
            Callable[[], Iterator[Collection[StructuredLine]]]: Callable yielding
                fixed-size row windows in the sink's canonical order.
        """

    def iter_time_window_sequences(
        self,
        time_span_ms: int,
        step_span_ms: int | None = None,  # defaults to time_span_ms (non-overlapping)
    ) -> Callable[[], Iterator[Collection[StructuredLine]]]:
        """Return a lazy iterator factory for time-based windows.

        Args:
            time_span_ms (int): Duration of each window in milliseconds.
            step_span_ms (int | None): Window advance in milliseconds. `None`
                means non-overlapping windows of width `time_span_ms`.

        Returns:
            Callable[[], Iterator[Collection[StructuredLine]]]: Callable yielding
                sliding windows ordered by event timestamp.
        """
