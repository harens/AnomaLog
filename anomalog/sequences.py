"""Utilities for building template sequences from structured log lines.

The module groups parsed log lines into windows (entity, fixed-size, or
time-based) and decorates them with inferred templates and anomaly labels.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

from typing_extensions import Self, override

from anomalog.parsers.structured.contracts import is_anomalous_label
from anomalog.representations import (
    SequenceRepresentation,
    SequenceRepresentationView,
    TRepresentation,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Iterator

    from anomalog.parsers.structured.contracts import StructuredLine, StructuredSink
    from anomalog.parsers.template.dataset import (
        ExtractedParameters,
        LogTemplate,
        TemplatedDataset,
    )


class GroupingMode(str, Enum):
    """Strategy for grouping structured lines into sequences."""

    ENTITY = "entity"
    FIXED = "fixed"
    TIME = "time"


class SplitLabel(str, Enum):
    """Dataset split membership for a sequence."""

    TRAIN = "train"
    TEST = "test"


@dataclass(slots=True, frozen=True)
class SequenceSplitSummary:
    """Serializable summary of requested versus effective split behavior."""

    requested_train_fraction: float
    train_fraction_scope: str
    train_on_normal_entities_only: bool
    eligible_train_sequence_count: int
    effective_train_fraction_of_eligible: float
    effective_train_fraction_overall: float

    def as_dict(self) -> dict[str, int | float | bool | str]:
        """Return a stable JSON-friendly representation.

        Returns:
            dict[str, int | float | bool | str]: Serialized split summary.
        """
        return {
            "requested_train_fraction": self.requested_train_fraction,
            "train_fraction_scope": self.train_fraction_scope,
            "train_on_normal_entities_only": self.train_on_normal_entities_only,
            "eligible_train_sequence_count": self.eligible_train_sequence_count,
            "effective_train_fraction_of_eligible": (
                self.effective_train_fraction_of_eligible
            ),
            "effective_train_fraction_overall": self.effective_train_fraction_overall,
        }


@dataclass(slots=True)
class TemplateSequence:
    """Grouped log window before any model-specific representation is applied.

    This keeps sequence semantics such as event ordering, labels, and entity
    membership. Model inputs derived from it live in `SequenceSample`.
    """

    events: list[
        tuple[str, list[str], int | None]
    ]  # (template, parameters, dt_prev_ms)
    label: int
    entity_ids: list[str]  # unique entity ids present (may be empty)
    window_id: int
    split_label: SplitLabel = SplitLabel.TRAIN

    @property
    def templates(self) -> list[str]:
        """Return the ordered template strings for this sequence."""
        return [tpl for tpl, _, _ in self.events]

    @property
    def sole_entity_id(self) -> str | None:
        """Return the entity id when the sequence belongs to exactly one entity.

        If multiple entities appear in the window, None is returned to avoid
        implying a single owning entity.
        """
        if len(self.entity_ids) == 1:
            return self.entity_ids[0]
        return None


@dataclass(slots=True, frozen=True)
class SequenceBuilder:
    """Common sequence-building behavior shared across grouping strategies."""

    sink: StructuredSink
    infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]]
    label_for_group: Callable[[str], int | None]
    train_frac: float = 0.8

    @property
    def mode(self) -> GroupingMode:
        """Return the grouping strategy for this builder."""
        msg = f"{type(self).__name__} must define a grouping mode."
        raise NotImplementedError(msg)

    def eligible_train_sequence_count(
        self,
        *,
        sequence_count: int,
        train_label_counts: dict[int, int],
        test_label_counts: dict[int, int],
    ) -> int:
        """Return the sequences eligible for train-fraction accounting.

        Args:
            sequence_count (int): Total number of generated sequences.
            train_label_counts (dict[int, int]): Label counts assigned to the
                train split.
            test_label_counts (dict[int, int]): Label counts assigned to the
                test split.

        Returns:
            int: Count of sequences considered eligible for train-fraction
                calculations for this grouping strategy.
        """
        del self
        del train_label_counts, test_label_counts
        return sequence_count

    def with_train_fraction(
        self,
        train_frac: float,
    ) -> Self:
        """Return a copy with an updated train/test split fraction.

        Args:
            train_frac (float): Requested fraction of eligible sequences to
                assign to the train split.

        Returns:
            Self: Copy with updated train/test split fraction.
        """
        return replace(self, train_frac=train_frac)

    def represent_with(
        self,
        representation: SequenceRepresentation[TRepresentation],
    ) -> SequenceRepresentationView[TRepresentation]:
        """Return a lazy builder that applies a representation per sequence.

        Args:
            representation (SequenceRepresentation[TRepresentation]): Sequence
                representation to apply lazily to each built sequence.

        Returns:
            SequenceRepresentationView[TRepresentation]: Lazy represented view of
                the generated sequences.
        """
        return SequenceRepresentationView(sequences=self, representation=representation)

    def __iter__(self) -> Iterator[TemplateSequence]:
        """Iterate over template sequences yielded by the configured grouping.

        Yields:
            TemplateSequence: One grouped and template-enriched sequence.

        Raises:
            ValueError: If the requested train split is impossible for the
                configured grouping and constraints.
        """
        rows_iter = self._iter_rows()
        infer_template = functools.lru_cache(maxsize=50_000)(self.infer_template)
        label_for_group = functools.lru_cache(maxsize=100_000)(self.label_for_group)

        if self.mode is GroupingMode.ENTITY:
            entity_counts = self.sink.count_entities_by_label(label_for_group)
            target_in_train = (
                math.ceil(self.train_frac * entity_counts.total_entities)
                if entity_counts.total_entities
                else 0
            )
            if (
                self._uses_normal_only_training()
                and target_in_train > entity_counts.normal_entities
            ):
                msg = (
                    "Requested train fraction is impossible with "
                    "train_on_normal_entities_only enabled: "
                    f"target_train_sequences={target_in_train}, "
                    f"eligible_normal_sequences={entity_counts.normal_entities}, "
                    f"total_sequences={entity_counts.total_entities}. "
                    "Lower train_fraction or disable normal-only training."
                )
                raise ValueError(msg)
            normals_seen_in_train = 0

            for window_id, rows in enumerate(rows_iter):
                if self._uses_normal_only_training():
                    entity_is_anomalous = any(
                        is_anomalous_label(label_for_group(r.entity_id))
                        for r in rows
                        if r.entity_id is not None
                    )
                    split_label = (
                        SplitLabel.TRAIN
                        if (not entity_is_anomalous)
                        and (normals_seen_in_train < target_in_train)
                        else SplitLabel.TEST
                    )
                else:
                    split_label = (
                        SplitLabel.TRAIN
                        if window_id < target_in_train
                        else SplitLabel.TEST
                    )

                seq = self._build_sequence(
                    window_id,
                    rows,
                    infer_template,
                    label_for_group,
                    split_label,
                )
                if seq is not None:
                    if (
                        self._uses_normal_only_training()
                        and split_label is SplitLabel.TRAIN
                        and seq.label == 0
                    ):
                        normals_seen_in_train += 1
                    yield seq
            return

        # Non-entity grouping: simple positional cutoff
        total_sequences = self._count_windows()
        target_in_train = (
            math.ceil(self.train_frac * total_sequences) if total_sequences else 0
        )

        for window_id, rows in enumerate(rows_iter):
            split_label = (
                SplitLabel.TRAIN if window_id < target_in_train else SplitLabel.TEST
            )
            seq = self._build_sequence(
                window_id,
                rows,
                infer_template,
                label_for_group,
                split_label,
            )
            if seq is not None:
                yield seq

    def build_split_summary(
        self,
        *,
        sequence_count: int,
        train_sequence_count: int,
        train_label_counts: dict[int, int],
        test_label_counts: dict[int, int],
    ) -> SequenceSplitSummary:
        """Describe requested versus effective split semantics for one run.

        Args:
            sequence_count (int): Total number of generated sequences.
            train_sequence_count (int): Number of sequences assigned to train.
            train_label_counts (dict[int, int]): Label counts in the train split.
            test_label_counts (dict[int, int]): Label counts in the test split.

        Returns:
            SequenceSplitSummary: Requested and effective split metrics.
        """
        eligible_train_sequence_count = self.eligible_train_sequence_count(
            sequence_count=sequence_count,
            train_label_counts=train_label_counts,
            test_label_counts=test_label_counts,
        )
        effective_train_fraction_of_eligible = (
            train_sequence_count / eligible_train_sequence_count
            if eligible_train_sequence_count
            else 0.0
        )
        effective_train_fraction_overall = (
            train_sequence_count / sequence_count if sequence_count else 0.0
        )
        return SequenceSplitSummary(
            requested_train_fraction=self.train_frac,
            train_fraction_scope="all_sequences",
            train_on_normal_entities_only=self._uses_normal_only_training(),
            eligible_train_sequence_count=eligible_train_sequence_count,
            effective_train_fraction_of_eligible=round(
                effective_train_fraction_of_eligible,
                8,
            ),
            effective_train_fraction_overall=round(
                effective_train_fraction_overall,
                8,
            ),
        )

    def _uses_normal_only_training(self) -> bool:
        """Return whether train is restricted to normal entities only."""
        del self
        return False

    def _iter_rows(self) -> Iterator[Collection[StructuredLine]]:
        """Return grouped rows for the configured strategy.

        Returns:
            Iterator[Collection[StructuredLine]]: Iterator over grouped windows
                of structured rows.

        Raises:
            NotImplementedError: Always, until implemented by a subclass.
        """
        msg = f"{type(self).__name__} must implement _iter_rows()."
        raise NotImplementedError(msg)
        return iter(())

    def _count_windows(self) -> int:
        """Return the number of grouped windows for non-entity splitting.

        Returns:
            int: Total count of grouped windows for positional splitting.

        Raises:
            NotImplementedError: Always, until implemented by a subclass.
        """
        msg = f"{type(self).__name__} must implement _count_windows()."
        raise NotImplementedError(msg)
        return 0

    def _build_sequence(
        self,
        window_id: int,
        rows: Collection[StructuredLine],
        infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]],
        label_for_group: Callable[[str], int | None],
        split_label: SplitLabel,
    ) -> TemplateSequence | None:
        """Convert a window of rows into a TemplateSequence if not empty.

        Args:
            window_id (int): Monotonic window identifier for the generated
                sequence.
            rows (Collection[StructuredLine]): Structured rows in the current
                window.
            infer_template (Callable[[str], tuple[LogTemplate, ExtractedParameters]]):
                Template inference function for untemplated row text.
            label_for_group (Callable[[str], int | None]): Group-level anomaly
                label lookup by entity id.
            split_label (SplitLabel): Assigned dataset split for the sequence.

        Returns:
            TemplateSequence | None: Built sequence, or `None` for empty windows.
        """
        if not rows:
            return None

        events: list[tuple[str, list[str], int | None]] = []
        seq_label = 0
        prev_ts: int | None = None

        unique_ids = self._entity_ids_for_rows(rows)

        for r in rows:
            template, params = infer_template(r.untemplated_message_text)
            dt, prev_ts = self._compute_dt(prev_ts, r.timestamp_unix_ms)

            events.append((template, list(params), dt))

            if seq_label == 1:
                continue

            line_lab = getattr(r, "anomalous", None)
            if is_anomalous_label(line_lab):
                seq_label = 1
                continue

            ent = r.entity_id
            if ent is not None and is_anomalous_label(label_for_group(ent)):
                seq_label = 1

        return TemplateSequence(
            events=events,
            label=seq_label,
            entity_ids=unique_ids,
            window_id=window_id,
            split_label=split_label,
        )

    def _entity_ids_for_rows(self, rows: Collection[StructuredLine]) -> list[str]:
        """Return unique entity ids for one window in first-seen order."""
        del self
        seen: set[str] = set()
        entity_ids: list[str] = []
        for row in rows:
            if row.entity_id is None or row.entity_id in seen:
                continue
            seen.add(row.entity_id)
            entity_ids.append(row.entity_id)
        return entity_ids

    @staticmethod
    def _compute_dt(
        prev_ts: int | None,
        ts: int | None,
    ) -> tuple[int | None, int | None]:
        """Compute delta time between events while preserving previous ts.

        Args:
            prev_ts (int | None): Previous event timestamp in milliseconds.
            ts (int | None): Current event timestamp in milliseconds.

        Examples:
            >>> SequenceBuilder._compute_dt(None, 1000)
            (None, 1000)
            >>> SequenceBuilder._compute_dt(1000, 1250)
            (250, 1250)
            >>> SequenceBuilder._compute_dt(2000, None)
            (None, 2000)

        Returns:
            tuple[int | None, int | None]: Delta from the previous timestamp and
                the updated previous timestamp to carry forward.
        """
        if ts is None:
            return None, prev_ts
        if prev_ts is None:
            return None, ts
        return int(ts) - int(prev_ts), ts

    @staticmethod
    def _count_fixed_windows(
        *,
        sink: StructuredSink,
        window_size: int,
        step: int | None,
    ) -> int:
        """Estimate number of fixed windows given window and step sizes.

        Args:
            sink (StructuredSink): Structured sink providing the total row count.
            window_size (int): Number of rows per window.
            step (int | None): Step between successive windows, or `None` to use
                `window_size`.

        Examples:
            >>> class _Sink:
            ...     def count_rows(self):
            ...         return 10
            ...
            >>> sb = FixedSequenceBuilder(
            ...     sink=_Sink(),
            ...     infer_template=lambda s: (s, ()),
            ...     label_for_group=lambda _: 0,
            ...     window_size=4,
            ...     step=2,
            ... )
            >>> sb._count_windows()
            4

        Returns:
            int: Estimated count of fixed windows for the sink.
        """
        if window_size <= 0:
            return 0
        step = step or window_size
        if step <= 0:
            return 0
        n = sink.count_rows()
        if n <= 0:
            return 0

        if n <= window_size:
            return 1
        return 1 + math.ceil((n - window_size) / step)

    @staticmethod
    def _count_time_windows(
        *,
        sink: StructuredSink,
        time_span_ms: int,
        step: int | None,
    ) -> int:
        """Estimate number of time windows from sink timestamp bounds.

        Args:
            sink (StructuredSink): Structured sink providing timestamp bounds.
            time_span_ms (int): Width of each time window in milliseconds.
            step (int | None): Step between successive windows, or `None` to use
                `time_span_ms`.

        Examples:
            >>> class _Sink:
            ...     def timestamp_bounds(self):
            ...         return 1_000, 3_500
            ...
            >>> sb = TimeSequenceBuilder(
            ...     sink=_Sink(),
            ...     infer_template=lambda s: (s, ()),
            ...     label_for_group=lambda _: 0,
            ...     time_span_ms=1_000,
            ...     step=500,
            ... )
            >>> sb._count_windows()
            4

        Returns:
            int: Estimated count of time windows for the sink.
        """
        if time_span_ms <= 0:
            return 0
        step = step or time_span_ms
        if step <= 0:
            return 0
        first_ts, last_ts = sink.timestamp_bounds()
        if first_ts is None or last_ts is None:
            return 0
        if last_ts < first_ts:
            return 0
        span = time_span_ms
        duration = last_ts - first_ts

        if duration < span:
            return 1

        return (duration - span) // step + 1


@dataclass(slots=True, frozen=True, kw_only=True)
class EntitySequenceBuilder(SequenceBuilder):
    """Sequence builder for per-entity grouping."""

    train_on_normal_entities_only: bool = False

    @classmethod
    def from_dataset(
        cls,
        td: TemplatedDataset,
    ) -> Self:
        """Create an entity-grouped builder from a templated dataset.

        Args:
            td (TemplatedDataset): Templated dataset to bind into the builder.

        Returns:
            Self: Builder bound to the templated dataset.
        """
        return cls(
            sink=td.sink,
            infer_template=td.template_parser.inference,
            label_for_group=td.anomaly_labels.label_for_group,
        )

    @property
    def mode(self) -> GroupingMode:
        """Return the grouping strategy for this builder."""
        return GroupingMode.ENTITY

    def with_train_on_normal_entities_only(
        self,
        *,
        enabled: bool = True,
    ) -> Self:
        """Limit training sequences to entities without anomalies.

        Args:
            enabled (bool): Whether to restrict train sequences to normal
                entities only.

        Returns:
            Self: Copy with updated normal-only training behavior.
        """
        return replace(self, train_on_normal_entities_only=enabled)

    @override
    def eligible_train_sequence_count(
        self,
        *,
        sequence_count: int,
        train_label_counts: dict[int, int],
        test_label_counts: dict[int, int],
    ) -> int:
        """Return the sequences eligible for train-fraction accounting.

        Args:
            sequence_count (int): Total number of generated entity sequences.
            train_label_counts (dict[int, int]): Label counts assigned to the
                train split.
            test_label_counts (dict[int, int]): Label counts assigned to the
                test split.

        Returns:
            int: Eligible entity-sequence count under the current policy.
        """
        del sequence_count
        if not self.train_on_normal_entities_only:
            return sum(train_label_counts.values()) + sum(test_label_counts.values())
        return sum(train_label_counts.values()) + test_label_counts.get(0, 0)

    @override
    def _uses_normal_only_training(self) -> bool:
        """Return whether train is restricted to normal entities only."""
        return self.train_on_normal_entities_only

    @override
    def _iter_rows(self) -> Iterator[Collection[StructuredLine]]:
        """Return rows grouped by entity."""
        return self.sink.iter_entity_sequences()()

    @override
    def _count_windows(self) -> int:
        """Return the entity-grouped window count.

        Entity splitting does not use this path because it needs label-aware
        budgeting over eligible entities rather than a simple positional cutoff.
        """
        msg = "EntitySequenceBuilder does not use _count_windows()."
        raise NotImplementedError(msg)

    @override
    def _entity_ids_for_rows(self, rows: Collection[StructuredLine]) -> list[str]:
        """Return the single entity id for an entity-grouped window."""
        for row in rows:
            if row.entity_id is not None:
                return [row.entity_id]
        return []


@dataclass(slots=True, frozen=True, kw_only=True)
class FixedSequenceBuilder(SequenceBuilder):
    """Sequence builder for fixed-size window grouping."""

    window_size: int
    step: int | None = None

    @property
    def mode(self) -> GroupingMode:
        """Return the grouping strategy for this builder."""
        return GroupingMode.FIXED

    @override
    def _iter_rows(self) -> Iterator[Collection[StructuredLine]]:
        """Return rows grouped by fixed-size windows."""
        return self.sink.iter_fixed_window_sequences(
            self.window_size,
            step_size=self.step,
        )()

    @override
    def _count_windows(self) -> int:
        """Return the number of fixed-size windows.

        Returns:
            int: Count of fixed-size windows implied by the sink and config.
        """
        return self._count_fixed_windows(
            sink=self.sink,
            window_size=self.window_size,
            step=self.step,
        )


@dataclass(slots=True, frozen=True, kw_only=True)
class TimeSequenceBuilder(SequenceBuilder):
    """Sequence builder for time-window grouping."""

    time_span_ms: int
    step: int | None = None

    @property
    def mode(self) -> GroupingMode:
        """Return the grouping strategy for this builder."""
        return GroupingMode.TIME

    @override
    def _iter_rows(self) -> Iterator[Collection[StructuredLine]]:
        """Return rows grouped by time windows."""
        return self.sink.iter_time_window_sequences(
            self.time_span_ms,
            step_span_ms=self.step,
        )()

    @override
    def _count_windows(self) -> int:
        """Return the number of time windows.

        Returns:
            int: Count of time windows implied by the sink timestamps and config.
        """
        return self._count_time_windows(
            sink=self.sink,
            time_span_ms=self.time_span_ms,
            step=self.step,
        )
