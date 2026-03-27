"""Utilities for building template sequences from structured log lines.

The module groups parsed log lines into windows (entity, fixed-size, or
time-based) and decorates them with inferred templates and anomaly labels.
"""

from __future__ import annotations

import functools
import math
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Collection, Iterator

    from anomalog.structured_parsers.contracts import StructuredLine, StructuredSink
    from anomalog.template_parsers.templated_dataset import (
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


@dataclass(slots=True)
class TemplateSequence:
    """Sequence of templated events used by downstream models."""

    events: list[
        tuple[str, list[str], int | None]
    ]  # (template, parameters, dt_prev_ms)
    counts: Counter[str]
    label: int
    entity_ids: list[str]  # unique entity ids present (may be empty)
    window_id: int
    split_label: SplitLabel = SplitLabel.TRAIN

    @property
    def templates(self) -> list[str]:
        """Backwards-compatible access to template strings only."""
        return [tpl for tpl, _, _ in self.events]

    @property
    def entity_id(self) -> str | None:
        """Return a single entity id if the sequence is homogenous.

        If multiple entities appear in the window, None is returned to avoid
        implying a dominant entity.
        """
        if len(self.entity_ids) == 1:
            return self.entity_ids[0]
        return None


@dataclass(slots=True, frozen=True)
class SequenceBuilder:
    """Build sequences from a structured sink and template inference function."""

    sink: StructuredSink
    infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]]
    label_for_group: Callable[[str], int | None]
    mode: GroupingMode
    window_size: int | None = None
    time_span_ms: int | None = None
    step: int | None = None
    train_frac: float = 0.8
    train_on_normal_entities_only: bool = False

    @classmethod
    def from_dataset(cls, td: TemplatedDataset) -> SequenceBuilder:
        """Create a builder using configuration from a templated dataset."""
        return cls(
            sink=td.sink,
            infer_template=td.template_parser.inference,
            label_for_group=td.anomaly_labels.label_for_group,
            mode=GroupingMode.ENTITY,
        )

    def fixed(self, size: int, step: int | None = None) -> SequenceBuilder:
        """Return a builder configured for fixed-size windows."""
        return replace(
            self,
            mode=GroupingMode.FIXED,
            window_size=size,
            time_span_ms=None,
            step=step,
        )

    def time(self, span_ms: int, step_ms: int | None = None) -> SequenceBuilder:
        """Return a builder configured for time-based windows."""
        return replace(
            self,
            mode=GroupingMode.TIME,
            window_size=None,
            time_span_ms=span_ms,
            step=step_ms,
        )

    def entity(self) -> SequenceBuilder:
        """Return a builder configured for per-entity windows."""
        return replace(
            self,
            mode=GroupingMode.ENTITY,
            window_size=None,
            time_span_ms=None,
            step=None,
        )

    def with_train_fraction(self, train_frac: float) -> SequenceBuilder:
        """Return a copy with an updated train/test split fraction."""
        return replace(self, train_frac=train_frac)

    def with_train_on_normal_entities_only(
        self,
        *,
        enabled: bool = True,
    ) -> SequenceBuilder:
        """Limit training sequences to entities without anomalies."""
        return replace(self, train_on_normal_entities_only=enabled)

    def __iter__(self) -> Iterator[TemplateSequence]:
        """Iterate over template sequences yielded by the configured grouping."""
        rows_iter = self._rows_iterator()
        infer_template = functools.lru_cache(maxsize=50_000)(self.infer_template)
        label_for_group = functools.lru_cache(maxsize=100_000)(self.label_for_group)

        if self.mode is GroupingMode.ENTITY:
            entity_counts = self.sink.count_entities_by_label(label_for_group)
            base = (
                entity_counts.normal_entities
                if self.train_on_normal_entities_only
                else entity_counts.total_entities
            )
            target_in_train = math.ceil(self.train_frac * base) if base else 0
            normals_seen_in_train = 0

            for window_id, rows in enumerate(rows_iter):
                if self.train_on_normal_entities_only:
                    entity_is_anomalous = any(
                        label_for_group(r.entity_id) == 1
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
                        self.train_on_normal_entities_only
                        and split_label is SplitLabel.TRAIN
                        and seq.label == 0
                    ):
                        normals_seen_in_train += 1
                    yield seq
            return

        # Non-entity grouping: simple positional cutoff
        total_sequences = (
            self._count_fixed_windows()
            if self.mode is GroupingMode.FIXED
            else self._count_time_windows()
        )
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

    def _rows_iterator(self) -> Iterator[Collection[StructuredLine]]:
        """Return an iterator over grouped rows based on the current mode."""
        if self.mode is GroupingMode.TIME:
            return self.sink.iter_time_window_sequences(
                self.time_span_ms,  # type: ignore[arg-type]
                step_span_ms=self.step,
            )()
        if self.mode is GroupingMode.FIXED:
            return self.sink.iter_fixed_window_sequences(
                self.window_size,  # type: ignore[arg-type]
                step_size=self.step,
            )()
        return self.sink.iter_entity_sequences()()

    def _build_sequence(
        self,
        window_id: int,
        rows: Collection[StructuredLine],
        infer_template: Callable[[str], tuple[LogTemplate, ExtractedParameters]],
        label_for_group: Callable[[str], int | None],
        split_label: SplitLabel,
    ) -> TemplateSequence | None:
        """Convert a window of rows into a TemplateSequence if not empty."""
        if not rows:
            return None

        events: list[tuple[str, list[str], int | None]] = []
        counts: Counter[str] = Counter()
        seq_label = 0
        prev_ts: int | None = None

        ids_in_window = [r.entity_id for r in rows if r.entity_id is not None]
        unique_ids = sorted(set(ids_in_window))

        for r in rows:
            template, params = infer_template(r.untemplated_message_text)
            dt, prev_ts = self._compute_dt(prev_ts, r.timestamp_unix_ms)

            events.append((template, list(params), dt))
            counts[template] += 1

            if seq_label == 1:
                continue

            line_lab = getattr(r, "anomalous", None)
            if line_lab == 1:
                seq_label = 1
                continue

            ent = r.entity_id
            if ent is not None and label_for_group(ent) == 1:
                seq_label = 1

        return TemplateSequence(
            events=events,
            counts=counts,
            label=seq_label,
            entity_ids=unique_ids,
            window_id=window_id,
            split_label=split_label,
        )

    @staticmethod
    def _compute_dt(
        prev_ts: int | None,
        ts: int | None,
    ) -> tuple[int | None, int | None]:
        """Compute delta time between events while preserving previous ts.

        >>> SequenceBuilder._compute_dt(None, 1000)
        (None, 1000)
        >>> SequenceBuilder._compute_dt(1000, 1250)
        (250, 1250)
        >>> SequenceBuilder._compute_dt(2000, None)
        (None, 2000)
        """
        if ts is None:
            return None, prev_ts
        if prev_ts is None:
            return None, ts
        return int(ts) - int(prev_ts), ts

    def _count_fixed_windows(self) -> int:
        """Estimate number of fixed windows given window and step sizes.

        >>> class _Sink:
        ...     def count_rows(self):
        ...         return 10
        ...
        >>> sb = SequenceBuilder(
        ...     sink=_Sink(),
        ...     infer_template=lambda s: (s, ()),
        ...     label_for_group=lambda _: 0,
        ...     mode=GroupingMode.FIXED,
        ...     window_size=4,
        ...     time_span_ms=None,
        ...     step=2,
        ... )
        >>> sb._count_fixed_windows()
        4
        """
        if self.window_size is None or self.window_size <= 0:
            return 0
        step = self.step or self.window_size
        if step <= 0:
            return 0
        n = self.sink.count_rows()
        if n <= 0:
            return 0

        if n <= self.window_size:
            return 1
        return 1 + math.ceil((n - self.window_size) / step)

    def _count_time_windows(self) -> int:
        """Estimate number of time windows from sink timestamp bounds.

        >>> class _Sink:
        ...     def timestamp_bounds(self):
        ...         return 1_000, 3_500
        ...
        >>> sb = SequenceBuilder(
        ...     sink=_Sink(),
        ...     infer_template=lambda s: (s, ()),
        ...     label_for_group=lambda _: 0,
        ...     mode=GroupingMode.TIME,
        ...     window_size=None,
        ...     time_span_ms=1_000,
        ...     step=500,
        ... )
        >>> sb._count_time_windows()
        4
        """
        if self.time_span_ms is None or self.time_span_ms <= 0:
            return 0
        step = self.step or self.time_span_ms
        if step <= 0:
            return 0
        first_ts, last_ts = self.sink.timestamp_bounds()
        if first_ts is None or last_ts is None:
            return 0
        if last_ts < first_ts:
            return 0
        span = self.time_span_ms
        duration = last_ts - first_ts

        if duration < span:
            return 1

        return (duration - span) // step + 1
