from __future__ import annotations

import functools
import math
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Collection, Iterable, Iterator

    from anomalog.structured_parsers.contracts import StructuredLine, StructuredSink
    from anomalog.template_parsers.templated_dataset import TemplatedDataset


class GroupingMode(str, Enum):
    ENTITY = "entity"
    FIXED = "fixed"
    TIME = "time"


class SplitLabel(str, Enum):
    TRAIN = "train"
    TEST = "test"


@dataclass(slots=True)
class TemplateSequence:
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
        if len(self.entity_ids) == 1:
            return self.entity_ids[0]
        return None


@dataclass(slots=True, frozen=True)
class SequenceBuilder:
    sink: StructuredSink
    infer: Callable[[str], tuple[str, Iterable[str]]]
    label_for_group: Callable[[str], int | None]
    mode: GroupingMode
    window_size: int | None = None
    time_span_ms: int | None = None
    step: int | None = None
    train_frac: float = 0.8

    @classmethod
    def from_dataset(cls, td: TemplatedDataset) -> SequenceBuilder:
        return cls(
            sink=td.sink,
            infer=td.template_parser.inference,
            label_for_group=td.anomaly_labels.label_for_group,
            mode=GroupingMode.ENTITY,
        )

    def fixed(self, size: int, step: int | None = None) -> SequenceBuilder:
        return replace(
            self,
            mode=GroupingMode.FIXED,
            window_size=size,
            time_span_ms=None,
            step=step,
        )

    def time(self, span_ms: int, step_ms: int | None = None) -> SequenceBuilder:
        return replace(
            self,
            mode=GroupingMode.TIME,
            window_size=None,
            time_span_ms=span_ms,
            step=step_ms,
        )

    def entity(self) -> SequenceBuilder:
        return replace(
            self,
            mode=GroupingMode.ENTITY,
            window_size=None,
            time_span_ms=None,
            step=None,
        )

    def with_train_fraction(self, train_frac: float) -> SequenceBuilder:
        return replace(self, train_frac=train_frac)

    def __iter__(self) -> Iterator[TemplateSequence]:
        rows_iter = self._rows_iterator()
        infer = functools.lru_cache(maxsize=50_000)(self.infer)
        label_for_group = functools.lru_cache(maxsize=100_000)(self.label_for_group)

        total = self._total_sequences_for_mode()
        train_cutoff = math.ceil(self.train_frac * total) if total else 0

        for window_id, rows in enumerate(rows_iter):
            split_label: SplitLabel = (
                SplitLabel.TRAIN if window_id < train_cutoff else SplitLabel.TEST
            )
            seq = self._build_sequence(
                window_id,
                rows,
                infer,
                label_for_group,
                split_label,
            )
            if seq is not None:
                yield seq

    def _rows_iterator(self) -> Iterator[Collection[StructuredLine]]:
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
        infer: Callable[[str], tuple[str, Iterable[str]]],
        label_for_group: Callable[[str], int | None],
        split_label: SplitLabel,
    ) -> TemplateSequence | None:
        if not rows:
            return None

        events: list[tuple[str, list[str], int | None]] = []
        counts: Counter[str] = Counter()
        seq_label = 0
        prev_ts: int | None = None

        ids_in_window = [r.entity_id for r in rows if r.entity_id is not None]
        unique_ids = sorted(set(ids_in_window))

        for r in rows:
            template, params = infer(r.untemplated_message_text)
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
        if ts is None:
            return None, prev_ts
        if prev_ts is None:
            return None, ts
        return int(ts) - int(prev_ts), ts

    def _total_sequences_for_mode(self) -> int:
        if self.mode == GroupingMode.ENTITY:
            return self.sink.count_entities()
        if self.mode == GroupingMode.FIXED:
            return self._count_fixed_windows()
        return self._count_time_windows()

    def _count_fixed_windows(self) -> int:
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
        return (last_ts - first_ts) // step + 1
