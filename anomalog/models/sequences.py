from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Iterator

    from anomalog.structured_parsers.contracts import StructuredLine, StructuredSink
    from anomalog.template_parsers.templated_dataset import TemplatedDataset


@dataclass(slots=True)
class TemplateSequence:
    events: list[
        tuple[str, list[str], int | None]
    ]  # (template, parameters, dt_prev_ms)
    counts: Counter[str]
    label: int
    entity_ids: list[str]  # unique entity ids present (may be empty)
    window_id: int

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
    mode: Literal["entity", "fixed", "time"]
    window_size: int | None = None
    time_span_ms: int | None = None
    step: int | None = None

    @classmethod
    def from_dataset(cls, td: TemplatedDataset) -> SequenceBuilder:
        return cls(
            sink=td.sink,
            infer=td.template_parser.inference,
            label_for_group=td.anomaly_labels.label_for_group,
            mode="entity",
        )

    def fixed(self, size: int, step: int | None = None) -> SequenceBuilder:
        return replace(
            self,
            mode="fixed",
            window_size=size,
            time_span_ms=None,
            step=step,
        )

    def time(self, span_ms: int, step_ms: int | None = None) -> SequenceBuilder:
        return replace(
            self,
            mode="time",
            window_size=None,
            time_span_ms=span_ms,
            step=step_ms,
        )

    def entity(self) -> SequenceBuilder:
        return replace(
            self,
            mode="entity",
            window_size=None,
            time_span_ms=None,
            step=None,
        )

    def __iter__(self) -> Iterator[TemplateSequence]:
        rows_iter = self._rows_iterator()
        infer = self.infer
        label_for_group = self.label_for_group

        for window_id, rows in enumerate(rows_iter):
            seq = self._build_sequence(
                window_id,
                rows,
                infer,
                label_for_group,
            )
            if seq is not None:
                yield seq

    def _rows_iterator(self) -> Iterator[list[StructuredLine]]:
        if self.mode == "time":
            return self.sink.iter_time_window_sequences(
                self.time_span_ms,  # type: ignore[arg-type]
                step_span_ms=self.step,
            )()
        if self.mode == "fixed":
            return self.sink.iter_fixed_window_sequences(
                self.window_size,  # type: ignore[arg-type]
                step_size=self.step,
            )()
        return self.sink.iter_entity_sequences()()

    def _build_sequence(
        self,
        window_id: int,
        rows: list[StructuredLine],
        infer: Callable[[str], tuple[str, Iterable[str]]],
        label_for_group: Callable[[str], int | None],
    ) -> TemplateSequence | None:
        if not rows:
            return None

        events: list[tuple[str, list[str], int | None]] = []
        counts: Counter[str] = Counter()
        seq_label = 0
        group_label_cache: dict[str, int | None] = {}
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
            if ent is not None:
                label = group_label_cache.get(ent)
                if label is None:
                    label = label_for_group(ent)
                    group_label_cache[ent] = label
                if label == 1:
                    seq_label = 1

        return TemplateSequence(
            events=events,
            counts=counts,
            label=seq_label,
            entity_ids=unique_ids,
            window_id=window_id,
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
