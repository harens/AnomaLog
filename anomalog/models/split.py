from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from anomalog.models.sequences import GroupingMode

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from anomalog.models.sequences import TemplateSequence

SplitLabel = Literal["train", "test"]


@dataclass(slots=True)
class ModeAwareSplit:
    """Default split: hold out per entity/window; chronological when grouped by time."""

    train_frac: float = 0.8
    random_state: int = 42
    cutoff_window: int | None = None

    _rng: random.Random = field(init=False, repr=False)
    _cache: dict[str, SplitLabel] = field(init=False, default_factory=dict, repr=False)
    _seen: int = field(init=False, default=0, repr=False)
    _train_seen: int = field(init=False, default=0, repr=False)
    _locked_to_test: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_rng", random.Random(self.random_state))  # noqa: S311

    def __call__(self, seq: TemplateSequence, mode: GroupingMode) -> SplitLabel:
        if mode is GroupingMode.TIME:
            return self._chrono_assign(seq)
        return self._entity_assign(seq)

    def _assign_cached(self, key: str, compute: Callable[[], SplitLabel]) -> SplitLabel:
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    def _entity_assign(self, seq: TemplateSequence) -> SplitLabel:
        key = seq.entity_id if seq.entity_id else f"window-{seq.window_id}"
        return self._assign_cached(
            key,
            lambda: "train" if self._rng.random() < self.train_frac else "test",
        )

    def _chrono_assign(self, seq: TemplateSequence) -> SplitLabel:
        key = f"time-{seq.window_id}"
        return self._assign_cached(key, lambda: self._chrono_label(seq))

    def _chrono_label(self, seq: TemplateSequence) -> SplitLabel:
        if self.cutoff_window is not None:
            return "train" if seq.window_id < self.cutoff_window else "test"

        if self._locked_to_test:
            return "test"

        self._seen += 1
        current_ratio = (self._train_seen / self._seen) if self._seen else 0.0
        if current_ratio < self.train_frac:
            self._train_seen += 1
            return "train"

        self._locked_to_test = True
        return "test"
