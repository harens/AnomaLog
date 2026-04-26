"""Shared experiment progress metadata and stage labels."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from dataclasses import dataclass
from typing import Generic, TypeVar

TItem = TypeVar("TItem")


@dataclass(frozen=True, slots=True)
class ProgressHint:
    """Exact bounded-progress metadata for a sequence stage.

    Attributes:
        total (int): Exact number of items expected in the stage.
        unit (str | None): Optional unit label shown beside the count.
    """

    total: int
    unit: str | None = None


@dataclass(frozen=True, slots=True)
class RunProgressPlan:
    """Shared bounded-progress hints for one experiment model run.

    Attributes:
        train (ProgressHint | None): Exact fit-stage metadata when known.
        score (ProgressHint | None): Exact test-scoring metadata when known.
    """

    train: ProgressHint | None = None
    score: ProgressHint | None = None


@dataclass(frozen=True, slots=True)
class _SizedIterable(Iterable[TItem], Sized, Generic[TItem]):
    """Lazy iterable wrapper that exposes a known total via ``len()``.

    Attributes:
        items (Iterable[TItem]): Lazy wrapped item stream.
        total (int): Exact number of wrapped items.
    """

    items: Iterable[TItem]
    total: int

    def __iter__(self) -> Iterator[TItem]:
        """Yield the wrapped item stream lazily.

        Returns:
            Iterator[TItem]: Iterator over the wrapped item stream.
        """
        return iter(self.items)

    def __len__(self) -> int:
        """Return the known number of wrapped items.

        Returns:
            int: Exact wrapped item count.
        """
        return self.total


def with_known_total(
    items: Iterable[TItem],
    *,
    hint: ProgressHint | None,
) -> Iterable[TItem]:
    """Wrap an iterable with ``len()`` support when an exact total is known.

    Args:
        items (Iterable[TItem]): Lazy item stream to expose as sized.
        hint (ProgressHint | None): Exact total metadata when cheaply known.

    Returns:
        Iterable[TItem]: Original iterable or a lazy sized wrapper.
    """
    if hint is None:
        return items
    return _SizedIterable(items=items, total=hint.total)


def fit_stage_description(detector_name: str) -> str:
    """Return the shared fit-stage label for one detector.

    Args:
        detector_name (str): Detector name shown in the progress bar.

    Returns:
        str: Shared fit-stage description.
    """
    return f"Fitting {detector_name} sequences"


def score_stage_description(detector_name: str) -> str:
    """Return the shared scoring-stage label for one detector.

    Args:
        detector_name (str): Detector name shown in the progress bar.

    Returns:
        str: Shared scoring-stage description.
    """
    return f"Scoring {detector_name} test sequences"
