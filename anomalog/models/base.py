from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    from anomalog.models.sequences import TemplateSequence


class SequenceModel(Protocol):
    """Streaming-friendly interface for sequence anomaly models."""

    id: str

    def setup(self) -> None:
        """One-time initialisation. Called before any training."""

    def partial_train(self, batch: list[TemplateSequence]) -> None:
        """Incrementally fit on a batch of sequences."""

    def predict_proba(self, batch: list[TemplateSequence]) -> list[float]:
        """Return anomaly probabilities aligned with batch order."""

    def finalize(self) -> None:
        """Hook after training to freeze/prepare for inference."""


class BatchSequenceModelAdapter(SequenceModel):
    """Wrap batch-only estimators (fit/predict_proba) for streaming batches."""

    def __init__(self, estimator: object, *, classes: Iterable[int] = (0, 1)) -> None:
        self.id = estimator.__class__.__name__
        self.estimator = estimator
        self._train_buf: list[TemplateSequence] = []
        self._classes = tuple(classes)

    def setup(self) -> None:  # pragma: no cover - thin wrapper
        pass

    def partial_train(self, batch: list[TemplateSequence]) -> None:
        self._train_buf.extend(batch)

    def _fit_if_needed(self) -> None:
        if not self._train_buf:
            return
        feats, labels = zip(
            *[(seq, seq.label) for seq in self._train_buf],
            strict=True,
        )
        self._fit(feats, labels)
        self._train_buf.clear()

    def _fit(
        self,
        feats: tuple[TemplateSequence, ...],
        labels: tuple[int, ...],
    ) -> None:
        raise NotImplementedError

    def predict_proba(self, batch: list[TemplateSequence]) -> list[float]:
        self._fit_if_needed()
        return self._predict(batch)

    def _predict(self, batch: list[TemplateSequence]) -> list[float]:
        raise NotImplementedError

    def finalize(self) -> None:
        self._fit_if_needed()
