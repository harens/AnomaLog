from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier

from anomalog.models.base import BatchSequenceModelAdapter
from anomalog.models.features import FeatureConfig, FeaturePipeline

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    from anomalog.models.sequences import TemplateSequence


@dataclass(slots=True)
class LogRegConfig:
    n_features: int = 2**18
    alpha: float = 0.0001
    random_state: int = 42
    include_params: bool = False
    use_length_feats: bool = True
    use_dt_stats: bool = True
    ngram: int = 1


def build_logreg_model(cfg: object | None) -> BatchSequenceModelAdapter:
    cfg = cfg if isinstance(cfg, LogRegConfig) or cfg is None else None
    cfg = cfg or LogRegConfig()
    feat_cfg = FeatureConfig(
        include_params=cfg.include_params,
        use_length_feats=cfg.use_length_feats,
        use_dt_stats=cfg.use_dt_stats,
        ngram=cfg.ngram,
    )
    pipeline = FeaturePipeline(feat_cfg)
    hasher = FeatureHasher(
        n_features=cfg.n_features,
        input_type="dict",
        alternate_sign=False,
    )
    estimator = SGDClassifier(
        loss="log_loss",
        alpha=cfg.alpha,
        random_state=cfg.random_state,
        max_iter=5,
        tol=1e-3,
        n_jobs=None,
    )

    class _LogRegAdapter(BatchSequenceModelAdapter):
        def __init__(self) -> None:
            super().__init__(estimator)
            self.estimator: SGDClassifier = estimator
            self.pipeline = pipeline
            self.hasher = hasher

        def _fit(
            self,
            feats: Iterable[TemplateSequence],
            labels: Iterable[int],
        ) -> None:
            x_mat = self.hasher.transform([self.pipeline(seq) for seq in feats])
            y = np.fromiter(labels, dtype=int)
            self.estimator.partial_fit(x_mat, y, classes=np.array([0, 1]))

        def _predict(self, batch: list[TemplateSequence]) -> list[float]:
            x_mat = self.hasher.transform([self.pipeline(seq) for seq in batch])
            proba = self.estimator.predict_proba(x_mat)
            return proba[:, 1].tolist()

    return _LogRegAdapter()
