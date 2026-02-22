"""Streaming Naive Bayes model compatible with the SequenceModel API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.naive_bayes import ComplementNB, MultinomialNB

from anomalog.models.base import SequenceModel

if TYPE_CHECKING:  # pragma: no cover
    from anomalog.models.sequences import TemplateSequence


@dataclass(slots=True, frozen=True)
class NBConfig:
    alpha: float = 0.5
    use_complement: bool = False
    include_params: bool = False
    use_dt_stats: bool = True
    use_length_feats: bool = True
    ngram: int = 1
    n_features: int = 2**18


class NaiveBayesModel(SequenceModel):
    """Lightweight wrapper around scikit-learn NB with hashing features."""

    def __init__(self, cfg: NBConfig | None = None) -> None:
        self.cfg = cfg or NBConfig()
        self.model_cls = ComplementNB if self.cfg.use_complement else MultinomialNB
        self.model = self.model_cls(alpha=self.cfg.alpha)
        self.vectorizer = FeatureHasher(
            n_features=self.cfg.n_features,
            input_type="dict",
            alternate_sign=False,
        )
        self.classes = np.array([0, 1], dtype=int)
        self._fitted = False
        self.id = "naive_bayes"

    def setup(self) -> None:  # pragma: no cover - nothing to init
        pass

    def partial_train(self, batch: list[TemplateSequence]) -> None:
        if not batch:
            return
        feats = [_feature_dict(seq, self.cfg) for seq in batch]
        x_mat = self.vectorizer.transform(feats)
        y_vec = np.fromiter((seq.label for seq in batch), dtype=int)
        self.model.partial_fit(
            x_mat,
            y_vec,
            classes=self.classes if not self._fitted else None,
        )
        self._fitted = True

    def predict_proba(self, batch: list[TemplateSequence]) -> list[float]:
        if not batch:
            return []
        if not self._fitted:
            return [0.0 for _ in batch]
        feats = [_feature_dict(seq, self.cfg) for seq in batch]
        x_mat = self.vectorizer.transform(feats)
        return self.model.predict_proba(x_mat)[:, 1].tolist()

    def finalize(self) -> None:  # pragma: no cover - nothing to finalize
        pass


def _feature_dict(seq: TemplateSequence, cfg: NBConfig) -> dict[str, int]:
    feats: dict[str, int] = {}

    for tpl, count in seq.counts.items():
        feats[f"tpl={tpl}"] = count

    _add_tpl_ngrams(feats, seq.templates, cfg.ngram)

    if cfg.include_params:
        for tpl, params, _ in seq.events:
            for param in params:
                key = f"param={tpl}:{param}"
                feats[key] = feats.get(key, 0) + 1

    if cfg.use_length_feats:
        feats[f"len_events={len(seq.events)}"] = 1
        feats[f"uniq_tpls={len(seq.counts)}"] = 1

    if cfg.use_dt_stats:
        dts = [dt for _, _, dt in seq.events if dt is not None]
        if dts:
            feats[f"dt_mean={_bucket(np.mean(dts))}"] = 1
            feats[f"dt_p95={_bucket(float(np.percentile(dts, 95)))}"] = 1
            feats[f"dt_max={_bucket(max(dts))}"] = 1

    return feats


def _add_tpl_ngrams(feats: dict[str, int], templates: list[str], max_n: int) -> None:
    if max_n <= 1:
        return
    max_n = min(max_n, len(templates))
    for n in range(2, max_n + 1):
        for i in range(len(templates) - n + 1):
            key = "|".join(templates[i : i + n])
            feats[f"tpl_ng{n}={key}"] = feats.get(f"tpl_ng{n}={key}", 0) + 1


def _bucket(value: float) -> str:
    if value <= 0:
        return "0"
    exp = int(np.floor(np.log10(value)))
    return f"1e{exp}"


def build_nb_model(cfg: object | None = None) -> NaiveBayesModel:
    nb_cfg = cfg if isinstance(cfg, NBConfig) or cfg is None else None
    return NaiveBayesModel(nb_cfg)
