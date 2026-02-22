from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from anomalog.models.sequences import TemplateSequence


@dataclass(slots=True)
class FeatureConfig:
    include_params: bool = False
    use_length_feats: bool = True
    use_dt_stats: bool = True
    ngram: int = 1
    max_vocab: int = 100_000


class FeaturePipeline:
    """Extract sparse feature dicts from TemplateSequence objects."""

    def __init__(self, cfg: FeatureConfig | None = None) -> None:
        self.cfg = cfg or FeatureConfig()
        self._vocab_seen: set[str] = set()

    def __call__(self, seq: TemplateSequence) -> dict[str, int]:
        feats: dict[str, int] = {}
        self._add_template_counts(feats, seq)
        self._add_tpl_ngrams(feats, seq.templates, self.cfg.ngram)
        if self.cfg.include_params:
            self._add_params(feats, seq)
        if self.cfg.use_length_feats:
            self._add_length_feats(feats, seq)
        if self.cfg.use_dt_stats:
            self._add_dt_stats(feats, seq.events)

        self._cap_features(feats)
        return feats

    def _cap_features(self, feats: dict[str, int]) -> None:
        for key in list(feats.keys()):
            if key in self._vocab_seen:
                continue
            if len(self._vocab_seen) < self.cfg.max_vocab:
                self._vocab_seen.add(key)
            else:
                feats.pop(key, None)

    @staticmethod
    def _add_template_counts(feats: dict[str, int], seq: TemplateSequence) -> None:
        for tpl, count in seq.counts.items():
            feats[f"tpl={tpl}"] = count

    @staticmethod
    def _add_tpl_ngrams(
        feats: dict[str, int],
        templates: list[str],
        max_n: int,
    ) -> None:
        if max_n <= 1:
            return
        max_n = min(max_n, len(templates))
        for n in range(2, max_n + 1):
            for i in range(len(templates) - n + 1):
                key = "|".join(templates[i : i + n])
                feats[f"tpl_ng{n}={key}"] = feats.get(f"tpl_ng{n}={key}", 0) + 1

    @staticmethod
    def _add_params(feats: dict[str, int], seq: TemplateSequence) -> None:
        for tpl, params, _ in seq.events:
            for param in params:
                key = f"param={tpl}:{param}"
                feats[key] = feats.get(key, 0) + 1

    @staticmethod
    def _add_length_feats(feats: dict[str, int], seq: TemplateSequence) -> None:
        feats[f"len_events={len(seq.events)}"] = 1
        feats[f"uniq_tpls={len(seq.counts)}"] = 1

    @staticmethod
    def _bucket(value: float) -> str:
        if value <= 0:
            return "0"
        exp = int(np.floor(np.log10(value)))
        return f"1e{exp}"

    def _add_dt_stats(
        self,
        feats: dict[str, int],
        events: list[tuple[str, list[str], int | None]],
    ) -> None:
        dts = [dt for _, _, dt in events if dt is not None]
        if not dts:
            return
        feats[f"dt_mean={self._bucket(statistics.fmean(dts))}"] = 1
        feats[f"dt_p95={self._bucket(float(np.percentile(dts, 95)))}"] = 1
        feats[f"dt_max={self._bucket(max(dts))}"] = 1
