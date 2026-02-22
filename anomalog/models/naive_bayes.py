"""Stream-friendly Naive Bayes classifier for templated log sequences."""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from logging import Logger, LoggerAdapter
from typing import TYPE_CHECKING, Literal

import numpy as np
from prefect.logging import get_logger
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterator
    from logging import Logger, LoggerAdapter

    from scipy.sparse import csr_matrix

    from anomalog.models.sequences import SequenceBuilder, TemplateSequence

# Using a constant keeps ruff happy and centralises threshold tweaks.
PROB_THRESHOLD = 0.5
LOG_EVERY = 20
TRAIN_BATCH = 8_192  # larger batch keeps partial_fit overhead lower for sliding windows
EVAL_BATCH = 4_096


@dataclass(slots=True, frozen=True)
class NBConfig:
    """Configuration for Naive Bayes training/evaluation."""

    mode: Literal["entity", "fixed", "time"] = "entity"
    alpha: float = 0.5
    # ComplementNB often helps imbalance; MultinomialNB streams slightly faster.
    use_complement: bool = False
    include_params: bool = False
    use_dt_stats: bool = True
    use_length_feats: bool = True
    interpretable: bool = False
    max_vocab: int = 100_000
    top_k_features: int = 20
    ngram: int = 1
    train_frac: float = 0.8
    random_state: int = 42
    n_features: int = 2**18
    top_k_entities: int = 20


def run_naive_bayes(
    sequence_builder: SequenceBuilder,
    cfg: NBConfig,
) -> dict:
    """Train + evaluate a streaming Naive Bayes classifier.

    The iterator yields TemplateSequence objects (grouped per entity/window).
    We split by entity to avoid leakage, hash features to stay sparse, and
    use partial_fit to keep memory bounded.
    """

    logger = get_logger()
    logger.info("NB: starting run (mode=%s)", cfg.mode)

    rng = random.Random(cfg.random_state)  # noqa: S311 - deterministic split only
    entity_split: dict[str, str] = {}

    if cfg.interpretable:
        vectorizer = _build_dict_vectorizer(
            sequence_builder,
            cfg,
            entity_split,
            rng,
            logger,
        )
    else:
        vectorizer = FeatureHasher(
            n_features=cfg.n_features,
            input_type="dict",
            alternate_sign=False,
        )

    model_cls = ComplementNB if cfg.use_complement else MultinomialNB
    model = model_cls(alpha=cfg.alpha)
    classes = np.array([0, 1], dtype=int)

    ctx = _NBContext(
        model=model,
        vectorizer=vectorizer,
        entity_split=entity_split,
        rng=rng,
        cfg=cfg,
        classes=classes,
        logger=logger,
    )

    logger.info("NB: training pass started")
    n_train, pos_train, fitted = _train_model(ctx, sequence_builder)
    logger.info(
        "NB: training pass finished (%s sequences, %s positives)",
        n_train,
        pos_train,
    )

    if not fitted:
        msg = "No training data was available for Naive Bayes."
        raise RuntimeError(msg)

    if cfg.interpretable:
        _log_feature_importance(ctx)

    logger.info("NB: evaluation pass started")
    eval_result = _evaluate_model(ctx, sequence_builder)
    logger.info(
        "NB: evaluation pass finished (precision=%.4f recall=%.4f f1=%.4f)",
        eval_result.precision,
        eval_result.recall,
        eval_result.f1,
    )

    return {
        "config": cfg,
        "model": model,
        "class_balance": {
            "train_total": n_train,
            "train_positive": pos_train,
            "test_total": eval_result.test_total,
            "test_positive": eval_result.test_positive,
            "test_pos_rate": eval_result.test_pos_rate,
        },
        "metrics": {
            "precision": eval_result.precision,
            "recall": eval_result.recall,
            "f1": eval_result.f1,
            "roc_auc": eval_result.roc_auc,
            "pr_auc": eval_result.pr_auc,
        },
        "confusion": eval_result.confusion,
        "top_anomaly_entities": eval_result.top_entities,
    }


def _bucket(value: float) -> str:
    if value <= 0:
        return "0"
    exp = math.floor(math.log10(value))
    return f"1e{exp}"


def _transform(
    vectorizer: FeatureHasher | DictVectorizer,
    feats: list[dict[str, int]],
) -> csr_matrix:
    return vectorizer.transform(feats)


def _build_dict_vectorizer(
    sequence_builder: SequenceBuilder,
    cfg: NBConfig,
    entity_split: dict[str, str],
    rng: random.Random,
    logger: Logger | LoggerAdapter | None,
) -> DictVectorizer:
    vocab_seen: set[str] = set()

    def gen() -> Iterator[dict[str, int]]:
        count = 0
        for seq in sequence_builder:
            split = _split_for_sequence(seq, cfg, entity_split, rng)
            if split != "train":
                continue

            feats = _feature_dict(seq, cfg)
            _cap_features(feats, vocab_seen, cfg.max_vocab)
            count += 1
            _log_every(
                logger,
                count,
                LOG_EVERY,
                "NB: vocab fit saw %s sequences",
                count,
            )
            yield feats

    vectorizer = DictVectorizer(sparse=True)
    vectorizer.fit(gen())
    if logger:
        logger.info(
            "NB: vocab fit done (features=%s, max=%s)",
            len(vectorizer.vocabulary_),
            cfg.max_vocab,
        )
    return vectorizer


def _cap_features(
    feats: dict[str, int],
    vocab_seen: set[str],
    max_vocab: int,
) -> None:
    for key in list(feats.keys()):
        if key in vocab_seen:
            continue
        if len(vocab_seen) < max_vocab:
            vocab_seen.add(key)
        else:
            feats.pop(key, None)


def _log_feature_importance(ctx: _NBContext) -> None:
    if not ctx.logger or not isinstance(ctx.vectorizer, DictVectorizer):
        return

    if not hasattr(ctx.model, "feature_log_prob_"):
        ctx.logger.info("NB: feature importances unavailable (model not fitted)")
        return

    log_prob = np.asarray(ctx.model.feature_log_prob_)
    if log_prob.shape[0] < len(ctx.classes):
        return

    delta = log_prob[1] - log_prob[0]
    top_k = min(ctx.cfg.top_k_features, delta.shape[0])
    if top_k == 0:
        return

    names = ctx.vectorizer.get_feature_names_out()
    top_idx = np.argpartition(delta, -top_k)[-top_k:]
    top_idx = top_idx[np.argsort(delta[top_idx])[::-1]]

    for rank, idx in enumerate(top_idx, start=1):
        ctx.logger.info(
            "NB: top feature #%s %s (Δlogprob=%.4f)",
            rank,
            names[idx],
            float(delta[idx]),
        )


def _feature_dict(seq: TemplateSequence, cfg: NBConfig) -> dict[str, int]:
    feats: dict[str, int] = {}

    _add_template_counts(feats, seq)
    _add_tpl_ngrams(feats, seq.templates, cfg.ngram)
    if cfg.include_params:
        _add_params(feats, seq)
    if cfg.use_length_feats:
        _add_length_feats(feats, seq)
    if cfg.use_dt_stats:
        _add_dt_stats(feats, seq.events)

    return feats


def _add_template_counts(feats: dict[str, int], seq: TemplateSequence) -> None:
    for tpl, count in seq.counts.items():
        feats[f"tpl={tpl}"] = count


def _add_tpl_ngrams(feats: dict[str, int], templates: list[str], max_n: int) -> None:
    if max_n <= 1:
        return
    max_n = min(max_n, len(templates))
    for n in range(2, max_n + 1):
        for i in range(len(templates) - n + 1):
            key = "|".join(templates[i : i + n])
            feats[f"tpl_ng{n}={key}"] = feats.get(f"tpl_ng{n}={key}", 0) + 1


def _add_params(feats: dict[str, int], seq: TemplateSequence) -> None:
    for tpl, params, _ in seq.events:
        for param in params:
            key = f"param={tpl}:{param}"
            feats[key] = feats.get(key, 0) + 1


def _add_length_feats(feats: dict[str, int], seq: TemplateSequence) -> None:
    feats[f"len_events={len(seq.events)}"] = 1
    feats[f"uniq_tpls={len(seq.counts)}"] = 1


def _add_dt_stats(
    feats: dict[str, int],
    events: list[tuple[str, list[str], int | None]],
) -> None:
    dts = [dt for _, _, dt in events if dt is not None]
    if not dts:
        return
    feats[f"dt_mean={_bucket(statistics.fmean(dts))}"] = 1
    feats[f"dt_p95={_bucket(float(np.percentile(dts, 95)))}"] = 1
    feats[f"dt_max={_bucket(max(dts))}"] = 1


def _split_for_sequence(
    seq: TemplateSequence,
    cfg: NBConfig,
    split_map: dict[str, str],
    rng: random.Random,
) -> str:
    """Choose (and cache) train/test split per grouping key.

    For entity mode we stick to the entity id; for fixed/time windows we want a
    deterministic key per window so a sliding window does not blend entities.
    """

    if cfg.mode == "entity":
        key = seq.entity_id or f"window-{seq.window_id}"
    else:
        key = f"window-{seq.window_id}"

    return split_map.setdefault(
        key,
        "train" if rng.random() < cfg.train_frac else "test",
    )


@dataclass(slots=True, frozen=True)
class _EvalResult:
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    confusion: dict
    top_entities: list[dict[str, float | int | str]]
    test_total: int
    test_positive: int
    test_pos_rate: float


@dataclass(slots=True)
class _NBContext:
    model: MultinomialNB | ComplementNB
    vectorizer: FeatureHasher | DictVectorizer
    entity_split: dict[str, str]
    rng: random.Random
    cfg: NBConfig
    classes: np.ndarray
    logger: Logger | LoggerAdapter | None = None


def _train_model(
    ctx: _NBContext,
    sequence_builder: SequenceBuilder,
) -> tuple[int, int, bool]:
    n_train = 0
    pos_train = 0
    fitted = False
    feat_batch: list[dict[str, int]] = []
    label_batch: list[int] = []

    for seq in sequence_builder:
        split = _split_for_sequence(seq, ctx.cfg, ctx.entity_split, ctx.rng)
        if split != "train":
            continue

        feat_batch.append(_feature_dict(seq, ctx.cfg))
        label_batch.append(seq.label)

        if len(feat_batch) >= TRAIN_BATCH:
            fitted = _partial_fit_batch(
                ctx,
                feat_batch,
                label_batch,
                fitted=fitted,
            )
            feat_batch.clear()
            label_batch.clear()

        n_train += 1
        if seq.label == 1:
            pos_train += 1

        _log_every(
            ctx.logger,
            n_train,
            LOG_EVERY,
            "NB: trained %s sequences so far",
            n_train,
        )

    if feat_batch:
        fitted = _partial_fit_batch(
            ctx,
            feat_batch,
            label_batch,
            fitted=fitted,
        )

    return n_train, pos_train, fitted


def _evaluate_model(
    ctx: _NBContext,
    sequence_builder: SequenceBuilder,
) -> _EvalResult:
    y_true: list[int] = []
    y_prob: list[float] = []
    per_entity: list[tuple[str, float, int]] = []
    accum = _EvalAccum(y_true=y_true, y_prob=y_prob, per_entity=per_entity)
    feat_batch: list[dict[str, int]] = []
    entity_batch: list[str] = []
    label_batch: list[int] = []

    for seq in sequence_builder:
        split = _split_for_sequence(seq, ctx.cfg, ctx.entity_split, ctx.rng)
        entity = seq.entity_id or f"window-{seq.window_id}"
        if split != "test":
            continue

        feat_batch.append(_feature_dict(seq, ctx.cfg))
        entity_batch.append(entity)
        label_batch.append(seq.label)

        if len(feat_batch) >= EVAL_BATCH:
            _predict_batch(
                ctx,
                feat_batch,
                entity_batch,
                label_batch,
                accum,
            )

        _log_every(
            ctx.logger,
            len(y_true),
            LOG_EVERY,
            "NB: evaluated %s test sequences so far",
            len(y_true),
        )

    if feat_batch:
        _predict_batch(
            ctx,
            feat_batch,
            entity_batch,
            label_batch,
            accum,
        )

    if not y_true:
        empty_conf = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "matrix": [[0, 0], [0, 0]]}
        return _EvalResult(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            roc_auc=None,
            pr_auc=None,
            confusion=empty_conf,
            top_entities=[],
            test_total=0,
            test_positive=0,
            test_pos_rate=0.0,
        )

    preds = [1 if p >= PROB_THRESHOLD else 0 for p in y_prob]
    cm = confusion_matrix(y_true, preds, labels=[0, 1]).tolist()
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    conf = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "matrix": cm,
    }

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        preds,
        average="binary",
        zero_division=0,
    )

    roc_auc = None
    pr_auc = None
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = None
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = None

    top_entities = sorted(per_entity, key=lambda x: x[1], reverse=True)[
        : ctx.cfg.top_k_entities
    ]

    return _EvalResult(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion=conf,
        top_entities=[
            {"entity_id": ent, "p_anom": prob, "label": lbl}
            for ent, prob, lbl in top_entities
        ],
        test_total=len(y_true),
        test_positive=sum(y_true),
        test_pos_rate=(sum(y_true) / len(y_true)) if y_true else 0.0,
    )


def _log_every(
    logger: Logger | LoggerAdapter | None,
    count: int,
    every: int,
    msg: str,
    *args: object,
) -> None:
    if logger and count and count % every == 0:  # pragma: no cover - thin helper
        logger.info(msg, *args)


def _partial_fit_batch(
    ctx: _NBContext,
    feats: list[dict[str, int]],
    labels: list[int],
    *,
    fitted: bool,
) -> bool:
    x_mat = _transform(ctx.vectorizer, feats)
    y_vec = np.fromiter(labels, dtype=int)
    ctx.model.partial_fit(x_mat, y_vec, classes=ctx.classes if not fitted else None)
    return True


@dataclass(slots=True)
class _EvalAccum:
    y_true: list[int]
    y_prob: list[float]
    per_entity: list[tuple[str, float, int]]


def _predict_batch(
    ctx: _NBContext,
    feats: list[dict[str, int]],
    entities: list[str],
    labels: list[int],
    accum: _EvalAccum,
) -> None:
    x_mat = _transform(ctx.vectorizer, feats)
    probs = ctx.model.predict_proba(x_mat)[:, 1]

    for ent, lbl, prob in zip(entities, labels, probs, strict=True):
        accum.y_true.append(lbl)
        accum.y_prob.append(float(prob))
        accum.per_entity.append((ent, float(prob), lbl))

    feats.clear()
    entities.clear()
    labels.clear()
