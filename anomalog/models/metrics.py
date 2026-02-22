from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

if TYPE_CHECKING:  # pragma: no cover
    from anomalog.models.sequences import TemplateSequence

PROB_THRESHOLD = 0.5


@dataclass(slots=True)
class MetricSet:
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    confusion: dict[str, Any]
    top_entities: list[dict[str, Any]]
    test_total: int
    test_positive: int
    test_pos_rate: float


@dataclass(slots=True)
class EvaluationResult:
    model_id: str
    grouping: str
    metrics: MetricSet
    class_balance: dict[str, Any]
    artifacts: dict[str, str] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


class MetricsComputer:
    """Collect metrics from prediction streams."""

    def __init__(self) -> None:
        self.y_true: list[int] = []
        self.y_prob: list[float] = []
        self.per_entity: list[tuple[str, float, int]] = []

    def add_batch(
        self,
        sequences: list[TemplateSequence],
        probs: list[float],
    ) -> None:
        for seq, prob in zip(sequences, probs, strict=True):
            self.y_true.append(seq.label)
            self.y_prob.append(prob)
            self.per_entity.append(
                (seq.entity_id or f"window-{seq.window_id}", prob, seq.label),
            )

    def compute(self, *, top_k_entities: int = 20) -> MetricSet:
        if not self.y_true:
            empty_conf = {
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
                "matrix": [[0, 0], [0, 0]],
            }
            return MetricSet(
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

        preds = [1 if p >= PROB_THRESHOLD else 0 for p in self.y_prob]
        cm = confusion_matrix(self.y_true, preds, labels=[0, 1]).tolist()
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        conf = {
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn),
            "matrix": cm,
        }

        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_true,
            preds,
            average="binary",
            zero_division=0,
        )

        roc_auc = None
        pr_auc = None
        try:
            roc_auc = float(roc_auc_score(self.y_true, self.y_prob))
        except ValueError:
            roc_auc = None
        try:
            pr_auc = float(average_precision_score(self.y_true, self.y_prob))
        except ValueError:
            pr_auc = None

        top_entities = sorted(self.per_entity, key=lambda x: x[1], reverse=True)[
            :top_k_entities
        ]

        return MetricSet(
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
            test_total=len(self.y_true),
            test_positive=sum(self.y_true),
            test_pos_rate=(sum(self.y_true) / len(self.y_true)) if self.y_true else 0.0,
        )


@dataclass(slots=True)
class ClassBalance:
    train_total: int
    train_positive: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_total": self.train_total,
            "train_positive": self.train_positive,
        }
