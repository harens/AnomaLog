from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from prefect.logging import get_logger

from anomalog.io_utils import make_spinner_progress
from anomalog.models.metrics import EvaluationResult, MetricsComputer, MetricSet
from anomalog.models.split import ModeAwareSplit

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from anomalog.models.base import SequenceModel
    from anomalog.models.sequences import (
        SequenceBuilder,
        TemplateSequence,
    )


TRAIN_BATCH = 4_096
EVAL_BATCH = 4_096


@dataclass(slots=True)
class ExperimentConfig:
    builders: list[SequenceBuilder]
    models: list[tuple[str, Callable[[object | None], SequenceModel], object | None]]
    artifact_root: Path
    split_name: str = "default"
    split: ModeAwareSplit | None = None
    show_progress: bool = True


class ExperimentRunner:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.logger = get_logger("anomalog.experiment")
        self.split = cfg.split or ModeAwareSplit()

    def run(self) -> list[EvaluationResult]:
        results: list[EvaluationResult] = []
        for builder in self.cfg.builders:
            grouping = builder.mode
            self.logger.info("Running experiments for grouping=%s", grouping)
            builder_name = f"group-{grouping}"
            for model_id, factory, cfg in self.cfg.models:
                model = factory(cfg)
                model.id = model_id
                model.setup()
                result = self._run_single(model, builder, grouping)
                self._persist(result, builder_name)
                results.append(result)
        return results

    def _run_single(
        self,
        model: SequenceModel,
        builder: SequenceBuilder,
        grouping: str,
    ) -> EvaluationResult:
        train_total, train_pos = self._train_pass(model, builder, grouping)
        model.finalize()
        metric_set = self._eval_pass(model, builder, grouping)

        return EvaluationResult(
            model_id=model.id,
            grouping=grouping,
            split_name=self.cfg.split_name,
            metrics=metric_set,
            class_balance={
                "train_total": train_total,
                "train_positive": train_pos,
            },
            artifacts={},
        )

    def _train_pass(
        self,
        model: SequenceModel,
        builder: SequenceBuilder,
        grouping: str,
    ) -> tuple[int, int]:
        train_total = 0
        train_pos = 0
        buffer: list[TemplateSequence] = []
        progress = make_spinner_progress(unit="train seqs")
        ctx = progress if self.cfg.show_progress else nullcontext()
        with ctx:
            task_id = (
                progress.add_task(f"train {model.id} ({grouping})")
                if self.cfg.show_progress
                else None
            )
            for seq in builder:
                if self.split(seq, builder.mode) != "train":
                    continue
                buffer.append(seq)
                if len(buffer) >= TRAIN_BATCH:
                    model.partial_train(buffer)
                    train_total += len(buffer)
                    train_pos += sum(s.label for s in buffer)
                    buffer.clear()
                if task_id is not None:
                    progress.update(task_id, advance=1)
            if buffer:
                model.partial_train(buffer)
                train_total += len(buffer)
                train_pos += sum(s.label for s in buffer)
                if task_id is not None:
                    progress.update(task_id, advance=len(buffer))
        return train_total, train_pos

    def _eval_pass(
        self,
        model: SequenceModel,
        builder: SequenceBuilder,
        grouping: str,
    ) -> MetricSet:
        metrics = MetricsComputer()
        eval_buffer: list[TemplateSequence] = []
        progress = make_spinner_progress(unit="eval seqs")
        ctx = progress if self.cfg.show_progress else nullcontext()
        with ctx:
            task_id = (
                progress.add_task(f"eval {model.id} ({grouping})")
                if self.cfg.show_progress
                else None
            )
            for seq in builder:
                if self.split(seq, builder.mode) != "test":
                    continue
                eval_buffer.append(seq)
                if len(eval_buffer) >= EVAL_BATCH:
                    probs = model.predict_proba(eval_buffer)
                    metrics.add_batch(eval_buffer, probs)
                    if task_id is not None:
                        progress.update(task_id, advance=len(eval_buffer))
                    eval_buffer.clear()
            if eval_buffer:
                probs = model.predict_proba(eval_buffer)
                metrics.add_batch(eval_buffer, probs)
                if task_id is not None:
                    progress.update(task_id, advance=len(eval_buffer))
        return metrics.compute()

    def _persist(self, result: EvaluationResult, grouping: str) -> None:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = self.cfg.artifact_root / grouping / result.model_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"metrics-{timestamp}.json"
        with Path.open(out_file, "w", encoding="utf-8") as f:
            json.dump(result.to_jsonable(), f, indent=2)
        result.artifacts["json"] = str(out_file)
        self.logger.info("Saved metrics to %s", out_file)
