"""Integration test for the river multinomial Naive Bayes runner path."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TypedDict

from experiments.runners.run_experiment import run_experiment

FIXTURE_ROOT = Path(__file__).parent / "experiment_fixtures" / "river"
DATASET_FIXTURE_ROOT = Path(__file__).parent / "experiment_fixtures" / "naive_bayes"
FIXTURE_LOG = Path(__file__).parent / "logs" / "tiny_bgl_happy_path.log"
EXPECTED_SEQUENCE_COUNT = 4
EXPECTED_TRAIN_SEQUENCE_COUNT = 3
EXPECTED_TEST_SEQUENCE_COUNT = 1
POSTERIOR_THRESHOLD = 0.4


class _PredictionRecord(TypedDict):
    entity_ids: list[str]
    event_count: int
    label: int
    predicted_label: int
    score: float
    split_label: str
    window_id: int


def _read_predictions(run_dir: Path) -> list[_PredictionRecord]:
    predictions: list[_PredictionRecord] = []
    for line in (
        (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ):
        raw = json.loads(line)
        prediction: _PredictionRecord = {
            "entity_ids": [str(value) for value in raw["entity_ids"]],
            "event_count": int(raw["event_count"]),
            "label": int(raw["label"]),
            "predicted_label": int(raw["predicted_label"]),
            "score": float(raw["score"]),
            "split_label": str(raw["split_label"]),
            "window_id": int(raw["window_id"]),
        }
        predictions.append(prediction)
    return predictions


def test_run_experiment_with_river_multinomial_nb_writes_predictions(
    tmp_path: Path,
) -> None:
    """River runs should write probabilities and backend metadata.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for copied config fixtures.
    """
    run_config = tmp_path / "experiments" / "configs" / "runs" / "tiny_river_run.toml"
    dataset_config = (
        tmp_path / "experiments" / "configs" / "datasets" / "tiny_dataset_nb.toml"
    )
    model_config = tmp_path / "experiments" / "configs" / "models" / "river.toml"
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    run_config.parent.mkdir(parents=True)
    dataset_config.parent.mkdir(parents=True)
    model_config.parent.mkdir(parents=True)
    shutil.copy2(FIXTURE_LOG, log_dir / FIXTURE_LOG.name)
    shutil.copy2(FIXTURE_ROOT / "tiny_river_run.toml", run_config)
    shutil.copy2(DATASET_FIXTURE_ROOT / "tiny_dataset_nb.toml", dataset_config)
    shutil.copy2(FIXTURE_ROOT / "river.toml", model_config)

    run_dir = run_experiment(run_config)
    rerun_dir = run_experiment(run_config, force=True)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dir / "dataset_manifest.json").read_text(encoding="utf-8"),
    )
    predictions = _read_predictions(run_dir)
    rerun_predictions = _read_predictions(rerun_dir)
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    train_predictions = [
        prediction for prediction in predictions if prediction["split_label"] == "train"
    ]
    test_predictions = [
        prediction for prediction in predictions if prediction["split_label"] == "test"
    ]
    anomalous_training_prediction = next(
        prediction for prediction in train_predictions if prediction["label"] == 1
    )
    held_out_prediction = test_predictions[0]

    assert metrics["sequence_count"] == EXPECTED_SEQUENCE_COUNT
    assert metrics["train_sequence_count"] == EXPECTED_TRAIN_SEQUENCE_COUNT
    assert metrics["test_sequence_count"] == EXPECTED_TEST_SEQUENCE_COUNT
    assert manifest["model_manifest"]["detector"] == "river"
    assert manifest["model_manifest"]["backend"] == "river"
    assert manifest["model_manifest"]["river_estimator"] == "naive_bayes.MultinomialNB"
    assert [prediction["window_id"] for prediction in predictions] == [0, 1, 2, 3]
    assert [prediction["split_label"] for prediction in predictions] == [
        "train",
        "train",
        "train",
        "test",
    ]
    assert anomalous_training_prediction["predicted_label"] == 1
    assert anomalous_training_prediction["score"] >= POSTERIOR_THRESHOLD
    assert held_out_prediction["label"] == 0
    assert held_out_prediction["predicted_label"] == 0
    assert held_out_prediction["score"] < POSTERIOR_THRESHOLD
    assert predictions == rerun_predictions
    assert "Fitting river detector" in run_log
