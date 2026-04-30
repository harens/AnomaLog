"""Integration test for the Naive Bayes experiment runner path."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TypedDict

from experiments.runners.run_experiment import run_experiment

FIXTURE_ROOT = Path(__file__).parent / "experiment_fixtures" / "naive_bayes"
FIXTURE_LOG = Path(__file__).parent / "logs" / "tiny_bgl_happy_path.log"
EXPECTED_SEQUENCE_COUNT = 4
EXPECTED_TRAIN_SEQUENCE_COUNT = 2
EXPECTED_TEST_SEQUENCE_COUNT = 2
POSTERIOR_THRESHOLD = 0.4


class _PredictionRecord(TypedDict):
    entity_ids: list[str]
    event_count: int
    key_phrases: list[str]
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
            "key_phrases": [str(value) for value in raw["key_phrases"]],
            "label": int(raw["label"]),
            "predicted_label": int(raw["predicted_label"]),
            "score": float(raw["score"]),
            "split_label": str(raw["split_label"]),
            "window_id": int(raw["window_id"]),
        }
        predictions.append(prediction)
    return predictions


class _KeyPhrasesByClass(TypedDict):
    anomalous: list[str]
    normal: list[str]


def _model_key_phrases_by_class(
    manifest: dict[str, object],
) -> _KeyPhrasesByClass:
    raw_model_manifest = manifest.get("model_manifest")
    assert isinstance(raw_model_manifest, dict)
    model_manifest = {str(key): value for key, value in raw_model_manifest.items()}
    raw_key_phrases = model_manifest["key_phrases_by_class"]
    assert isinstance(raw_key_phrases, dict)
    key_phrases_by_class = {str(key): value for key, value in raw_key_phrases.items()}
    anomalous = key_phrases_by_class["anomalous"]
    normal = key_phrases_by_class["normal"]
    assert isinstance(anomalous, list)
    assert isinstance(normal, list)
    return {
        "anomalous": [str(phrase) for phrase in anomalous],
        "normal": [str(phrase) for phrase in normal],
    }


def _prepare_run_tree(tmp_path: Path) -> Path:
    """Copy the checked-in sweep fixture files into a writable temp tree.

    Args:
        tmp_path (Path): Temporary directory to populate with fixture configs.

    Returns:
        Path: Sweep config path inside the prepared temp tree.
    """
    sweep_config = tmp_path / "experiments" / "configs" / "sweeps" / "tiny_nb_run.toml"
    dataset_config = (
        tmp_path / "experiments" / "configs" / "datasets" / "tiny_dataset_nb.toml"
    )
    model_config = tmp_path / "experiments" / "configs" / "models" / "naive_bayes.toml"
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    sweep_config.parent.mkdir(parents=True)
    dataset_config.parent.mkdir(parents=True)
    model_config.parent.mkdir(parents=True)
    shutil.copy2(FIXTURE_LOG, log_dir / FIXTURE_LOG.name)
    shutil.copy2(FIXTURE_ROOT / "tiny_nb_run.toml", sweep_config)
    shutil.copy2(FIXTURE_ROOT / "tiny_dataset_nb.toml", dataset_config)
    shutil.copy2(FIXTURE_ROOT / "naive_bayes.toml", model_config)
    return sweep_config


def test_run_experiment_with_naive_bayes_emits_key_phrases(tmp_path: Path) -> None:
    """Naive Bayes runs should write phrase-aware predictions and model metadata.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for copied config fixtures.
    """
    sweep_config = _prepare_run_tree(tmp_path)

    [run_dir] = run_experiment(sweep_config, write_predictions=True)
    [rerun_dir] = run_experiment(
        sweep_config,
        force=True,
        write_predictions=True,
    )

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dir / "dataset_manifest.json").read_text(encoding="utf-8"),
    )
    predictions = _read_predictions(run_dir)
    rerun_predictions = _read_predictions(rerun_dir)
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    test_predictions = [
        prediction for prediction in predictions if prediction["split_label"] == "test"
    ]
    held_out_prediction = test_predictions[-1]
    key_phrases_by_class = _model_key_phrases_by_class(manifest)
    normal_key_phrases = key_phrases_by_class["normal"]

    assert run_dir.name == manifest["run_fingerprint"][:12]
    assert metrics["sequence_count"] == EXPECTED_SEQUENCE_COUNT
    assert metrics["train_sequence_count"] == EXPECTED_TRAIN_SEQUENCE_COUNT
    assert metrics["test_sequence_count"] == EXPECTED_TEST_SEQUENCE_COUNT
    assert manifest["model_manifest"]["detector"] == "naive_bayes"
    assert normal_key_phrases
    assert any(
        "ras app fatal" in phrase
        for phrase in manifest["model_manifest"]["key_phrases_by_class"]["anomalous"]
    )
    assert len(predictions) == EXPECTED_TEST_SEQUENCE_COUNT
    assert [prediction["window_id"] for prediction in predictions] == [2, 3]
    assert [prediction["split_label"] for prediction in predictions] == [
        "test",
        "test",
    ]
    assert held_out_prediction["label"] == 0
    assert held_out_prediction["predicted_label"] == 0
    assert held_out_prediction["score"] < POSTERIOR_THRESHOLD
    assert held_out_prediction["key_phrases"]
    assert any(
        phrase in normal_key_phrases for phrase in held_out_prediction["key_phrases"]
    )
    assert predictions == rerun_predictions
    assert "Fitting naive_bayes detector" in run_log
