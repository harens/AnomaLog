"""Integration tests for the config-driven experiment runner."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TypedDict

import pytest

from experiments.results import sha256_for_file
from experiments.runners.run_experiment import run_experiment

FIXTURE_ROOT = Path(__file__).parent / "experiment_fixtures" / "template_frequency"
FIXTURE_LOG = Path(__file__).parent / "logs" / "tiny_bgl_happy_path.log"
EXPECTED_SEQUENCE_COUNT = 4
EXPECTED_TRAIN_SEQUENCE_COUNT = 2
EXPECTED_TEST_SEQUENCE_COUNT = 2
EXPECTED_STRUCTURED_ROWS = 8
FINGERPRINT_HEX_LENGTH = 64
TEMPLATE_FREQUENCY_SCORE_THRESHOLD = 1.2


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


def _assert_template_frequency_predictions(
    predictions: list[_PredictionRecord],
    *,
    score_threshold: float,
) -> None:
    test_predictions = [
        prediction for prediction in predictions if prediction["split_label"] == "test"
    ]

    assert len(predictions) == EXPECTED_TEST_SEQUENCE_COUNT
    assert [prediction["window_id"] for prediction in predictions] == [2, 3]
    assert [prediction["split_label"] for prediction in predictions] == [
        "test",
        "test",
    ]
    assert len(test_predictions) == EXPECTED_TEST_SEQUENCE_COUNT
    assert [prediction["label"] for prediction in test_predictions] == [0, 0]
    flagged_test_prediction = next(
        prediction
        for prediction in test_predictions
        if prediction["predicted_label"] == 1
    )
    normal_test_prediction = next(
        prediction
        for prediction in test_predictions
        if prediction["predicted_label"] == 0
    )
    assert flagged_test_prediction["score"] > score_threshold
    assert normal_test_prediction["score"] < score_threshold


def test_run_experiment_writes_reproducible_result_bundle(tmp_path: Path) -> None:
    """A run config should materialize dataset, metrics, predictions, and metadata.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for copied config fixtures.
    """
    run_config = tmp_path / "experiments" / "configs" / "runs" / "tiny_run.toml"
    dataset_config = (
        tmp_path / "experiments" / "configs" / "datasets" / "tiny_dataset.toml"
    )
    model_config = (
        tmp_path / "experiments" / "configs" / "models" / "template_frequency.toml"
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    run_config.parent.mkdir(parents=True)
    dataset_config.parent.mkdir(parents=True)
    model_config.parent.mkdir(parents=True)
    shutil.copy2(FIXTURE_LOG, log_dir / FIXTURE_LOG.name)
    shutil.copy2(FIXTURE_ROOT / "tiny_run.toml", run_config)
    shutil.copy2(FIXTURE_ROOT / "tiny_dataset.toml", dataset_config)
    model_config.write_text(
        'name = "template_frequency"\n'
        'detector = "template_frequency"\n'
        f"score_threshold = {TEMPLATE_FREQUENCY_SCORE_THRESHOLD}\n",
        encoding="utf-8",
    )

    run_dir = run_experiment(run_config)
    rerun_dir = run_experiment(run_config, force=True)

    assert run_dir.exists()
    assert rerun_dir == run_dir
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dir / "dataset_manifest.json").read_text(encoding="utf-8"),
    )
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    run_config_payload = json.loads(
        (run_dir / "run_config.json").read_text(encoding="utf-8"),
    )
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")
    predictions = _read_predictions(run_dir)
    rerun_metrics = json.loads((rerun_dir / "metrics.json").read_text(encoding="utf-8"))
    rerun_manifest = json.loads(
        (rerun_dir / "dataset_manifest.json").read_text(encoding="utf-8"),
    )
    rerun_predictions = _read_predictions(rerun_dir)

    assert run_dir.name == manifest["run_fingerprint"][:12]
    assert metrics["sequence_count"] == EXPECTED_SEQUENCE_COUNT
    assert metrics["train_sequence_count"] == EXPECTED_TRAIN_SEQUENCE_COUNT
    assert metrics["test_sequence_count"] == EXPECTED_TEST_SEQUENCE_COUNT
    assert manifest["dataset_variant"] == "tiny_dataset"
    assert len(manifest["dataset_fingerprint"]) == FINGERPRINT_HEX_LENGTH
    assert manifest["structured_rows"] == EXPECTED_STRUCTURED_ROWS
    assert manifest["sequence_split_counts"] == {
        "train": EXPECTED_TRAIN_SEQUENCE_COUNT,
        "test": EXPECTED_TEST_SEQUENCE_COUNT,
    }
    assert manifest["sequence_split_summary"] == {
        "requested_train_fraction": 0.34,
        "eligible_train_sequence_count": EXPECTED_SEQUENCE_COUNT,
        "effective_train_fraction_of_eligible": 0.5,
        "effective_train_fraction_overall": 0.5,
    }
    assert (
        manifest["model_manifest"]["score_threshold"]
        == TEMPLATE_FREQUENCY_SCORE_THRESHOLD
    )
    assert manifest["raw_logs"]["sha256"] == sha256_for_file(
        tmp_path / "logs" / FIXTURE_LOG.name,
    )
    assert environment["repository"]["root"] == tmp_path.as_posix()
    assert environment["run_fingerprint"] == manifest["run_fingerprint"]
    assert run_config_payload["run"]["name"] == "tiny_runner_smoke"
    assert "Building dataset tiny-bgl-runner" in run_log
    assert "Wrote experiment artifacts" in run_log
    _assert_template_frequency_predictions(
        predictions,
        score_threshold=TEMPLATE_FREQUENCY_SCORE_THRESHOLD,
    )
    assert metrics == rerun_metrics
    assert manifest == rerun_manifest
    assert predictions == rerun_predictions


@pytest.mark.allow_no_new_coverage
def test_run_experiment_errors_when_normal_only_train_fraction_is_impossible(
    tmp_path: Path,
) -> None:
    """Normal-only entity splits should fail when the overall target is impossible.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for copied config fixtures.
    """
    # This protects experiment-runner behavior outside the configured
    # `anomalog` coverage target.
    run_config = tmp_path / "experiments" / "configs" / "runs" / "tiny_run.toml"
    dataset_config = (
        tmp_path / "experiments" / "configs" / "datasets" / "tiny_dataset.toml"
    )
    model_config = (
        tmp_path / "experiments" / "configs" / "models" / "template_frequency.toml"
    )
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    run_config.parent.mkdir(parents=True)
    dataset_config.parent.mkdir(parents=True)
    model_config.parent.mkdir(parents=True)
    shutil.copy2(FIXTURE_LOG, log_dir / FIXTURE_LOG.name)
    shutil.copy2(FIXTURE_ROOT / "tiny_run.toml", run_config)
    dataset_config.write_text(
        'name = "tiny_dataset"\n'
        'dataset_name = "tiny-bgl-runner"\n'
        'structured_parser = "bgl"\n'
        "\n[source]\n"
        'type = "local_dir"\n'
        'path = "."\n'
        'raw_logs_relpath = "logs/tiny_bgl_happy_path.log"\n'
        "\n[sequence]\n"
        'grouping = "entity"\n'
        "train_fraction = 0.8\n"
        "train_on_normal_entities_only = true\n",
        encoding="utf-8",
    )
    shutil.copy2(FIXTURE_ROOT / "template_frequency.toml", model_config)

    with pytest.raises(
        ValueError,
        match="Requested train fraction is impossible",
    ):
        run_experiment(run_config)
