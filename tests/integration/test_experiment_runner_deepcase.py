"""Integration test for the DeepCase experiment runner path."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from experiments.runners.run_experiment import run_experiment

FIXTURE_ROOT = Path(__file__).parent / "experiment_fixtures" / "deepcase"
FIXTURE_LOG = Path(__file__).parent / "logs" / "deepcase_fixture.log"
EXPECTED_CONTEXT_LENGTH = 2
EXPECTED_TIMEOUT_SECONDS = 86_400.0
EXPECTED_MIN_SAMPLES = 1


def _prepare_run_tree(tmp_path: Path) -> Path:
    sweep_config = tmp_path / "experiments" / "configs" / "sweeps" / "deepcase_run.toml"
    dataset_config = (
        tmp_path / "experiments" / "configs" / "datasets" / "deepcase_dataset.toml"
    )
    model_config = tmp_path / "experiments" / "configs" / "models" / "deepcase.toml"
    log_path = tmp_path / "logs" / "deepcase_fixture.log"

    log_path.parent.mkdir(parents=True)
    sweep_config.parent.mkdir(parents=True)
    dataset_config.parent.mkdir(parents=True)
    model_config.parent.mkdir(parents=True)

    shutil.copy2(FIXTURE_LOG, log_path)
    shutil.copy2(FIXTURE_ROOT / "deepcase_run.toml", sweep_config)
    dataset_config.write_text(
        'name = "deepcase_dataset"\n'
        'dataset_name = "deepcase-bgl-fixture"\n'
        'structured_parser = "bgl"\n'
        "\n[source]\n"
        'type = "local_dir"\n'
        'path = "."\n'
        'raw_logs_relpath = "logs/deepcase_fixture.log"\n'
        "\n[sequence]\n"
        'grouping = "entity"\n'
        "train_fraction = 0.5\n"
        "test_fraction = 0.5\n",
        encoding="utf-8",
    )
    shutil.copy2(FIXTURE_ROOT / "deepcase.toml", model_config)
    return sweep_config


def _read_predictions(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _assert_deepcase_outputs(
    *,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    model_manifest: dict[str, Any],
    prediction_diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> None:
    sequence_config = manifest["sequence_config"]
    sequence_split_summary = manifest["sequence_split_summary"]
    assert metrics["sequence_count"] > 0
    assert metrics["train_sequence_count"] >= 0
    assert metrics["test_sequence_count"] >= 0
    assert metrics["ignored_sequence_count"] >= 0
    assert metrics["sequence_count"] == (
        metrics["train_sequence_count"]
        + metrics["test_sequence_count"]
        + metrics["ignored_sequence_count"]
    )
    assert len(predictions) == metrics["test_sequence_count"]
    assert [prediction["split_label"] for prediction in predictions] == ["test"] * len(
        predictions,
    )
    assert prediction_diagnostics["event_count"] == sum(
        prediction["event_count"] for prediction in predictions
    )
    assert (
        prediction_diagnostics["confident_event_count"]
        + prediction_diagnostics["abstained_event_count"]
        == prediction_diagnostics["event_count"]
    )
    assert prediction_diagnostics[
        "sequence_confident_anomaly_count"
    ] + prediction_diagnostics[
        "sequence_confident_normal_count"
    ] + prediction_diagnostics["sequence_abstained_count"] == len(predictions)
    assert model_manifest["detector"] == "deepcase"
    assert model_manifest["context_length"] == EXPECTED_CONTEXT_LENGTH
    assert model_manifest["timeout_seconds"] == EXPECTED_TIMEOUT_SECONDS
    assert model_manifest["min_samples"] == EXPECTED_MIN_SAMPLES
    assert model_manifest["train_sample_count"] > 0
    assert model_manifest["train_event_vocabulary_size"] > 0
    assert model_manifest["prediction_diagnostics"] == prediction_diagnostics
    next_event_prediction = model_manifest["next_event_prediction"]
    assert isinstance(next_event_prediction, dict)
    assert next_event_prediction["task"] == "next_event_prediction"
    totals: Any = next_event_prediction["totals"]
    top_k: Any = next_event_prediction["top_k"]
    exclusions: Any = next_event_prediction["exclusions"]
    assert totals["events_seen"] > 0
    assert totals["events_eligible"] >= 0
    assert 0.0 <= totals["coverage"] <= 1.0
    assert 1 in top_k["k_values"]
    assert "1" in top_k["hit_count"]
    assert "1" in top_k["accuracy"]
    assert exclusions["insufficient_history"] == 0
    assert exclusions["unknown_history"] == 0
    assert exclusions["unknown_target"] == 0
    assert next_event_prediction["vocabulary_policy"] == "full_dataset"
    assert sequence_config["train_fraction"] == pytest.approx(
        sequence_split_summary["requested_train_fraction"],
    )
    assert model_manifest["prediction_diagnostics"] == prediction_diagnostics
    assert all(
        len(prediction["findings"]) == prediction["event_count"]
        for prediction in predictions
    )
    assert all("sequence_decision" in prediction for prediction in predictions)
    assert all("abstained_event_count" in prediction for prediction in predictions)
    assert all(
        "raw_score" in finding
        for prediction in predictions
        for finding in prediction["findings"]
    )
    assert all(
        "is_abstained" in finding
        for prediction in predictions
        for finding in prediction["findings"]
    )
    for prediction in predictions:
        if prediction["predicted_label"] == 1:
            assert prediction["score"] == pytest.approx(1.0)
        else:
            assert prediction["score"] == pytest.approx(0.0)


def test_run_experiment_with_deepcase_writes_event_findings(
    tmp_path: Path,
) -> None:
    """DeepCase runs should write manifest metadata and event findings.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for copied config fixtures.
    """
    sweep_config = _prepare_run_tree(tmp_path)

    [run_dir] = run_experiment(sweep_config, write_predictions=True)

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dir / "dataset_manifest.json").read_text(encoding="utf-8"),
    )
    predictions = _read_predictions(run_dir)
    run_log = (run_dir / "run.log").read_text(encoding="utf-8")

    model_manifest = manifest["model_manifest"]
    prediction_diagnostics = model_manifest["prediction_diagnostics"]

    assert isinstance(model_manifest, dict)
    assert isinstance(prediction_diagnostics, dict)
    _assert_deepcase_outputs(
        metrics=metrics,
        manifest=manifest,
        model_manifest=model_manifest,
        prediction_diagnostics=prediction_diagnostics,
        predictions=predictions,
    )
    assert manifest["sequence_config"]["grouping"] == "entity"
    assert "Fitting deepcase detector" in run_log
    assert "DeepCase resolved torch device: cpu" in run_log
    assert "Training DeepCase context builder" in run_log
    assert "Clustering DeepCase interpreter" in run_log
