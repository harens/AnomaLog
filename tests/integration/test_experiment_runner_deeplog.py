"""Integration test for the DeepLog experiment runner path."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments.runners.run_experiment import run_experiment

FIXTURE_ROOT = Path(__file__).parent / "experiment_fixtures" / "deeplog"
FIXTURE_LOG = Path(__file__).parent / "logs" / "deeplog_bgl_fixture.log"
EXPECTED_SEQUENCE_COUNT = 5
EXPECTED_TRAIN_SEQUENCE_COUNT = 2
EXPECTED_TEST_SEQUENCE_COUNT = 3
EXPECTED_TRUE_POSITIVES = 2
EXPECTED_TRUE_NEGATIVES = 1
EXPECTED_FALSE_POSITIVES = 0
EXPECTED_FALSE_NEGATIVES = 0
EXPECTED_ELIGIBLE_NORMAL_SEQUENCE_COUNT = 3
EXPECTED_KEY_VOCABULARY_SIZE = 2
EXPECTED_PARAMETER_MODEL_COUNT = 2
PAPER_DEFAULT_HISTORY_SIZE = 10
PAPER_DEFAULT_TOP_G = 9
PAPER_DEFAULT_NUM_LAYERS = 2
PAPER_DEFAULT_HIDDEN_SIZE = 64
EVENTS_PER_ENTITY = 40
KEY_ANOMALY_EVENT_INDEX = 30
PARAMETER_ANOMALY_EVENT_INDEX = 31
KEY_ANOMALY_TEMPLATE = "RAS APP FATAL gamma breaker meltdown shard <:NUM:> panic"


def _prepare_run_tree(tmp_path: Path) -> Path:
    sweep_config = tmp_path / "experiments" / "configs" / "sweeps" / "deeplog_run.toml"
    dataset_config = (
        tmp_path / "experiments" / "configs" / "datasets" / "deeplog_dataset.toml"
    )
    model_config = tmp_path / "experiments" / "configs" / "models" / "deeplog.toml"
    log_path = tmp_path / "logs" / "deeplog_bgl.log"

    log_path.parent.mkdir(parents=True)
    sweep_config.parent.mkdir(parents=True)
    dataset_config.parent.mkdir(parents=True)
    model_config.parent.mkdir(parents=True)

    shutil.copy2(FIXTURE_LOG, log_path)
    shutil.copy2(FIXTURE_ROOT / "deeplog_run.toml", sweep_config)
    shutil.copy2(FIXTURE_ROOT / "deeplog_dataset.toml", dataset_config)
    shutil.copy2(FIXTURE_ROOT / "deeplog.toml", model_config)
    return sweep_config


def _read_predictions(run_dir: Path) -> list[dict[str, object]]:
    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _object_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


def _object_list(value: object) -> list[object]:
    assert isinstance(value, list)
    return list(value)


def _prediction_details(prediction: dict[str, object]) -> dict[str, object]:
    return prediction


def _anomalous_key_findings(prediction: dict[str, object]) -> list[dict[str, object]]:
    findings = _object_list(_prediction_details(prediction)["findings"])
    anomalous_findings: list[dict[str, object]] = []
    for raw_finding in findings:
        finding = _object_dict(raw_finding)
        key_model_finding = finding.get("key_model_finding")
        if not isinstance(key_model_finding, dict):
            continue
        normalized_finding = _object_dict(key_model_finding)
        if bool(normalized_finding["is_anomalous"]):
            anomalous_findings.append(normalized_finding)
    return anomalous_findings


def _anomalous_parameter_findings(
    prediction: dict[str, object],
) -> list[dict[str, object]]:
    findings = _object_list(_prediction_details(prediction)["findings"])
    anomalous_findings: list[dict[str, object]] = []
    for raw_finding in findings:
        finding = _object_dict(raw_finding)
        parameter_model_finding = finding.get("parameter_model_finding")
        if not isinstance(parameter_model_finding, dict):
            continue
        normalized_finding = _object_dict(parameter_model_finding)
        if bool(normalized_finding["is_anomalous"]):
            anomalous_findings.append(normalized_finding)
    return anomalous_findings


def _assert_deeplog_metrics(metrics: dict[str, object]) -> None:
    assert metrics["sequence_count"] == EXPECTED_SEQUENCE_COUNT
    assert metrics["train_sequence_count"] == EXPECTED_TRAIN_SEQUENCE_COUNT
    assert metrics["test_sequence_count"] == EXPECTED_TEST_SEQUENCE_COUNT
    assert metrics["tp"] == EXPECTED_TRUE_POSITIVES
    assert metrics["tn"] == EXPECTED_TRUE_NEGATIVES
    assert metrics["fp"] == EXPECTED_FALSE_POSITIVES
    assert metrics["fn"] == EXPECTED_FALSE_NEGATIVES


def _assert_deeplog_manifest(manifest: dict[str, object]) -> None:
    model_manifest_raw = manifest["model_manifest"]
    assert isinstance(model_manifest_raw, dict)
    model_manifest = {str(key): value for key, value in model_manifest_raw.items()}
    parameter_models_raw = model_manifest["parameter_models"]
    assert isinstance(parameter_models_raw, list)
    parameter_models: list[dict[str, object]] = []
    for parameter_model in parameter_models_raw:
        assert isinstance(parameter_model, dict)
        parameter_models.append(
            {str(key): value for key, value in parameter_model.items()},
        )
    assert model_manifest["detector"] == "deeplog"
    assert model_manifest["history_size"] == PAPER_DEFAULT_HISTORY_SIZE
    assert model_manifest["top_g"] == PAPER_DEFAULT_TOP_G
    assert model_manifest["num_layers"] == PAPER_DEFAULT_NUM_LAYERS
    assert model_manifest["hidden_size"] == PAPER_DEFAULT_HIDDEN_SIZE
    assert model_manifest["train_key_vocabulary_size"] == EXPECTED_KEY_VOCABULARY_SIZE
    assert model_manifest["trained_parameter_model_count"] == (
        EXPECTED_PARAMETER_MODEL_COUNT
    )
    assert model_manifest["include_elapsed_time"] is True
    assert parameter_models[0]["feature_names"] == [
        "dt_prev_ms",
        "param_0",
    ]
    assert manifest["sequence_split_summary"] == {
        "requested_train_fraction": 0.4,
        "train_on_normal_entities_only": True,
        "eligible_train_sequence_count": EXPECTED_ELIGIBLE_NORMAL_SEQUENCE_COUNT,
        "effective_train_fraction_of_eligible": pytest.approx(2 / 3),
        "effective_train_fraction_overall": 0.4,
    }


def _select_deeplog_test_predictions(
    predictions: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    assert len(predictions) == EXPECTED_TEST_SEQUENCE_COUNT
    assert [prediction["split_label"] for prediction in predictions] == [
        "test",
        "test",
        "test",
    ]
    test_predictions = predictions
    normal_test_prediction = next(
        prediction for prediction in test_predictions if prediction["label"] == 0
    )
    key_anomaly_prediction = next(
        prediction
        for prediction in test_predictions
        if prediction["label"] == 1
        and bool(_prediction_details(prediction)["triggered_by_key_model"])
    )
    parameter_anomaly_prediction = next(
        prediction
        for prediction in test_predictions
        if prediction["label"] == 1
        and bool(_prediction_details(prediction)["triggered_by_parameter_model"])
    )
    return (
        normal_test_prediction,
        key_anomaly_prediction,
        parameter_anomaly_prediction,
    )


def test_run_experiment_with_deeplog_follows_paper_defaults(
    tmp_path: Path,
) -> None:
    """DeepLog runs should use the paper defaults and flag both anomaly modes.

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

    _assert_deeplog_metrics(metrics)
    _assert_deeplog_manifest(manifest)
    (
        normal_test_prediction,
        key_anomaly_prediction,
        parameter_anomaly_prediction,
    ) = _select_deeplog_test_predictions(predictions)

    assert normal_test_prediction["predicted_label"] == 0
    assert normal_test_prediction["score"] == pytest.approx(0.0)

    assert key_anomaly_prediction["predicted_label"] == 1
    key_findings = _anomalous_key_findings(key_anomaly_prediction)
    assert key_findings
    assert any(bool(finding["is_oov"]) for finding in key_findings)
    assert any(
        finding["actual_template"] == KEY_ANOMALY_TEMPLATE for finding in key_findings
    )

    assert parameter_anomaly_prediction["predicted_label"] == 1
    parameter_findings = _anomalous_parameter_findings(parameter_anomaly_prediction)
    assert parameter_findings
    primary_parameter_finding = next(
        finding
        for finding in parameter_findings
        if finding["event_index"] == PARAMETER_ANOMALY_EVENT_INDEX
        and finding["most_anomalous_feature"] == "param_0"
    )
    assert primary_parameter_finding["feature_names"] == ["dt_prev_ms", "param_0"]
    assert primary_parameter_finding["observed_vector"] == [1000.0, 999.0]
    predicted_vector = primary_parameter_finding["predicted_vector"]
    assert isinstance(predicted_vector, list)
    assert predicted_vector[0] == pytest.approx(1000.0)
    assert predicted_vector[1] == pytest.approx(20.0)

    assert "Fitting deeplog detector" in run_log
    assert "DeepLog resolved torch device:" in run_log
    assert "Normal-only training excludes anomalous entities from train" in run_log
