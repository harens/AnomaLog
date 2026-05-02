"""Integration test for the DeepLog experiment runner path."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from experiments.runners.run_experiment import run_experiment

FIXTURE_ROOT = Path(__file__).parent / "experiment_fixtures" / "deeplog"
FIXTURE_LOG = Path(__file__).parent / "logs" / "deeplog_bgl_fixture.log"
EXPECTED_KEY_VOCABULARY_SIZE = 2
EXPECTED_PARAMETER_MODEL_COUNT = 2
PAPER_DEFAULT_HISTORY_SIZE = 10
PAPER_DEFAULT_TOP_G = 9
PAPER_DEFAULT_NUM_LAYERS = 2
PAPER_DEFAULT_HIDDEN_SIZE = 64


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


def _int_value(mapping: dict[str, object], key: str) -> int:
    value = mapping[key]
    assert isinstance(value, int)
    return value


def _float_value(mapping: dict[str, object], key: str) -> float:
    value = mapping[key]
    assert isinstance(value, int | float)
    return float(value)


def _list_value(mapping: dict[str, object], key: str) -> list[object]:
    value = mapping[key]
    assert isinstance(value, list)
    return list(value)


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
    sequence_count = _int_value(metrics, "sequence_count")
    train_sequence_count = _int_value(metrics, "train_sequence_count")
    test_sequence_count = _int_value(metrics, "test_sequence_count")
    ignored_sequence_count = _int_value(metrics, "ignored_sequence_count")
    assert sequence_count > 0
    assert train_sequence_count >= 0
    assert test_sequence_count >= 0
    assert ignored_sequence_count >= 0
    assert sequence_count == (
        train_sequence_count + test_sequence_count + ignored_sequence_count
    )
    next_event_prediction_raw = metrics["next_event_prediction"]
    assert isinstance(next_event_prediction_raw, dict)
    next_event_prediction = {
        str(key): value for key, value in next_event_prediction_raw.items()
    }
    assert next_event_prediction["task"] == "next_event_prediction"
    totals = _object_dict(next_event_prediction["totals"])
    top_k = _object_dict(next_event_prediction["top_k"])
    exclusions = _object_dict(next_event_prediction["exclusions"])
    assert _int_value(totals, "events_seen") > 0
    assert _int_value(totals, "events_eligible") >= 0
    assert 0.0 <= _float_value(totals, "coverage") <= 1.0
    assert 1 in _list_value(top_k, "k_values")
    assert "1" in _object_dict(top_k["hit_count"])
    assert "1" in _object_dict(top_k["accuracy"])
    assert _int_value(exclusions, "insufficient_history") >= 0
    assert _int_value(exclusions, "unknown_history") >= 0
    assert _int_value(exclusions, "unknown_target") >= 0
    assert next_event_prediction["vocabulary_policy"] == "full_dataset"


def _assert_deeplog_manifest(
    *,
    metrics: dict[str, object],
    sequence_config: dict[str, object],
    manifest: dict[str, object],
) -> None:
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
    sequence_split_summary = _object_dict(manifest["sequence_split_summary"])
    assert sequence_split_summary["train_on_normal_entities_only"] is True
    assert sequence_split_summary["requested_train_fraction"] == pytest.approx(
        sequence_config["train_fraction"],
    )
    assert sequence_split_summary["requested_test_fraction"] == pytest.approx(
        sequence_config["test_fraction"],
    )
    ignored_sequence_count = _int_value(metrics, "ignored_sequence_count")
    train_sequence_count = _int_value(metrics, "train_sequence_count")
    test_sequence_count = _int_value(metrics, "test_sequence_count")
    assert sequence_split_summary["ignored_sequence_count"] == ignored_sequence_count
    assert sequence_split_summary["train_pool_sequence_count"] == (
        train_sequence_count + ignored_sequence_count
    )
    assert sequence_split_summary["realised_train_sequence_count"] == (
        train_sequence_count
    )
    assert sequence_split_summary["excluded_from_train_count"] == (
        _int_value(sequence_split_summary, "train_pool_sequence_count")
        - train_sequence_count
    )
    assert _int_value(sequence_split_summary, "ineligible_train_pool_count") >= 0
    eligible_train_sequence_count = _int_value(
        sequence_split_summary,
        "eligible_train_sequence_count",
    )
    if eligible_train_sequence_count > 0:
        assert sequence_split_summary["effective_train_fraction_of_eligible"] == (
            pytest.approx(train_sequence_count / eligible_train_sequence_count)
        )
    scored_sequence_count = train_sequence_count + test_sequence_count
    if scored_sequence_count > 0:
        assert sequence_split_summary["effective_train_fraction_overall"] == (
            pytest.approx(train_sequence_count / _int_value(metrics, "sequence_count"))
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
    _assert_deeplog_manifest(
        metrics=metrics,
        sequence_config=manifest["sequence_config"],
        manifest=manifest,
    )

    assert len(predictions) == metrics["test_sequence_count"]
    assert all(prediction["split_label"] == "test" for prediction in predictions)
    for prediction in predictions:
        assert prediction["predicted_label"] in {0, 1}
        if prediction["predicted_label"] == 1:
            assert _float_value(prediction, "score") > 0.0
            key_findings = _anomalous_key_findings(prediction)
            parameter_findings = _anomalous_parameter_findings(prediction)
            assert key_findings or parameter_findings
        else:
            assert prediction["score"] == pytest.approx(0.0)
            assert not _anomalous_key_findings(prediction)
            assert not _anomalous_parameter_findings(prediction)

    assert "Fitting deeplog detector" in run_log
    assert "DeepLog resolved torch device:" in run_log
    assert "chronological train pool" in run_log
