"""Integration test for the DeepLog experiment runner path."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from experiments.config import ExperimentBundle, load_experiment_bundles
from experiments.models.deeplog.detector import DeepLogModelConfig
from experiments.runners.run_experiment import run_experiment

FIXTURE_ROOT = Path(__file__).parent / "experiment_fixtures" / "deeplog"
FIXTURE_LOG = Path(__file__).parent / "logs" / "deeplog_bgl_fixture.log"
EXPECTED_KEY_VOCABULARY_SIZE = 3
EXPECTED_PARAMETER_MODEL_COUNT = 0


def _prepare_run_tree(tmp_path: Path) -> Path:
    baseline_source = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "configs"
        / "models"
        / "template_frequency_default.toml"
    )
    sweep_config = tmp_path / "experiments" / "configs" / "sweeps" / "deeplog_run.toml"
    dataset_config = (
        tmp_path / "experiments" / "configs" / "datasets" / "deeplog_dataset.toml"
    )
    model_config = tmp_path / "experiments" / "configs" / "models" / "deeplog.toml"
    baseline_model_config = (
        tmp_path
        / "experiments"
        / "configs"
        / "models"
        / "template_frequency_default.toml"
    )
    log_path = tmp_path / "logs" / "deeplog_bgl.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    sweep_config.parent.mkdir(parents=True, exist_ok=True)
    dataset_config.parent.mkdir(parents=True, exist_ok=True)
    model_config.parent.mkdir(parents=True, exist_ok=True)
    baseline_model_config.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(FIXTURE_LOG, log_path)
    shutil.copy2(FIXTURE_ROOT / "deeplog_run.toml", sweep_config)
    shutil.copy2(FIXTURE_ROOT / "deeplog_dataset.toml", dataset_config)
    shutil.copy2(FIXTURE_ROOT / "deeplog.toml", model_config)
    baseline_model_config.write_text(
        baseline_source.read_text(encoding="utf-8") + "\nscore_threshold = 0.0\n",
        encoding="utf-8",
    )
    return sweep_config


def _bundle_named(bundles: list[ExperimentBundle], detector: str) -> ExperimentBundle:
    return next(bundle for bundle in bundles if bundle.model.detector == detector)


def _read_predictions(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def _detector_name(run_dir: Path) -> str:
    manifest = json.loads(
        (run_dir / "dataset_manifest.json").read_text(encoding="utf-8"),
    )
    model_manifest = _object_dict(manifest["model_manifest"])
    return str(model_manifest["detector"])


def _object_dict(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return {str(key): item for key, item in value.items()}


def _int_value(mapping: dict[str, Any], key: str) -> int:
    value = mapping[key]
    assert isinstance(value, int)
    return value


def _float_value(mapping: dict[str, Any], key: str) -> float:
    value = mapping[key]
    assert isinstance(value, int | float)
    return float(value)


def _list_value(mapping: dict[str, Any], key: str) -> list[Any]:
    value = mapping[key]
    assert isinstance(value, list)
    return list(value)


def _object_list(value: object) -> list[Any]:
    assert isinstance(value, list)
    return list(value)


def _prediction_details(prediction: dict[str, Any]) -> dict[str, Any]:
    return prediction


def _anomalous_key_findings(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    findings = _object_list(_prediction_details(prediction)["findings"])
    anomalous_findings: list[dict[str, Any]] = []
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
    prediction: dict[str, Any],
) -> list[dict[str, Any]]:
    findings = _object_list(_prediction_details(prediction)["findings"])
    anomalous_findings: list[dict[str, Any]] = []
    for raw_finding in findings:
        finding = _object_dict(raw_finding)
        parameter_model_finding = finding.get("parameter_model_finding")
        if not isinstance(parameter_model_finding, dict):
            continue
        normalized_finding = _object_dict(parameter_model_finding)
        if bool(normalized_finding["is_anomalous"]):
            anomalous_findings.append(normalized_finding)
    return anomalous_findings


def _assert_deeplog_metrics(metrics: dict[str, Any]) -> None:
    _assert_deeplog_metric_metadata(metrics)
    _assert_deeplog_sequence_counts(metrics)
    _assert_deeplog_metric_blocks(metrics)


def _assert_deeplog_metric_metadata(metrics: dict[str, Any]) -> None:
    assert "accuracy" not in metrics
    assert "f1" not in metrics
    assert "tp" not in metrics
    assert "tn" not in metrics
    assert "fp" not in metrics
    assert "fn" not in metrics
    assert "next_event_prediction" not in metrics
    assert metrics["evaluation_unit"] == "continuous_event_stream"
    assert metrics["primary_metric_scope"] == "event_level_detection"
    assert metrics["prediction_unit"] == "event"
    assert metrics["label_unit"] == "event"
    assert "primary_metrics" not in metrics
    assert "legacy_metrics" not in metrics


def _assert_deeplog_sequence_counts(metrics: dict[str, Any]) -> None:
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


def _assert_deeplog_metric_blocks(metrics: dict[str, Any]) -> None:
    metric_blocks_raw = metrics["metric_blocks"]
    assert isinstance(metric_blocks_raw, dict)
    metric_blocks = {str(key): value for key, value in metric_blocks_raw.items()}
    _assert_deeplog_event_block(metric_blocks)
    _assert_deeplog_next_event_block(metric_blocks, metrics)
    _assert_deeplog_sequence_block(metric_blocks)


def _assert_deeplog_event_block(metric_blocks: dict[str, Any]) -> None:
    event_level_detection = _object_dict(metric_blocks["event_level_detection"])
    assert event_level_detection["status"] == "valid"
    assert _int_value(event_level_detection["class_counts"], "normal") >= 0
    assert _int_value(event_level_detection["class_counts"], "anomalous") >= 0
    assert event_level_detection["headline_metrics"]["precision"] >= 0.0
    assert event_level_detection["headline_metrics"]["recall"] >= 0.0
    assert event_level_detection["headline_metrics"]["f1"] >= 0.0


def _assert_deeplog_sequence_block(metric_blocks: dict[str, Any]) -> None:
    sequence_level_detection = _object_dict(metric_blocks["sequence_level_detection"])
    assert sequence_level_detection["status"] == "invalid"
    assert sequence_level_detection["invalid_reason"] == "single_class_test_set"
    assert sequence_level_detection["headline_metrics"] == {}


def _assert_deeplog_next_event_block(
    metric_blocks: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    next_event_prediction = _object_dict(metric_blocks["next_event_prediction"])
    assert next_event_prediction["status"] == "valid"
    totals = _object_dict(next_event_prediction["diagnostics"]["totals"])
    top_k = _object_dict(next_event_prediction["diagnostics"]["top_k"])
    exclusions = _object_dict(next_event_prediction["diagnostics"]["exclusions"])
    segment_diagnostics_raw = next_event_prediction["diagnostics"][
        "segment_diagnostics"
    ]
    assert isinstance(segment_diagnostics_raw, dict)
    segment_diagnostics = {
        str(key): value for key, value in segment_diagnostics_raw.items()
    }
    hit_count = _object_dict(top_k["hit_count"])
    accuracy = _object_dict(top_k["accuracy"])
    assert _int_value(totals, "events_seen") >= _int_value(
        metrics,
        "test_sequence_count",
    )
    assert _int_value(totals, "events_eligible") > 0
    assert 0.0 < _float_value(totals, "coverage") <= 1.0
    assert 1 in _list_value(top_k, "k_values")
    assert "1" in hit_count
    assert "1" in accuracy
    assert any(_int_value(hit_count, key) > 0 for key in hit_count)
    assert any(_float_value(accuracy, key) > 0.0 for key in accuracy)
    assert _int_value(exclusions, "insufficient_history") >= 0
    assert _int_value(exclusions, "unknown_history") >= 0
    assert _int_value(exclusions, "unknown_target") >= 0
    assert next_event_prediction["diagnostics"]["vocabulary_policy"] == "full_dataset"
    assert _int_value(segment_diagnostics, "segment_count") >= 1
    assert _int_value(segment_diagnostics, "history_size") > 0
    assert (
        _int_value(
            segment_diagnostics,
            "expected_insufficient_history_from_segments",
        )
        >= 0
    )


def _assert_deeplog_manifest(
    *,
    metrics: dict[str, Any],
    bundle: ExperimentBundle,
    manifest: dict[str, Any],
) -> None:
    assert isinstance(bundle.model, DeepLogModelConfig)
    deeplog_model = bundle.model
    model_manifest_raw = manifest["model_manifest"]
    assert isinstance(model_manifest_raw, dict)
    model_manifest = {str(key): value for key, value in model_manifest_raw.items()}
    parameter_models_raw = model_manifest["parameter_models"]
    assert isinstance(parameter_models_raw, list)
    parameter_models: list[dict[str, Any]] = []
    for parameter_model in parameter_models_raw:
        assert isinstance(parameter_model, dict)
        parameter_models.append(
            {str(key): value for key, value in parameter_model.items()},
        )
    assert model_manifest["detector"] == deeplog_model.detector
    assert model_manifest["history_size"] == deeplog_model.history_size
    assert model_manifest["top_g"] == deeplog_model.top_g
    assert model_manifest["num_layers"] == deeplog_model.num_layers
    assert model_manifest["hidden_size"] == deeplog_model.hidden_size
    assert model_manifest["train_key_vocabulary_size"] == EXPECTED_KEY_VOCABULARY_SIZE
    assert model_manifest["trained_parameter_model_count"] == (
        EXPECTED_PARAMETER_MODEL_COUNT
    )
    assert model_manifest["include_elapsed_time"] == deeplog_model.include_elapsed_time
    assert parameter_models == []
    sequence_config = bundle.dataset.sequence
    sequence_split_summary = _object_dict(manifest["sequence_split_summary"])
    assert sequence_split_summary.get("train_on_normal_entities_only") is None
    assert sequence_split_summary["requested_train_fraction"] == pytest.approx(
        sequence_config.train_fraction,
    )
    assert sequence_split_summary["requested_test_fraction"] == pytest.approx(
        sequence_config.test_fraction,
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
    assert manifest["evaluation_unit"] == "continuous_event_stream"
    assert manifest["primary_metric_scope"] == "event_level_detection"
    assert "event_level_detection" in manifest["available_metric_scopes"]
    assert "sequence_level_detection" in manifest["available_metric_scopes"]
    assert manifest["split_policy"]["train_fraction"] == pytest.approx(
        sequence_config.train_fraction,
    )
    assert manifest["stream_segment_policy"]["mode"] == "continuous_event_stream"


def test_run_experiment_with_deeplog_matches_resolved_config(
    tmp_path: Path,
) -> None:
    """DeepLog runs should match the resolved config and flag both anomaly modes.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for copied config fixtures.
    """
    sweep_config = _prepare_run_tree(tmp_path)
    bundles = load_experiment_bundles(sweep_config)
    deeplog_bundle = _bundle_named(bundles, "deeplog")
    baseline_bundle = _bundle_named(bundles, "template_frequency")

    run_dirs = run_experiment(sweep_config, write_predictions=True)
    assert {_detector_name(run_dir) for run_dir in run_dirs} == {
        deeplog_bundle.model.detector,
        baseline_bundle.model.detector,
    }

    deeplog_run_dir = next(
        run_dir
        for run_dir in run_dirs
        if _detector_name(run_dir) == deeplog_bundle.model.detector
    )
    baseline_run_dir = next(
        run_dir
        for run_dir in run_dirs
        if _detector_name(run_dir) == baseline_bundle.model.detector
    )

    metrics = json.loads((deeplog_run_dir / "metrics.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (deeplog_run_dir / "dataset_manifest.json").read_text(encoding="utf-8"),
    )
    predictions = _read_predictions(deeplog_run_dir)
    run_log = (deeplog_run_dir / "run.log").read_text(encoding="utf-8")
    baseline_metrics = json.loads(
        (baseline_run_dir / "metrics.json").read_text(encoding="utf-8"),
    )

    _assert_deeplog_metrics(metrics)
    _assert_deeplog_manifest(
        metrics=metrics,
        bundle=deeplog_bundle,
        manifest=manifest,
    )

    assert len(predictions) == metrics["test_sequence_count"]
    assert {prediction["split_label"] for prediction in predictions} <= {
        "train",
        "test",
    }
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
    assert "Primary metric scope: event_level_detection" in run_log
    assert baseline_metrics["primary_metric_scope"] == "sequence_level_detection"
    assert baseline_metrics["prediction_unit"] == "sequence"
    assert baseline_metrics["label_unit"] == "sequence"
