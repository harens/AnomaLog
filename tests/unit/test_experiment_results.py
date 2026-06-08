"""Tests for experiment result manifest helpers."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import msgspec
import pytest

import experiments.results as experiment_results
from anomalog.cache import CachePathsConfig
from anomalog.parsers.template import IdentityTemplateParser, TemplatedDataset
from anomalog.sequences import (
    EntitySequenceBuilder,
    RawEntrySplitSummary,
    SequenceSplitSummary,
)
from experiments.config import load_experiment_bundles
from experiments.models.base import ModelManifest, ModelRunSummary, SequenceSummary
from experiments.models.metric_reporting import MetricBlock
from experiments.models.metric_schema import EvaluationUnit, MetricScope, MetricStatus
from tests.unit.helpers import (
    InMemoryStructuredSink,
    NullStructuredParser,
    label_lookup,
)


@pytest.mark.allow_no_new_coverage
def test_build_sequence_split_summary_exposes_effective_fraction_for_normal_only() -> (
    None
):
    """Normal-only split summaries should show requested and effective fractions."""
    # This protects experiment-layer manifest metadata outside the configured
    # `anomalog` coverage target.
    expected_requested_train_fraction = 0.5
    expected_requested_test_fraction = 0.5
    expected_sequence_count = 10
    expected_train_sequence_count = 5
    expected_test_sequence_count = 3
    expected_ignored_sequence_count = 2
    expected_train_pool_sequence_count = 7
    expected_ineligible_train_pool_count = 1
    expected_eligible_train_sequence_count = 6
    expected_effective_eligible_train_fraction = round(
        expected_train_sequence_count / expected_eligible_train_sequence_count,
        8,
    )
    expected_effective_overall_train_fraction = round(
        expected_train_sequence_count / expected_sequence_count,
        8,
    )
    summary = experiment_results.build_sequence_split_summary(
        EntitySequenceBuilder(
            sink=InMemoryStructuredSink(
                dataset_name="demo",
                raw_dataset_path=Path("raw.log"),
                parser=NullStructuredParser(),
                rows=[],
            ),
            infer_template=lambda _: ("", ()),
            label_for_group=lambda _: 0,
            train_frac=expected_requested_train_fraction,
            test_frac=0.5,
            train_on_normal_entities_only=True,
        ),
        sequence_summary=SequenceSummary(
            sequence_count=expected_sequence_count,
            train_sequence_count=expected_train_sequence_count,
            test_sequence_count=expected_test_sequence_count,
            ignored_label_counts={0: 1, 1: 1},
            ignored_sequence_count=expected_ignored_sequence_count,
            train_label_counts={0: expected_train_sequence_count},
            test_label_counts={0: 1, 1: 4},
        ),
    )

    assert summary.requested_train_fraction == expected_requested_train_fraction
    assert summary.requested_test_fraction == pytest.approx(
        expected_requested_test_fraction,
    )
    assert summary.train_pool_sequence_count == expected_train_pool_sequence_count
    assert summary.ineligible_train_pool_count == expected_ineligible_train_pool_count
    assert summary.realised_train_sequence_count == expected_train_sequence_count
    assert (
        summary.eligible_train_sequence_count == expected_eligible_train_sequence_count
    )
    assert summary.ignored_sequence_count == expected_ignored_sequence_count
    assert (
        summary.effective_train_fraction_of_eligible
        == expected_effective_eligible_train_fraction
    )
    assert (
        summary.effective_train_fraction_overall
        == expected_effective_overall_train_fraction
    )


def test_build_run_metrics_report_drops_debug_only_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default metrics reports should keep the paper-facing next-event summary.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patch helper used to stub the report
            builders.
    """
    bundle = next(
        bundle
        for bundle in load_experiment_bundles(
            Path("experiments/configs/datasets/bgl/entity_chronological.toml"),
        )
        if bundle.model.detector == "deeplog"
    )
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=Path("raw.log"),
        parser=NullStructuredParser(),
        rows=[],
    )
    templated = TemplatedDataset(
        sink=sink,
        cache_paths=CachePathsConfig(
            data_root=Path("data"),
            cache_root=Path("cache"),
        ),
        template_parser=IdentityTemplateParser(),
        anomaly_labels=label_lookup(),
    )
    sequences = templated.sequences()
    model_summary = ModelRunSummary(
        metrics={},
        model_manifest=ModelManifest(
            detector="deeplog",
            train_sequence_count=0,
            test_sequence_count=0,
            train_label_counts={},
            test_label_counts={},
            ignored_sequence_count=0,
        ),
        sequence_summary=SequenceSummary(
            sequence_count=0,
            train_sequence_count=0,
            test_sequence_count=0,
            train_label_counts={},
            test_label_counts={},
        ),
    )

    monkeypatch.setattr(
        experiment_results,
        "build_metric_metadata",
        lambda **_: {
            "evaluation_unit": EvaluationUnit.SEQUENCE.value,
            "split_policy": {},
            "stream_segment_policy": {},
            "primary_metric_scope": None,
        },
    )
    monkeypatch.setattr(
        experiment_results,
        "_build_metric_blocks",
        lambda **_: {
            MetricScope.NEXT_EVENT_PREDICTION: MetricBlock(
                prediction_unit=EvaluationUnit.NEXT_EVENT,
                label_unit=EvaluationUnit.NEXT_EVENT,
                status=MetricStatus.VALID,
                diagnostics={
                    "task": "next_event_prediction",
                    "totals": {"coverage": 0.5},
                    "top_k": {
                        "k_values": [1, 2],
                        "hit_count": {"1": 3},
                        "accuracy": {},
                    },
                    "classification_top1_macro": {"precision": 0.1},
                    "classification_top1_weighted": {"precision": 0.2},
                    "exclusions": {"insufficient_history": 4},
                    "segment_diagnostics": {"segment_count": 2},
                    "vocabulary_policy": "full_dataset",
                },
            ),
        },
    )

    report: Any = experiment_results.build_run_metrics_report(
        bundle=bundle,
        sequences=sequences,
        model_summary=model_summary,
    )
    diagnostics = report["metric_blocks"]["next_event_prediction"]["diagnostics"]

    assert "classification_top1_macro" not in diagnostics
    assert "segment_diagnostics" not in diagnostics
    assert diagnostics["top_k"] == {"k_values": [1, 2], "accuracy": {}}
    assert "hit_count" not in diagnostics["top_k"]
    assert diagnostics["classification_top1_weighted"] == {"precision": 0.2}


def test_build_run_metrics_report_reuses_precomputed_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Precomputed metric reports should bypass the expensive report builder."""

    @dataclass(slots=True)
    class _SummaryCache:
        split_summary: SequenceSplitSummary | None = None
        raw_entry_split_summary: RawEntrySplitSummary | None = None
        metric_report: dict[str, object] | None = None

    bundle = next(
        bundle
        for bundle in load_experiment_bundles(
            Path("experiments/configs/datasets/bgl/entity_chronological.toml"),
        )
        if bundle.model.detector == "deeplog"
    )
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=Path("raw.log"),
        parser=NullStructuredParser(),
        rows=[],
    )
    templated = TemplatedDataset(
        sink=sink,
        cache_paths=CachePathsConfig(
            data_root=Path("data"),
            cache_root=Path("cache"),
        ),
        template_parser=IdentityTemplateParser(),
        anomaly_labels=label_lookup(),
    )
    sequences = templated.sequences()
    model_summary = ModelRunSummary(
        metrics={},
        model_manifest=ModelManifest(
            detector="deeplog",
            train_sequence_count=0,
            test_sequence_count=0,
            train_label_counts={},
            test_label_counts={},
            ignored_sequence_count=0,
        ),
        sequence_summary=SequenceSummary(
            sequence_count=0,
            train_sequence_count=0,
            test_sequence_count=0,
            train_label_counts={},
            test_label_counts={},
        ),
    )
    precomputed_report: dict[str, object] = {"sequence_count": 0}
    monkeypatch.setattr(
        experiment_results,
        "_build_metric_blocks",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("metric blocks should not be rebuilt"),
        ),
    )

    report = experiment_results.build_run_metrics_report(
        bundle=bundle,
        sequences=sequences,
        model_summary=model_summary,
        cached_summaries=_SummaryCache(metric_report=precomputed_report),
    )

    assert report == precomputed_report


def test_write_run_outputs_reuses_cached_split_summaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cached split summaries should avoid a second sequence replay."""
    bundle = next(
        bundle
        for bundle in load_experiment_bundles(
            Path("experiments/configs/datasets/bgl/entity_chronological.toml"),
        )
        if bundle.model.detector == "deeplog"
    )
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=tmp_path / "raw.log",
        parser=NullStructuredParser(),
        rows=[],
    )
    sink.raw_dataset_path.write_text("", encoding="utf-8")
    templated = TemplatedDataset(
        sink=sink,
        cache_paths=CachePathsConfig(
            data_root=tmp_path / "data",
            cache_root=tmp_path / "cache",
        ),
        template_parser=IdentityTemplateParser(),
        anomaly_labels=label_lookup(),
    )
    sequences = templated.sequences()
    result_paths = experiment_results.ResultPaths(
        run_fingerprint="fingerprint",
        run_root=tmp_path,
        run_dir=tmp_path,
        config_path=tmp_path / "experiment_config.json",
        dataset_manifest_path=tmp_path / "dataset_manifest.json",
        metrics_path=tmp_path / "metrics.json",
        predictions_path=tmp_path / "predictions.jsonl",
        environment_path=tmp_path / "environment.json",
        run_log_path=tmp_path / "run.log",
    )
    model_summary = ModelRunSummary(
        metrics={"sequence_count": 0},
        model_manifest=ModelManifest(
            detector="deeplog",
            train_sequence_count=0,
            test_sequence_count=0,
            train_label_counts={},
            test_label_counts={},
            ignored_sequence_count=0,
        ),
        sequence_summary=SequenceSummary(
            sequence_count=0,
            train_sequence_count=0,
            test_sequence_count=0,
            train_label_counts={},
            test_label_counts={},
        ),
    )
    context = experiment_results.ResultWriteContext(
        bundle=bundle,
        templated=templated,
        sequences=sequences,
        model_summary=model_summary,
        result_paths=result_paths,
    )
    split_summary = SequenceSplitSummary(
        requested_train_fraction=0.5,
        requested_test_fraction=0.5,
        train_on_normal_entities_only=None,
        train_pool_sequence_count=0,
        ineligible_train_pool_count=0,
        realised_train_sequence_count=0,
        excluded_from_train_count=0,
        eligible_train_sequence_count=0,
        ignored_sequence_count=0,
        effective_train_fraction_of_eligible=0.0,
        effective_train_fraction_overall=0.0,
    )
    raw_entry_split_summary = RawEntrySplitSummary(
        split_mode="raw_entry_prefix_count",
        application_order="before_grouping",
        cutoff_entry_index=0,
        train_raw_entry_count=0,
        train_normal_entry_count=0,
        train_anomalous_entry_count=0,
        test_raw_entry_count=0,
        test_normal_entry_count=0,
        test_anomalous_entry_count=0,
    )
    precomputed_report: dict[str, object] = {"sequence_count": 0}
    monkeypatch.setattr(
        experiment_results,
        "build_sequence_split_summary",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("split summary should not be rebuilt"),
        ),
    )
    monkeypatch.setattr(
        type(sequences),
        "build_raw_entry_split_summary",
        lambda _self: (_ for _ in ()).throw(
            AssertionError("raw-entry split summary should not be rebuilt"),
        ),
    )
    monkeypatch.setattr(
        experiment_results,
        "build_environment_metadata",
        lambda **_: {"environment": "metadata"},
    )

    experiment_results.write_run_outputs(
        context=context,
        split_summary=split_summary,
        raw_entry_split_summary=raw_entry_split_summary,
        metric_report=precomputed_report,
    )

    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8")) == {
        "sequence_count": 0,
    }


def test_prepare_result_paths_places_reruns_under_attempt_directories() -> None:
    """Reruns should keep the fingerprint root stable and add attempt folders."""
    bundle = next(
        bundle
        for bundle in load_experiment_bundles(
            Path("experiments/configs/datasets/bgl/entity_chronological.toml"),
        )
        if bundle.model.detector == "deeplog"
    )

    base_paths = experiment_results.prepare_result_paths(bundle)
    rerun_paths = experiment_results.prepare_result_paths(bundle, run_attempt=3)

    assert base_paths.run_dir == base_paths.run_root
    assert rerun_paths.run_root == base_paths.run_root
    assert rerun_paths.run_dir.parent == base_paths.run_root
    assert rerun_paths.run_dir.name == "attempt-0003"
    assert rerun_paths.config_path.parent == rerun_paths.run_dir


def test_build_metric_metadata_keeps_only_manifest_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persisted metric metadata should stay thin and manifest-oriented.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patch helper used to stub the report
            builders.
    """
    bundle = next(
        bundle
        for bundle in load_experiment_bundles(
            Path("experiments/configs/datasets/bgl/entity_chronological.toml"),
        )
        if bundle.model.detector == "deeplog"
    )
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=Path("raw.log"),
        parser=NullStructuredParser(),
        rows=[],
    )
    templated = TemplatedDataset(
        sink=sink,
        cache_paths=CachePathsConfig(
            data_root=Path("data"),
            cache_root=Path("cache"),
        ),
        template_parser=IdentityTemplateParser(),
        anomaly_labels=label_lookup(),
    )
    sequences = templated.sequences()
    metric_block = MetricBlock(
        prediction_unit=EvaluationUnit.SEQUENCE,
        label_unit=EvaluationUnit.SEQUENCE,
        status=MetricStatus.VALID,
    )
    model_summary = ModelRunSummary(
        metrics={},
        model_manifest=ModelManifest(
            detector="deeplog",
            train_sequence_count=0,
            test_sequence_count=0,
            train_label_counts={},
            test_label_counts={},
            ignored_sequence_count=0,
        ),
        sequence_summary=SequenceSummary(
            sequence_count=0,
            train_sequence_count=0,
            test_sequence_count=0,
            train_label_counts={},
            test_label_counts={},
        ),
    )
    monkeypatch.setattr(
        experiment_results,
        "_build_metric_blocks",
        lambda **_: {MetricScope.SEQUENCE_LEVEL_DETECTION: metric_block},
    )
    monkeypatch.setattr(
        experiment_results,
        "_evaluation_unit_for_dataset",
        lambda _: EvaluationUnit.SEQUENCE,
    )
    monkeypatch.setattr(
        experiment_results,
        "_build_split_policy",
        lambda **_: {"train_fraction": 0.5},
    )
    monkeypatch.setattr(
        experiment_results,
        "_build_stream_segment_policy",
        lambda _: {"mode": "none"},
    )

    metadata = experiment_results.build_metric_metadata(
        bundle=bundle,
        sequences=sequences,
        model_summary=model_summary,
    )

    assert "aggregation_policy" not in metadata
    assert "prediction_unit" not in metadata
    assert "label_unit" not in metadata
    assert metadata["evaluation_unit"] == EvaluationUnit.SEQUENCE.value
    assert (
        metadata["primary_metric_scope"] == MetricScope.SEQUENCE_LEVEL_DETECTION.value
    )


def test_build_dataset_manifest_trims_debug_only_fields(
    tmp_path: Path,
) -> None:
    """Persisted manifests should keep only the compact detector summaries.

    Args:
        tmp_path (Path): Temporary directory used to hold the synthetic raw
            dataset and result artefacts.
    """

    class DeepLogModelManifest(ModelManifest, frozen=True, kw_only=True):
        """Detector-specific manifest fields used to exercise compaction.

        Attributes:
            parameter_models (list[dict[str, str]]): Parameter models that
                should be omitted from the compact manifest.
            skipped_parameter_models (list[dict[str, str]]): Skipped parameter
                models that should be omitted from the compact manifest.
        """

        parameter_models: list[dict[str, str]] = msgspec.field(
            default_factory=list,
        )
        skipped_parameter_models: list[dict[str, str]] = msgspec.field(
            default_factory=list,
        )

    class DeepCaseModelManifest(ModelManifest, frozen=True, kw_only=True):
        """Detector-specific manifest fields used to exercise compaction.

        Attributes:
            prediction_diagnostics (dict[str, Any]): Detector-specific
                diagnostics that should be compacted before persistence.
        """

        prediction_diagnostics: dict[str, Any]

    bundle = next(
        bundle
        for bundle in load_experiment_bundles(
            Path("experiments/configs/datasets/bgl/entity_chronological.toml"),
        )
        if bundle.model.detector == "deeplog"
    )
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=tmp_path / "raw.log",
        parser=NullStructuredParser(),
        rows=[],
    )
    sink.raw_dataset_path.write_text("", encoding="utf-8")
    templated = TemplatedDataset(
        sink=sink,
        cache_paths=CachePathsConfig(
            data_root=tmp_path / "data",
            cache_root=tmp_path / "cache",
        ),
        template_parser=IdentityTemplateParser(),
        anomaly_labels=label_lookup(),
    )
    sequences = templated.sequences()
    result_paths = experiment_results.ResultPaths(
        run_fingerprint="fingerprint",
        run_root=tmp_path,
        run_dir=tmp_path,
        config_path=tmp_path / "experiment_config.json",
        dataset_manifest_path=tmp_path / "dataset_manifest.json",
        metrics_path=tmp_path / "metrics.json",
        predictions_path=tmp_path / "predictions.jsonl",
        environment_path=tmp_path / "environment.json",
        run_log_path=tmp_path / "run.log",
    )

    deeplog_context = experiment_results.ResultWriteContext(
        bundle=bundle,
        templated=templated,
        sequences=sequences,
        model_summary=ModelRunSummary(
            metrics={},
            model_manifest=DeepLogModelManifest(
                detector="deeplog",
                train_sequence_count=0,
                test_sequence_count=0,
                train_label_counts={},
                test_label_counts={},
                ignored_sequence_count=0,
            ),
            sequence_summary=SequenceSummary(
                sequence_count=0,
                train_sequence_count=0,
                test_sequence_count=0,
                train_label_counts={},
                test_label_counts={},
            ),
        ),
        result_paths=result_paths,
    )
    deepcase_context = experiment_results.ResultWriteContext(
        bundle=bundle,
        templated=templated,
        sequences=sequences,
        model_summary=ModelRunSummary(
            metrics={},
            model_manifest=DeepCaseModelManifest(
                detector="deepcase",
                train_sequence_count=0,
                test_sequence_count=0,
                train_label_counts={},
                test_label_counts={},
                ignored_sequence_count=0,
                prediction_diagnostics={
                    "event_count": 10,
                    "confident_event_count": 7,
                    "abstained_event_count": 3,
                    "confident_anomaly_event_count": 2,
                    "sequence_confident_anomaly_count": 1,
                    "sequence_confident_normal_count": 4,
                    "sequence_abstained_count": 2,
                    "abstained_anomalous_label_count": 1,
                    "abstained_normal_label_count": 2,
                    "reason_counts": {"known_benign_cluster": 8},
                    "event_decision_metrics": {
                        "event_count": 10,
                        "event_auto_decision_count": 7,
                        "event_abstained_decision_count": 3,
                        "event_auto_coverage": 0.7,
                        "event_abstain_rate": 0.3,
                        "event_tp": 2,
                        "event_fp": 1,
                        "event_tn": 3,
                        "event_fn": 1,
                        "event_precision": 0.5,
                        "event_recall": 0.4,
                        "event_f1": 0.44,
                        "event_accuracy": 0.6,
                        "event_predicted_normal_count": 5,
                        "event_predicted_anomalous_count": 2,
                        "event_true_normal_count": 6,
                        "event_true_anomalous_count": 4,
                    },
                },
            ),
            sequence_summary=SequenceSummary(
                sequence_count=0,
                train_sequence_count=0,
                test_sequence_count=0,
                train_label_counts={},
                test_label_counts={},
            ),
        ),
        result_paths=result_paths,
    )

    deeplog_manifest: Any = experiment_results.build_dataset_manifest(
        context=deeplog_context,
    )
    deepcase_manifest: Any = experiment_results.build_dataset_manifest(
        context=deepcase_context,
    )

    deeplog_model_manifest = deeplog_manifest["model_manifest"]
    assert "parameter_models" not in deeplog_model_manifest
    assert "skipped_parameter_models" not in deeplog_model_manifest
    deepcase_model_manifest = deepcase_manifest["model_manifest"]
    prediction_diagnostics = deepcase_model_manifest["prediction_diagnostics"]
    assert "reason_counts" not in prediction_diagnostics
    assert "abstained_anomalous_label_count" not in prediction_diagnostics
    assert "abstained_normal_label_count" not in prediction_diagnostics
    assert "event_decision_metrics" in prediction_diagnostics


def test_write_run_outputs_emits_parameter_ci_report_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parameter-only DeepLog results should write the concise report artefact.

    Args:
        tmp_path (Path): Temporary directory used to hold the synthetic run
            outputs.
        monkeypatch (pytest.MonkeyPatch): Patch helper used to stub the report
            builders.
    """
    bundle = next(
        bundle
        for bundle in load_experiment_bundles(
            Path("experiments/configs/datasets/bgl/entity_chronological.toml"),
        )
        if bundle.model.detector == "deeplog"
    )
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=tmp_path / "raw.log",
        parser=NullStructuredParser(),
        rows=[],
    )
    templated = TemplatedDataset(
        sink=sink,
        cache_paths=CachePathsConfig(
            data_root=tmp_path / "data",
            cache_root=tmp_path / "cache",
        ),
        template_parser=IdentityTemplateParser(),
        anomaly_labels=label_lookup(),
    )
    sequences = templated.sequences()
    model_summary = ModelRunSummary(
        metrics={
            "sequence_count": 1,
            "train_sequence_count": 0,
            "test_sequence_count": 1,
            "ignored_sequence_count": 0,
            "parameter_ci_report": {"task": "parameter_ci_approximation"},
        },
        model_manifest=ModelManifest(
            detector="deeplog",
            train_sequence_count=0,
            test_sequence_count=1,
            train_label_counts={},
            test_label_counts={1: 1},
            ignored_sequence_count=0,
        ),
        sequence_summary=SequenceSummary(
            sequence_count=1,
            train_sequence_count=0,
            test_sequence_count=1,
            train_label_counts={},
            test_label_counts={1: 1},
        ),
    )
    result_paths = experiment_results.ResultPaths(
        run_fingerprint="fingerprint",
        run_root=tmp_path,
        run_dir=tmp_path,
        config_path=tmp_path / "experiment_config.json",
        dataset_manifest_path=tmp_path / "dataset_manifest.json",
        metrics_path=tmp_path / "metrics.json",
        predictions_path=tmp_path / "predictions.jsonl",
        environment_path=tmp_path / "environment.json",
        run_log_path=tmp_path / "run.log",
    )
    context = experiment_results.ResultWriteContext(
        bundle=bundle,
        templated=templated,
        sequences=sequences,
        model_summary=model_summary,
        result_paths=result_paths,
    )
    monkeypatch.setattr(
        experiment_results,
        "build_dataset_manifest",
        lambda **_: {"dataset": "manifest"},
    )
    monkeypatch.setattr(
        experiment_results,
        "build_run_metrics_report",
        lambda **kwargs: {
            "sequence_count": kwargs["model_summary"].metrics["sequence_count"],
        },
    )
    monkeypatch.setattr(
        experiment_results,
        "build_environment_metadata",
        lambda **_: {"environment": "metadata"},
    )

    experiment_results.write_run_outputs(context=context)

    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "parameter_ci_report" not in metrics
    assert "parameter_ci_trace" not in metrics
    assert json.loads(
        (tmp_path / "figure9_parameter_ci.json").read_text(
            encoding="utf-8",
        ),
    ) == {"task": "parameter_ci_approximation"}
    assert not (tmp_path / "figure9_parameter_ci_debug.json").exists()


def test_write_run_outputs_emits_parameter_ci_debug_artifact_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Debug reporting should persist the verbose parameter trace separately.

    Args:
        tmp_path (Path): Temporary directory used to hold the synthetic run
            outputs.
        monkeypatch (pytest.MonkeyPatch): Patch helper used to stub the report
            builders.
    """
    bundle = next(
        bundle
        for bundle in load_experiment_bundles(
            Path("experiments/configs/datasets/bgl/entity_chronological.toml"),
        )
        if bundle.model.detector == "deeplog"
    )
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=tmp_path / "raw.log",
        parser=NullStructuredParser(),
        rows=[],
    )
    templated = TemplatedDataset(
        sink=sink,
        cache_paths=CachePathsConfig(
            data_root=tmp_path / "data",
            cache_root=tmp_path / "cache",
        ),
        template_parser=IdentityTemplateParser(),
        anomaly_labels=label_lookup(),
    )
    sequences = templated.sequences()
    model_summary = ModelRunSummary(
        metrics={
            "sequence_count": 1,
            "train_sequence_count": 0,
            "test_sequence_count": 1,
            "ignored_sequence_count": 0,
            "parameter_ci_report": {"task": "parameter_ci_approximation"},
            "parameter_ci_trace": {
                "task": "parameter_ci_approximation",
                "series_count": 1,
            },
        },
        model_manifest=ModelManifest(
            detector="deeplog",
            train_sequence_count=0,
            test_sequence_count=1,
            train_label_counts={},
            test_label_counts={1: 1},
            ignored_sequence_count=0,
        ),
        sequence_summary=SequenceSummary(
            sequence_count=1,
            train_sequence_count=0,
            test_sequence_count=1,
            train_label_counts={},
            test_label_counts={1: 1},
        ),
    )
    result_paths = experiment_results.ResultPaths(
        run_fingerprint="fingerprint",
        run_root=tmp_path,
        run_dir=tmp_path,
        config_path=tmp_path / "experiment_config.json",
        dataset_manifest_path=tmp_path / "dataset_manifest.json",
        metrics_path=tmp_path / "metrics.json",
        predictions_path=tmp_path / "predictions.jsonl",
        environment_path=tmp_path / "environment.json",
        run_log_path=tmp_path / "run.log",
    )
    context = experiment_results.ResultWriteContext(
        bundle=bundle,
        templated=templated,
        sequences=sequences,
        model_summary=model_summary,
        result_paths=result_paths,
        debug_reporting=True,
    )
    monkeypatch.setattr(
        experiment_results,
        "build_dataset_manifest",
        lambda **_: {"dataset": "manifest"},
    )
    monkeypatch.setattr(
        experiment_results,
        "build_run_metrics_report",
        lambda **kwargs: {
            "sequence_count": kwargs["model_summary"].metrics["sequence_count"],
        },
    )
    monkeypatch.setattr(
        experiment_results,
        "build_environment_metadata",
        lambda **_: {"environment": "metadata"},
    )

    experiment_results.write_run_outputs(context=context)

    assert json.loads(
        (tmp_path / "figure9_parameter_ci_debug.json").read_text(
            encoding="utf-8",
        ),
    ) == {
        "task": "parameter_ci_approximation",
        "series_count": 1,
    }
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert "parameter_ci_report" not in metrics
    assert "parameter_ci_trace" not in metrics


def test_build_environment_metadata_records_deepcase_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run provenance should record the pinned DeepCASE dependency version.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patch helper used to substitute
            package version lookups.
    """
    captured_dist_names: list[str] = []

    def _fake_package_version(dist_name: str) -> str:
        captured_dist_names.append(dist_name)
        if dist_name == "anomalog":
            return "0.3.0"
        if dist_name == "deepcase":
            return "1.0.3"
        msg = f"Unexpected package request: {dist_name}"
        raise AssertionError(msg)

    monkeypatch.setattr(experiment_results, "_package_version", _fake_package_version)

    bundle = next(
        bundle
        for bundle in load_experiment_bundles(
            Path("experiments/configs/datasets/bgl/entity_chronological.toml"),
        )
        if bundle.model.detector == "deepcase"
    )
    metadata = experiment_results.build_environment_metadata(
        bundle=bundle,
        result_paths=experiment_results.prepare_result_paths(bundle),
    )

    assert metadata["packages"] == {
        "anomalog": "0.3.0",
        "deepcase": "1.0.3",
    }
    assert captured_dist_names == ["anomalog", "deepcase"]
