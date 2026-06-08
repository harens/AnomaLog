"""Result-directory management and manifest utilities."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any, Protocol

import msgspec

from anomalog.parsers.structured.contracts import is_anomalous_label
from anomalog.parsers.structured.parsers import ThunderbirdParser
from experiments.config import serialise_config
from experiments.config_types import (
    ChronologicalStreamSequenceConfig,
    DatasetVariantConfig,
    EntitySequenceConfig,
    FixedSequenceConfig,
    TimeSequenceConfig,
)
from experiments.datasets import dataset_source_summary
from experiments.models.metric_reporting import (
    BinaryMetricBlockRequest,
    DiagnosticMetricBlockRequest,
    MetricBlock,
    build_binary_metric_block,
    build_diagnostic_metric_block,
    build_not_applicable_metric_block,
    select_primary_metric_scope,
)
from experiments.models.metric_schema import EvaluationUnit, MetricScope, MetricStatus

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from anomalog.parsers.template import TemplatedDataset
    from anomalog.sequences import (
        RawEntrySplitSummary,
        SequenceBuilder,
        SequenceSplitSummary,
    )
    from experiments.config import ExperimentBundle
    from experiments.models import ModelRunSummary, SequenceSummary


class _DatasetStatisticsBundle(Protocol):
    dataset: DatasetVariantConfig


class _DatasetStatisticsContext(Protocol):
    templated: TemplatedDataset
    model_summary: ModelRunSummary


class _SupportsResultSummaryCache(Protocol):
    split_summary: SequenceSplitSummary | None
    raw_entry_split_summary: RawEntrySplitSummary | None
    metric_report: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class ResultPaths:
    """Concrete artifact paths inside a single run directory.

    The run fingerprint is derived from the fully resolved config so repeated
    executions of the same experiment land under one deterministic fingerprint
    root. Keeping all artifact paths together avoids ad-hoc filename drift
    across result writers.

    Attributes:
        run_fingerprint (str): Stable fingerprint for the resolved run config.
        run_root (Path): Deterministic fingerprint directory for the concrete
            run family.
        run_dir (Path): Root directory containing all artifacts for the run.
        config_path (Path): Serialised normalised concrete experiment config
            path.
        dataset_manifest_path (Path): Dataset provenance manifest path.
        metrics_path (Path): Detector metrics output path.
        predictions_path (Path): Prediction records output path.
        environment_path (Path): Environment/provenance metadata path.
        run_log_path (Path): Captured run log path.
    """

    run_fingerprint: str
    run_root: Path
    run_dir: Path
    config_path: Path
    dataset_manifest_path: Path
    metrics_path: Path
    predictions_path: Path
    environment_path: Path
    run_log_path: Path

    @classmethod
    def for_bundle(
        cls,
        bundle: ExperimentBundle,
        *,
        run_attempt: int | None = None,
    ) -> ResultPaths:
        """Create deterministic result paths for the experiment bundle.

        Args:
            bundle (ExperimentBundle): Resolved experiment bundle.
            run_attempt (int | None): Optional 1-based attempt number written
                beneath the fingerprint root. When omitted, the concrete run
                writes directly to the fingerprint directory.

        Returns:
            ResultPaths: Deterministic run artifact paths for the bundle.
        """
        combined_config = bundle.normalized_config()
        run_fingerprint = stable_fingerprint(combined_config)
        results_root = (
            bundle.repo_root / bundle.sweep.results_root
            if not bundle.sweep.results_root.is_absolute()
            else bundle.sweep.results_root
        )
        run_root = results_root / bundle.concrete_name / run_fingerprint[:12]
        run_dir = (
            run_root if run_attempt is None else run_root / f"attempt-{run_attempt:04d}"
        )
        return cls(
            run_fingerprint=run_fingerprint,
            run_root=run_root,
            run_dir=run_dir,
            config_path=run_dir / "experiment_config.json",
            dataset_manifest_path=run_dir / "dataset_manifest.json",
            metrics_path=run_dir / "metrics.json",
            predictions_path=run_dir / "predictions.jsonl",
            environment_path=run_dir / "environment.json",
            run_log_path=run_dir / "run.log",
        )


@dataclass(frozen=True, slots=True)
class ResultWriteContext:
    """Inputs needed to persist one concrete experiment result bundle.

    Attributes:
        bundle (ExperimentBundle): Resolved concrete experiment bundle.
        templated (TemplatedDataset): Materialised templated dataset view.
        sequences (SequenceBuilder): Sequence builder used to replay the run.
        model_summary (ModelRunSummary): Model-side summary for the completed
            run.
        result_paths (ResultPaths): Deterministic output paths for the bundle.
        debug_reporting (bool): Whether verbose diagnostics should be written.
    """

    bundle: ExperimentBundle
    templated: TemplatedDataset
    sequences: SequenceBuilder
    model_summary: ModelRunSummary
    result_paths: ResultPaths
    debug_reporting: bool = False


@dataclass(frozen=True, slots=True)
class _ResultSummaryCache:
    """Optional precomputed summaries reused during result persistence.

    Attributes:
        split_summary (SequenceSplitSummary | None): Cached sequence split
            summary reused when writing the dataset manifest.
        raw_entry_split_summary (RawEntrySplitSummary | None): Cached raw-entry
            split summary reused when writing the dataset manifest.
        metric_report (dict[str, object] | None): Cached metric report reused
            when writing the metrics artefact.
    """

    split_summary: SequenceSplitSummary | None = None
    raw_entry_split_summary: RawEntrySplitSummary | None = None
    metric_report: dict[str, object] | None = None


def prepare_result_paths(
    bundle: ExperimentBundle,
    *,
    run_attempt: int | None = None,
) -> ResultPaths:
    """Create deterministic result paths for the experiment bundle.

    Args:
        bundle (ExperimentBundle): Resolved experiment bundle.
        run_attempt (int | None): Optional 1-based attempt number written
            beneath the fingerprint root. When omitted, the concrete run
            writes directly to the fingerprint directory.

    Returns:
        ResultPaths: Deterministic run artifact paths for the bundle.
    """
    return ResultPaths.for_bundle(bundle, run_attempt=run_attempt)


def write_run_outputs(
    *,
    context: ResultWriteContext,
    split_summary: SequenceSplitSummary | None = None,
    raw_entry_split_summary: RawEntrySplitSummary | None = None,
    metric_report: dict[str, object] | None = None,
) -> None:
    """Persist the full experiment result bundle.

    Args:
        context (ResultWriteContext): Resolved run inputs and persistence
            targets for the run.
        split_summary (SequenceSplitSummary | None): Optional precomputed
            split-summary metadata to reuse when building the manifest.
        raw_entry_split_summary (RawEntrySplitSummary | None): Optional
            precomputed raw-entry split summary to reuse when building the
            manifest.
        metric_report (dict[str, object] | None): Optional precomputed metric
            report to reuse when writing the metrics artefact.
    """
    bundle = context.bundle
    sequences = context.sequences
    model_summary = context.model_summary
    result_paths = context.result_paths
    debug_reporting = context.debug_reporting
    cached_summaries = _ResultSummaryCache(
        split_summary=split_summary,
        raw_entry_split_summary=raw_entry_split_summary,
        metric_report=metric_report,
    )
    _write_json(result_paths.config_path, bundle.normalized_config())
    _write_json(
        result_paths.dataset_manifest_path,
        build_dataset_manifest(
            context=context,
            split_summary=split_summary,
            raw_entry_split_summary=raw_entry_split_summary,
        ),
    )
    metric_report = build_run_metrics_report(
        bundle=bundle,
        sequences=sequences,
        model_summary=model_summary,
        debug_reporting=debug_reporting,
        cached_summaries=cached_summaries,
    )
    _write_json(result_paths.metrics_path, metric_report)
    parameter_ci_report = _parameter_ci_report(model_summary.metrics)
    if parameter_ci_report is not None:
        _write_json(
            result_paths.run_dir / "figure9_parameter_ci.json",
            parameter_ci_report,
        )
    if debug_reporting:
        parameter_ci_trace = _parameter_ci_trace(model_summary.metrics)
        if parameter_ci_trace is not None:
            _write_json(
                result_paths.run_dir / "figure9_parameter_ci_debug.json",
                parameter_ci_trace,
            )
    _write_json(
        result_paths.environment_path,
        build_environment_metadata(
            bundle=bundle,
            result_paths=result_paths,
        ),
    )


def stable_fingerprint(payload: object) -> str:
    """Return a deterministic fingerprint for a JSON-serialisable payload.

    Args:
        payload (object): JSON-serialisable payload to fingerprint.

    Returns:
        str: SHA-256 fingerprint for the serialised payload.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8",
    )
    return hashlib.sha256(encoded).hexdigest()


def build_dataset_manifest(
    *,
    context: ResultWriteContext,
    split_summary: SequenceSplitSummary | None = None,
    raw_entry_split_summary: RawEntrySplitSummary | None = None,
) -> dict[str, object]:
    """Build a provenance manifest for the preprocessed dataset and sequences.

    Args:
        context (ResultWriteContext): Resolved run inputs and persistence
            targets for the run.
        split_summary (SequenceSplitSummary | None): Optional precomputed
            split-summary metadata to reuse instead of replaying the builder.
        raw_entry_split_summary (RawEntrySplitSummary | None): Optional
            precomputed raw-entry split summary to reuse instead of replaying
            the builder.

    Returns:
        dict[str, object]: Dataset and sequence provenance manifest.
    """
    bundle = context.bundle
    model_summary = context.model_summary
    debug_reporting = context.debug_reporting
    sequence_summary = model_summary.sequence_summary
    raw_logs_path = context.templated.sink.raw_dataset_path.resolve()
    timestamp_min, timestamp_max = context.templated.sink.timestamp_bounds()
    if split_summary is None:
        split_summary = build_sequence_split_summary(
            context.sequences,
            sequence_summary=sequence_summary,
        )
    if raw_entry_split_summary is None:
        raw_entry_split_summary = context.sequences.build_raw_entry_split_summary()
    manifest = {
        "run_fingerprint": context.result_paths.run_fingerprint,
        "dataset_fingerprint": stable_fingerprint(serialise_config(bundle.dataset)),
        "dataset_variant": bundle.dataset.name,
        "dataset_name": bundle.dataset.dataset_name,
        **(
            {"experiment": _experiment_metadata(bundle)}
            if bundle.experiment_name is not None
            else {}
        ),
        "source": dataset_source_summary(bundle.dataset, repo_root=bundle.repo_root),
        "structured_parser": _structured_parser_name(bundle),
        "template_parser": bundle.dataset.template_parser,
        "raw_logs": {
            "path": raw_logs_path.as_posix(),
            "sha256": sha256_for_file(raw_logs_path),
        },
        "structured_rows": context.templated.sink.count_rows(),
        **(
            {
                "dataset_statistics": _build_dataset_statistics(
                    bundle=bundle,
                    context=context,
                ),
            }
            if bundle.dataset.preset == "ait_ads"
            or bundle.dataset.preset in {"thunderbird", "thunderbird_smoke"}
            else {}
        ),
        "timestamp_bounds": {
            "min_unix_ms": timestamp_min,
            "max_unix_ms": timestamp_max,
        },
        "sequence_config": serialise_config(bundle.dataset.sequence),
        "sequence_split_summary": split_summary.as_dict(),
        **(
            {"raw_entry_split_summary": raw_entry_split_summary.as_dict()}
            if raw_entry_split_summary is not None
            else {}
        ),
        "sequence_count": sequence_summary.sequence_count,
        "sequence_split_counts": {
            "train": sequence_summary.train_sequence_count,
            "test": sequence_summary.test_sequence_count,
            "ignored": sequence_summary.ignored_sequence_count,
        },
        "label_counts": {
            "train": sequence_summary.train_label_counts,
            "test": sequence_summary.test_label_counts,
        },
        **build_metric_metadata(
            bundle=bundle,
            sequences=context.sequences,
            model_summary=model_summary,
            split_summary=split_summary,
            raw_entry_split_summary=raw_entry_split_summary,
        ),
    }
    manifest["model_manifest"] = _compact_model_manifest(
        msgspec.to_builtins(model_summary.model_manifest),
        debug_reporting=debug_reporting,
    )
    return _compact_dataset_manifest(manifest, debug_reporting=debug_reporting)


def _build_dataset_statistics(
    *,
    bundle: _DatasetStatisticsBundle,
    context: _DatasetStatisticsContext,
) -> dict[str, object] | None:
    """Return optional dataset-specific manifest statistics.

    Args:
        bundle (_DatasetStatisticsBundle): Resolved bundle used to identify
            the dataset family.
        context (_DatasetStatisticsContext): Resolved result context backing
            the manifest.

    Returns:
        dict[str, object] | None: AIT-ADS or Thunderbird summary fields, or
            `None` when the dataset family does not expose those statistics.
    """
    if bundle.dataset.preset != "ait_ads":
        if bundle.dataset.preset not in {"thunderbird", "thunderbird_smoke"}:
            return None
        return _build_thunderbird_dataset_statistics(context=context)

    total_alerts = 0
    anomalous_alert_count = 0
    missing_timestamp_alert_count = 0
    total_alerts_per_scenario: dict[str, int] = {}
    total_alerts_per_ids_source: dict[str, int] = {}

    raw_logs_path = context.templated.sink.raw_dataset_path.resolve()
    for line in raw_logs_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            continue
        total_alerts += 1
        scenario = str(record.get("scenario"))
        ids_source = str(record.get("ids_source"))
        total_alerts_per_scenario[scenario] = (
            total_alerts_per_scenario.get(scenario, 0) + 1
        )
        total_alerts_per_ids_source[ids_source] = (
            total_alerts_per_ids_source.get(ids_source, 0) + 1
        )
        if record.get("timestamp_unix_ms") is None:
            missing_timestamp_alert_count += 1
        anomalous = record.get("anomalous")
        if anomalous not in {None, 0}:
            anomalous_alert_count += 1

    return {
        "total_alerts_parsed": total_alerts,
        "total_alerts_per_scenario": dict(sorted(total_alerts_per_scenario.items())),
        "total_alerts_per_ids_source": dict(
            sorted(total_alerts_per_ids_source.items()),
        ),
        "anomalous_alert_count": anomalous_alert_count,
        "anomalous_alert_fraction": (
            0.0 if total_alerts == 0 else anomalous_alert_count / total_alerts
        ),
        "missing_timestamp_alert_count": missing_timestamp_alert_count,
    }


def _build_thunderbird_dataset_statistics(
    *,
    context: _DatasetStatisticsContext,
) -> dict[str, object]:
    """Return Thunderbird-specific manifest statistics.

    Args:
        context (_DatasetStatisticsContext): Resolved result context for the
            build.

    Returns:
        dict[str, object]: Thunderbird-specific manifest statistics.
    """
    total_lines = 0
    emitted_events = 0
    normal_events = 0
    anomalous_events = 0
    missing_timestamp_events = 0
    skipped_reasons: Counter[str] = Counter()

    raw_logs_path = context.templated.sink.raw_dataset_path.resolve()
    for raw_line in raw_logs_path.open(encoding="utf-8", errors="replace"):
        total_lines += 1
        parsed, reason = ThunderbirdParser.analyse_line(raw_line)
        if parsed is None:
            skipped_reasons[reason or "unknown"] += 1
            continue
        emitted_events += 1
        if parsed.timestamp_unix_ms is None:
            missing_timestamp_events += 1
        if is_anomalous_label(parsed.anomalous):
            anomalous_events += 1
        else:
            normal_events += 1

    template_vocabulary: set[str] = set()
    template_vocabulary.update(
        context.templated.template_parser.inference(
            row.untemplated_message_text,
        )[0]
        for row in context.templated.sink.iter_structured_lines(
            columns=["untemplated_message_text"],
        )()
    )

    return {
        "total_lines_parsed": total_lines,
        "total_events_emitted": emitted_events,
        "normal_event_count": normal_events,
        "anomalous_event_count": anomalous_events,
        "anomalous_event_fraction": (
            0.0 if emitted_events == 0 else anomalous_events / emitted_events
        ),
        "missing_timestamp_event_count": missing_timestamp_events,
        "skipped_line_count": total_lines - emitted_events,
        "skipped_line_reasons": dict(sorted(skipped_reasons.items())),
        "sequence_window_count": context.model_summary.sequence_summary.sequence_count,
        "train_sequence_count": (
            context.model_summary.sequence_summary.train_sequence_count
        ),
        "test_sequence_count": (
            context.model_summary.sequence_summary.test_sequence_count
        ),
        "ignored_sequence_count": (
            context.model_summary.sequence_summary.ignored_sequence_count
        ),
        "template_vocabulary_size": len(template_vocabulary),
    }


def build_sequence_split_summary(
    sequences: SequenceBuilder,
    *,
    sequence_summary: SequenceSummary,
) -> SequenceSplitSummary:
    """Describe requested versus effective split semantics for one run.

    Args:
        sequences (SequenceBuilder): Sequence builder whose split semantics are
            being summarised.
        sequence_summary (SequenceSummary): Aggregate split and label counts.

    Returns:
        SequenceSplitSummary: Requested and effective split metrics.
    """
    return sequences.build_split_summary(
        sequence_summary=sequence_summary,
    )


def build_metric_metadata(
    *,
    bundle: ExperimentBundle,
    sequences: SequenceBuilder,
    model_summary: ModelRunSummary,
    split_summary: SequenceSplitSummary | None = None,
    raw_entry_split_summary: RawEntrySplitSummary | None = None,
) -> dict[str, object]:
    """Build the task metadata that accompanies persisted metric blocks.

    Args:
        bundle (ExperimentBundle): Experiment bundle being evaluated.
        sequences (SequenceBuilder): Sequence builder used for the run.
        model_summary (ModelRunSummary): Model-side summary for the run.
        split_summary (SequenceSplitSummary | None): Optional precomputed
            split-summary metadata to reuse when building the split policy.
        raw_entry_split_summary (RawEntrySplitSummary | None): Optional
            precomputed raw-entry split summary to reuse when building the
            split policy.

    Returns:
        dict[str, object]: Shared task metadata for the persisted dataset
            manifest and metrics report.
    """
    metric_blocks = _build_metric_blocks(
        bundle=bundle,
        model_summary=model_summary,
    )
    evaluation_unit = _evaluation_unit_for_dataset(bundle.dataset)
    primary_scope = select_primary_metric_scope(
        metric_blocks,
        requested_primary_scope=bundle.model.primary_metric_scope,
        evaluation_unit=evaluation_unit,
    )
    split_policy = _build_split_policy(
        bundle=bundle,
        sequences=sequences,
        model_summary=model_summary,
        split_summary=split_summary,
        raw_entry_split_summary=raw_entry_split_summary,
    )
    stream_segment_policy = _build_stream_segment_policy(bundle.dataset)
    return {
        "evaluation_unit": evaluation_unit.value,
        "split_policy": split_policy,
        "stream_segment_policy": stream_segment_policy,
        "primary_metric_scope": (
            None if primary_scope is None else primary_scope.value
        ),
    }


def build_run_metrics_report(
    *,
    bundle: ExperimentBundle,
    sequences: SequenceBuilder,
    model_summary: ModelRunSummary,
    debug_reporting: bool = False,
    cached_summaries: _SupportsResultSummaryCache | None = None,
) -> dict[str, object]:
    """Build the final task-aware metric report written to ``metrics.json``.

    Args:
        bundle (ExperimentBundle): Experiment bundle being evaluated.
        sequences (SequenceBuilder): Sequence builder used for the run.
        model_summary (ModelRunSummary): Model-side summary for the run.
        debug_reporting (bool): Whether to preserve the verbose diagnostic
            payloads in the written metrics report.
        cached_summaries (_SupportsResultSummaryCache | None): Optional precomputed
            summaries to reuse when building or persisting the report.

    Returns:
        dict[str, object]: Serialised task-aware metric report with metadata
            and canonical metric blocks.
    """
    if cached_summaries is not None and cached_summaries.metric_report is not None:
        return _compact_run_metrics_report(
            cached_summaries.metric_report,
            debug_reporting=debug_reporting,
        )
    split_summary = None if cached_summaries is None else cached_summaries.split_summary
    raw_entry_split_summary = (
        None if cached_summaries is None else cached_summaries.raw_entry_split_summary
    )
    metric_blocks = _build_metric_blocks(
        bundle=bundle,
        model_summary=model_summary,
    )
    metadata = build_metric_metadata(
        bundle=bundle,
        sequences=sequences,
        model_summary=model_summary,
        split_summary=split_summary,
        raw_entry_split_summary=raw_entry_split_summary,
    )
    run_metrics = model_summary.metrics
    report = {
        **metadata,
        "sequence_count": run_metrics.get("sequence_count"),
        "train_sequence_count": run_metrics.get("train_sequence_count"),
        "test_sequence_count": run_metrics.get("test_sequence_count"),
        "ignored_sequence_count": run_metrics.get("ignored_sequence_count"),
        "train_label_counts": run_metrics.get("train_label_counts"),
        "test_label_counts": run_metrics.get("test_label_counts"),
        "ignored_label_counts": run_metrics.get("ignored_label_counts"),
        "mean_test_score": run_metrics.get("mean_test_score"),
        "metric_blocks": {
            scope.value: msgspec.to_builtins(block)
            for scope, block in metric_blocks.items()
        },
    }
    return _compact_run_metrics_report(report, debug_reporting=debug_reporting)


def _parameter_ci_report(metrics: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the detector-owned parameter CI summary, if present.

    Args:
        metrics (Mapping[str, Any]): Canonical run metrics emitted by the
            detector.

    Returns:
        dict[str, Any] | None: Parameter CI summary payload when present,
            otherwise `None`.
    """
    report = metrics.get("parameter_ci_report")
    if isinstance(report, dict):
        return report
    return None


def _parameter_ci_trace(metrics: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the detector-owned parameter CI debug trace, if present.

    Args:
        metrics (Mapping[str, Any]): Canonical run metrics emitted by the
            detector.

    Returns:
        dict[str, Any] | None: Parameter CI debug payload when present,
            otherwise `None`.
    """
    trace = metrics.get("parameter_ci_trace")
    if isinstance(trace, dict):
        return trace
    return None


def _build_metric_blocks(
    *,
    bundle: ExperimentBundle,
    model_summary: ModelRunSummary,
) -> dict[MetricScope, MetricBlock]:
    metrics = _as_metric_mapping(model_summary.metrics)
    sequence_summary = model_summary.sequence_summary
    metric_blocks: dict[MetricScope, MetricBlock] = {}
    evaluation_unit = _evaluation_unit_for_dataset(bundle.dataset)
    primary_scope = bundle.model.primary_metric_scope

    if _should_emit_sequence_level_detection(bundle.dataset):
        sequence_block = _build_sequence_level_detection_block(
            metrics=metrics,
            sequence_summary=sequence_summary,
            evaluation_unit=evaluation_unit,
            primary_scope=primary_scope,
            allow_single_class_reporting=bundle.model.allow_single_class_reporting,
        )
        if sequence_block is not None:
            metric_blocks[MetricScope.SEQUENCE_LEVEL_DETECTION] = sequence_block

    event_block = _build_event_level_detection_block(
        metrics=metrics,
        primary_scope=primary_scope,
        allow_single_class_reporting=bundle.model.allow_single_class_reporting,
    )
    if event_block is not None:
        metric_blocks[MetricScope.EVENT_LEVEL_DETECTION] = event_block

    next_event_block = _build_next_event_prediction_block(metrics=metrics)
    if next_event_block is not None:
        metric_blocks[MetricScope.NEXT_EVENT_PREDICTION] = next_event_block

    manual_block = _build_manual_workload_block(metrics=metrics)
    if manual_block is not None:
        metric_blocks[MetricScope.MANUAL_WORKLOAD_REDUCTION] = manual_block

    semi_automatic_block = _build_semi_automatic_workload_block(metrics=metrics)
    if semi_automatic_block is not None:
        metric_blocks[MetricScope.SEMI_AUTOMATIC_WORKLOAD_REDUCTION] = (
            semi_automatic_block
        )

    if MetricScope.NEXT_EVENT_PREDICTION in metric_blocks:
        if MetricScope.EVENT_LEVEL_DETECTION not in metric_blocks:
            metric_blocks[MetricScope.EVENT_LEVEL_DETECTION] = (
                build_not_applicable_metric_block(
                    prediction_unit=EvaluationUnit.EVENT,
                    label_unit=EvaluationUnit.EVENT,
                    diagnostics={"reason": "no_event_level_labels"},
                )
            )
        if MetricScope.SEQUENCE_LEVEL_DETECTION not in metric_blocks:
            metric_blocks[MetricScope.SEQUENCE_LEVEL_DETECTION] = (
                build_not_applicable_metric_block(
                    prediction_unit=EvaluationUnit.SEQUENCE,
                    label_unit=EvaluationUnit.SEQUENCE,
                    diagnostics={"reason": "no_sequence_level_labels"},
                )
            )

    return metric_blocks


def _should_emit_sequence_level_detection(dataset: DatasetVariantConfig) -> bool:
    """Return whether the run should surface sequence-level headline metrics.

    Args:
        dataset (DatasetVariantConfig): Dataset configuration for the current
            run.

    AIT-ADS is evaluated at the alert level in the paper, so we keep the
    sequence label internally for container semantics but suppress the
    sequence-level headline block in the report.

    Returns:
        bool: `True` when sequence-level headline metrics should be emitted.
    """
    return dataset.preset != "ait_ads"


def _build_sequence_level_detection_block(
    *,
    metrics: dict[str, object],
    sequence_summary: SequenceSummary,
    evaluation_unit: EvaluationUnit,
    primary_scope: MetricScope | None,
    allow_single_class_reporting: bool,
) -> MetricBlock | None:
    sequence_inputs = _sequence_detection_inputs(metrics)
    if sequence_inputs is None:
        return None
    (
        tp_value,
        tn_value,
        fp_value,
        fn_value,
        counted_predictions_value,
        abstained_prediction_count_value,
        normal_count,
        anomalous_count,
    ) = sequence_inputs
    sequence_block = build_binary_metric_block(
        request=BinaryMetricBlockRequest(
            prediction_unit=EvaluationUnit.SEQUENCE,
            label_unit=EvaluationUnit.SEQUENCE,
            tp=tp_value,
            fp=fp_value,
            tn=tn_value,
            fn=fn_value,
            normal_count=normal_count,
            anomalous_count=anomalous_count,
            evaluation_unit_count=sequence_summary.test_sequence_count,
            counted_predictions=counted_predictions_value,
            abstained_prediction_count=abstained_prediction_count_value,
            ignored_prediction_count=0,
            allow_single_class_reporting=allow_single_class_reporting,
            diagnostic_only=(
                primary_scope is not MetricScope.SEQUENCE_LEVEL_DETECTION
                or evaluation_unit
                in {
                    EvaluationUnit.CHRONOLOGICAL_EVENT_STREAM,
                    EvaluationUnit.CONTINUOUS_EVENT_STREAM,
                    EvaluationUnit.STREAM,
                }
            ),
            diagnostics={
                "class_counts": {
                    "normal": normal_count,
                    "anomalous": anomalous_count,
                },
                "auto_coverage": (
                    counted_predictions_value / sequence_summary.test_sequence_count
                    if sequence_summary.test_sequence_count
                    else 0.0
                ),
                "abstain_rate": (
                    abstained_prediction_count_value
                    / sequence_summary.test_sequence_count
                    if sequence_summary.test_sequence_count
                    else 0.0
                ),
            },
        ),
    )
    if sequence_block.status is MetricStatus.NOT_APPLICABLE:
        return None
    return sequence_block


def _sequence_detection_inputs(
    metrics: dict[str, object],
) -> tuple[int, int, int, int, int, int, int, int] | None:
    tp = _metric_count(metrics, "tp")
    tn = _metric_count(metrics, "tn")
    fp = _metric_count(metrics, "fp")
    fn = _metric_count(metrics, "fn")
    counted_predictions = _metric_count(metrics, "counted_predictions")
    abstained_prediction_count = _metric_count(metrics, "abstained_prediction_count")
    test_label_counts = _int_count_map(metrics.get("test_label_counts"))
    if any(
        value is None
        for value in (
            tp,
            tn,
            fp,
            fn,
            counted_predictions,
            abstained_prediction_count,
            test_label_counts,
        )
    ):
        return None
    if test_label_counts is None:
        return None
    anomalous_count = sum(
        count for label, count in test_label_counts.items() if is_anomalous_label(label)
    )
    normal_count = test_label_counts.get(0, 0)
    return (
        _require_int(tp),
        _require_int(tn),
        _require_int(fp),
        _require_int(fn),
        _require_int(counted_predictions),
        _require_int(abstained_prediction_count),
        normal_count,
        anomalous_count,
    )


def _build_event_level_detection_block(
    *,
    metrics: dict[str, object],
    primary_scope: MetricScope | None,
    allow_single_class_reporting: bool,
) -> MetricBlock | None:
    event_detection = _event_detection_inputs(metrics)
    if event_detection is None:
        return None
    (
        events_seen,
        events_eligible,
        abstained_prediction_count,
        tp,
        tn,
        fp,
        fn,
        true_normal,
        true_anomalous,
        diagnostics,
        legacy_stream,
    ) = event_detection
    event_block = build_binary_metric_block(
        request=BinaryMetricBlockRequest(
            prediction_unit=EvaluationUnit.EVENT,
            label_unit=EvaluationUnit.EVENT,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            normal_count=true_normal,
            anomalous_count=true_anomalous,
            evaluation_unit_count=events_seen,
            counted_predictions=events_eligible,
            abstained_prediction_count=abstained_prediction_count,
            ignored_prediction_count=(
                events_seen - events_eligible if legacy_stream else 0
            ),
            allow_single_class_reporting=allow_single_class_reporting,
            diagnostic_only=primary_scope is not MetricScope.EVENT_LEVEL_DETECTION,
            diagnostics=diagnostics,
        ),
    )
    if event_block.status is MetricStatus.NOT_APPLICABLE:
        return None
    return event_block


def _event_detection_inputs(
    metrics: dict[str, object],
) -> tuple[int, int, int, int, int, int, int, int, int, dict[str, Any], bool] | None:
    event_level_detection = metrics.get("event_level_detection")
    if isinstance(event_level_detection, dict):
        return _event_detection_inputs_from_stream(event_level_detection)

    prediction_diagnostics = metrics.get("prediction_diagnostics")
    if not isinstance(prediction_diagnostics, dict):
        return None
    return _event_detection_inputs_from_prediction_diagnostics(prediction_diagnostics)


def _event_detection_inputs_from_stream(
    event_level_detection: Any,  # noqa: ANN401
) -> tuple[int, int, int, int, int, int, int, int, int, dict[str, Any], bool] | None:
    event_level_detection_mapping = _as_metric_mapping(event_level_detection)
    events_seen = _int_value(event_level_detection_mapping, "events_seen")
    events_eligible = _int_value(event_level_detection_mapping, "events_eligible")
    tp = _int_value(event_level_detection_mapping, "tp")
    tn = _int_value(event_level_detection_mapping, "tn")
    fp = _int_value(event_level_detection_mapping, "fp")
    fn = _int_value(event_level_detection_mapping, "fn")
    true_normal = _int_value(
        event_level_detection_mapping,
        "normal_event_count",
        default=0,
    )
    true_anomalous = _int_value(
        event_level_detection_mapping,
        "anomalous_event_count",
        default=0,
    )
    diagnostics = {
        "events_seen": events_seen,
        "events_eligible": events_eligible,
        "source": "event_level_detection",
    }
    return (
        events_seen,
        events_eligible,
        0,
        tp,
        tn,
        fp,
        fn,
        true_normal,
        true_anomalous,
        diagnostics,
        True,
    )


def _event_detection_inputs_from_prediction_diagnostics(
    prediction_diagnostics: Any,  # noqa: ANN401
) -> tuple[int, int, int, int, int, int, int, int, int, dict[str, Any], bool] | None:
    prediction_diagnostics_mapping = _as_metric_mapping(prediction_diagnostics)
    event_decision_metrics = _mapping_value(
        prediction_diagnostics_mapping,
        "event_decision_metrics",
    )
    if not isinstance(event_decision_metrics, dict):
        return None
    event_decision_metrics_mapping = _as_metric_mapping(event_decision_metrics)
    events_seen = _int_value(event_decision_metrics_mapping, "event_count")
    events_eligible = _int_value(
        event_decision_metrics_mapping,
        "event_auto_decision_count",
    )
    abstained_prediction_count = _int_value(
        event_decision_metrics_mapping,
        "event_abstained_decision_count",
    )
    tp = _int_value(event_decision_metrics_mapping, "event_tp")
    tn = _int_value(event_decision_metrics_mapping, "event_tn")
    fp = _int_value(event_decision_metrics_mapping, "event_fp")
    fn = _int_value(event_decision_metrics_mapping, "event_fn")
    true_normal = _int_value(event_decision_metrics_mapping, "event_true_normal_count")
    true_anomalous = _int_value(
        event_decision_metrics_mapping,
        "event_true_anomalous_count",
    )
    diagnostics = {
        "events_seen": events_seen,
        "events_eligible": events_eligible,
        "event_auto_coverage": _nested_float(
            event_decision_metrics_mapping,
            "event_auto_coverage",
        ),
        "event_abstain_rate": _nested_float(
            event_decision_metrics_mapping,
            "event_abstain_rate",
        ),
        "source": "prediction_diagnostics.event_decision_metrics",
        "prediction_diagnostics": prediction_diagnostics_mapping,
        "event_decision_metrics": event_decision_metrics_mapping,
    }
    return (
        events_seen,
        events_eligible,
        abstained_prediction_count,
        tp,
        tn,
        fp,
        fn,
        true_normal,
        true_anomalous,
        diagnostics,
        False,
    )


def _build_next_event_prediction_block(
    *,
    metrics: dict[str, object],
) -> MetricBlock | None:
    next_event_prediction = metrics.get("next_event_prediction")
    if not isinstance(next_event_prediction, dict):
        return None
    next_event_prediction_mapping = _as_metric_mapping(next_event_prediction)
    totals = _mapping_value(next_event_prediction_mapping, "totals")
    coverage = (
        None if not isinstance(totals, dict) else _nested_float(totals, "coverage")
    )
    return build_diagnostic_metric_block(
        request=DiagnosticMetricBlockRequest(
            prediction_unit=EvaluationUnit.NEXT_EVENT,
            label_unit=EvaluationUnit.NEXT_EVENT,
            headline_metrics={} if coverage is None else {"coverage": coverage},
            diagnostics=next_event_prediction_mapping,
        ),
    )


def _build_manual_workload_block(
    *,
    metrics: dict[str, object],
) -> MetricBlock | None:
    manual_workload = metrics.get("manual_workload_reduction")
    if not isinstance(manual_workload, dict):
        return None
    manual_workload_mapping = _as_metric_mapping(manual_workload)
    return build_diagnostic_metric_block(
        request=DiagnosticMetricBlockRequest(
            prediction_unit=EvaluationUnit.CLUSTER,
            label_unit=EvaluationUnit.CLUSTER,
            diagnostics=manual_workload_mapping,
        ),
    )


def _build_semi_automatic_workload_block(
    *,
    metrics: dict[str, object],
) -> MetricBlock | None:
    semi_automatic_workload = metrics.get("semi_automatic_workload_reduction")
    if not isinstance(semi_automatic_workload, dict):
        return None
    semi_automatic_workload_mapping = _as_metric_mapping(semi_automatic_workload)
    return build_diagnostic_metric_block(
        request=DiagnosticMetricBlockRequest(
            prediction_unit=EvaluationUnit.CLUSTER,
            label_unit=EvaluationUnit.CLUSTER,
            diagnostics=semi_automatic_workload_mapping,
        ),
    )


def _evaluation_unit_for_dataset(dataset: DatasetVariantConfig) -> EvaluationUnit:
    if dataset.evaluation_unit is not None:
        return dataset.evaluation_unit
    sequence = dataset.sequence
    if isinstance(sequence, ChronologicalStreamSequenceConfig):
        return EvaluationUnit.CONTINUOUS_EVENT_STREAM
    if isinstance(sequence, EntitySequenceConfig):
        return EvaluationUnit.SEQUENCE
    if isinstance(sequence, (FixedSequenceConfig, TimeSequenceConfig)):
        return EvaluationUnit.WINDOW
    return EvaluationUnit.SEQUENCE


def _compact_dataset_manifest(
    manifest: Mapping[str, Any],
    *,
    debug_reporting: bool,
) -> dict[str, Any]:
    compact_manifest = dict(manifest)
    if debug_reporting:
        return compact_manifest
    compact_manifest.pop("cache_paths", None)
    model_manifest = compact_manifest.get("model_manifest")
    if isinstance(model_manifest, dict):
        compact_manifest["model_manifest"] = _compact_model_manifest(
            model_manifest,
            debug_reporting=debug_reporting,
        )
    return compact_manifest


def _compact_run_metrics_report(
    report: Mapping[str, Any],
    *,
    debug_reporting: bool,
) -> dict[str, Any]:
    compact_report = dict(report)
    if debug_reporting:
        return compact_report
    metric_blocks = compact_report.get("metric_blocks")
    if not isinstance(metric_blocks, dict):
        return compact_report
    compact_report["metric_blocks"] = {
        str(scope): _compact_metric_block(str(scope), block)
        for scope, block in metric_blocks.items()
        if isinstance(block, dict)
    }
    return compact_report


def _compact_metric_block(
    scope: str,
    block: Mapping[str, Any],
) -> dict[str, Any]:
    compact_block = dict(block)
    diagnostics = compact_block.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return compact_block

    if scope == MetricScope.NEXT_EVENT_PREDICTION.value:
        compact_block["diagnostics"] = _compact_next_event_diagnostics(diagnostics)
        return compact_block

    if scope in {
        MetricScope.EVENT_LEVEL_DETECTION.value,
        MetricScope.SEQUENCE_LEVEL_DETECTION.value,
    }:
        compact_block["diagnostics"] = _compact_detection_diagnostics(diagnostics)
        return compact_block

    return compact_block


def _compact_detection_diagnostics(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    compact = {
        key: diagnostics[key]
        for key in ("source", "events_seen", "events_eligible")
        if key in diagnostics
    }
    for key in (
        "auto_coverage",
        "abstain_rate",
        "event_auto_coverage",
        "event_abstain_rate",
    ):
        if key in diagnostics:
            compact[key] = diagnostics[key]
    return compact


def _compact_next_event_diagnostics(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key in (
        "task",
        "totals",
        "top_k",
        "classification_top1_weighted",
        "exclusions",
        "vocabulary_policy",
    ):
        value = diagnostics.get(key)
        if value is None:
            continue
        if key == "top_k" and isinstance(value, dict):
            top_k = dict(value)
            top_k.pop("hit_count", None)
            compact[key] = top_k
            continue
        compact[key] = value
    return compact


def _compact_model_manifest(
    model_manifest: Mapping[str, Any],
    *,
    debug_reporting: bool,
) -> dict[str, Any]:
    compact_manifest = dict(model_manifest)
    if debug_reporting:
        return compact_manifest

    detector = compact_manifest.get("detector")
    if detector == "deeplog":
        compact_manifest.pop("parameter_models", None)
        compact_manifest.pop("skipped_parameter_models", None)
        return compact_manifest

    if detector == "deepcase":
        prediction_diagnostics = compact_manifest.get("prediction_diagnostics")
        if isinstance(prediction_diagnostics, dict):
            compact_manifest["prediction_diagnostics"] = _compact_deepcase_diagnostics(
                prediction_diagnostics,
            )
        return compact_manifest

    return compact_manifest


def _compact_deepcase_diagnostics(
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    compact = {
        key: diagnostics[key]
        for key in (
            "event_count",
            "confident_event_count",
            "abstained_event_count",
            "confident_anomaly_event_count",
            "sequence_confident_anomaly_count",
            "sequence_confident_normal_count",
            "sequence_abstained_count",
            "event_decision_metrics",
        )
        if key in diagnostics
    }
    event_decision_metrics = compact.get("event_decision_metrics")
    if isinstance(event_decision_metrics, dict):
        compact["event_decision_metrics"] = {
            key: event_decision_metrics[key]
            for key in (
                "event_count",
                "event_auto_decision_count",
                "event_abstained_decision_count",
                "event_auto_coverage",
                "event_abstain_rate",
                "event_tp",
                "event_fp",
                "event_tn",
                "event_fn",
                "event_precision",
                "event_recall",
                "event_f1",
                "event_accuracy",
                "event_predicted_normal_count",
                "event_predicted_anomalous_count",
                "event_true_normal_count",
                "event_true_anomalous_count",
            )
            if key in event_decision_metrics
        }
    return compact


def _build_split_policy(
    *,
    bundle: ExperimentBundle,
    sequences: SequenceBuilder,
    model_summary: ModelRunSummary,
    split_summary: SequenceSplitSummary | None = None,
    raw_entry_split_summary: RawEntrySplitSummary | None = None,
) -> dict[str, object]:
    if split_summary is None:
        split_summary = build_sequence_split_summary(
            sequences,
            sequence_summary=model_summary.sequence_summary,
        )
    policy: dict[str, object] = {
        "train_fraction": split_summary.requested_train_fraction,
        "test_fraction": split_summary.requested_test_fraction,
        "train_on_normal_entities_only": split_summary.train_on_normal_entities_only,
    }
    if raw_entry_split_summary is None:
        raw_entry_split_summary = sequences.build_raw_entry_split_summary()
    sequence = bundle.dataset.sequence
    raw_split = getattr(sequence, "split", None)
    if raw_split is not None:
        policy["application_order"] = raw_split.application_order.value
        policy["straddling_group_policy"] = raw_split.straddling_group_policy.value
        policy["raw_entry_split"] = serialise_config(raw_split)
    else:
        policy["application_order"] = None
        policy["straddling_group_policy"] = None
        policy["raw_entry_split"] = None
    policy["raw_entry_split_summary"] = (
        None if raw_entry_split_summary is None else raw_entry_split_summary.as_dict()
    )
    return policy


def _build_stream_segment_policy(dataset: DatasetVariantConfig) -> dict[str, object]:
    sequence = dataset.sequence
    if isinstance(sequence, ChronologicalStreamSequenceConfig):
        return {
            "mode": "continuous_event_stream",
            "chunk_size": sequence.chunk_size,
        }
    if isinstance(sequence, EntitySequenceConfig):
        return {
            "mode": "entity_sequence",
            "train_on_normal_entities_only": sequence.train_on_normal_entities_only,
        }
    if isinstance(sequence, FixedSequenceConfig):
        policy: dict[str, object] = {
            "mode": "fixed_window",
            "window_size": sequence.window_size,
            "step": sequence.step,
        }
        if hasattr(sequence, "window_basis"):
            policy["window_basis"] = sequence.window_basis.value
            policy["window_alignment_offset"] = sequence.window_alignment_offset
        policy["trailing_window_policy"] = "drop"
        return policy
    if isinstance(sequence, TimeSequenceConfig):
        return {
            "mode": "time_window",
            "time_span_ms": sequence.time_span_ms,
            "step": sequence.step,
        }
    return {"mode": "unknown"}


def _metric_count(metrics: dict[str, object], key: str) -> int | None:
    value = metrics.get(key)
    return value if isinstance(value, int) else None


def _as_metric_mapping(mapping: object) -> Any:  # noqa: ANN401
    builtins_mapping = msgspec.to_builtins(mapping)
    if not isinstance(builtins_mapping, dict):
        msg = "Expected mapping-like metric payload."
        raise TypeError(msg)
    return builtins_mapping


def _mapping_value(mapping: Any, key: str) -> Any:  # noqa: ANN401
    return mapping.get(key)


def _require_int(value: int | None) -> int:
    if value is None:
        msg = "Expected integer metric value."
        raise TypeError(msg)
    return value


def _int_value(
    mapping: Any,  # noqa: ANN401
    key: str,
    *,
    default: int | None = None,
) -> int:
    value = mapping.get(key)
    if isinstance(value, int):
        return value
    if default is not None:
        return default
    msg = f"Expected integer metric value for {key!r}."
    raise TypeError(msg)


def _int_count_map(raw_counts: object | None) -> dict[int, int] | None:
    if raw_counts is None or not isinstance(raw_counts, dict):
        return None
    counts: dict[int, int] = {}
    for key, value in raw_counts.items():
        if not isinstance(key, int) or not isinstance(value, int):
            return None
        counts[key] = value
    return counts


def _nested_float(mapping: Any, key: str) -> float | None:  # noqa: ANN401
    if not isinstance(mapping, dict):
        return None
    value = mapping.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def build_environment_metadata(
    *,
    bundle: ExperimentBundle,
    result_paths: ResultPaths,
) -> dict[str, object]:
    """Capture the local environment for reproducibility and provenance.

    Args:
        bundle (ExperimentBundle): Resolved experiment bundle.
        result_paths (ResultPaths): Materialised artifact paths for the run.

    Returns:
        dict[str, object]: Serialisable environment metadata.
    """
    return {
        "recorded_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_fingerprint": result_paths.run_fingerprint,
        "command": list(sys.argv),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
        },
        "repository": {
            "root": bundle.repo_root.as_posix(),
            "git_commit": _read_git_commit(bundle.repo_root),
        },
        "packages": {
            "anomalog": _package_version("anomalog"),
            "deepcase": _package_version("deepcase"),
        },
    }


def sha256_for_file(path: Path) -> str:
    """Hash a file without loading it all into memory.

    Args:
        path (Path): File path to hash.

    Returns:
        str: SHA-256 hex digest for the file contents.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _structured_parser_name(bundle: ExperimentBundle) -> str:
    if bundle.dataset.structured_parser is not None:
        return bundle.dataset.structured_parser
    if bundle.dataset.preset is not None:
        return bundle.dataset.preset
    msg = "Dataset manifest requires either a structured parser or a preset."
    raise ValueError(msg)


def _experiment_metadata(bundle: ExperimentBundle) -> dict[str, object]:
    return {"name": bundle.experiment_name, "groups": list(bundle.experiment_groups)}


def _read_git_commit(repo_root: Path) -> str | None:
    git_head_path = repo_root / ".git" / "HEAD"
    if not git_head_path.exists():
        return None
    head_value = git_head_path.read_text(encoding="utf-8").strip()
    if not head_value.startswith("ref: "):
        return head_value or None
    ref_path = repo_root / ".git" / head_value.removeprefix("ref: ")
    if not ref_path.exists():
        return None
    commit = ref_path.read_text(encoding="utf-8").strip()
    return commit or None


def _package_version(dist_name: str) -> str | None:
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return None
