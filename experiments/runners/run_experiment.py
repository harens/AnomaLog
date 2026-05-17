"""CLI entrypoint for an AnomaLog dataset experiment manifest."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from prefect.logging.configuration import (
    DEFAULT_LOGGING_SETTINGS_PATH,
    load_logging_config,
)
from prefect.logging.formatters import PrefectFormatter

from anomalog.io_utils import get_shared_console
from experiments import ConfigError
from experiments.config import ExperimentBundle, load_experiment_bundles
from experiments.datasets import build_dataset_spec
from experiments.models import ProgressHint, RunProgressPlan, run_model
from experiments.models.evaluate import PredictionOutputConfig
from experiments.registry import resolve_registry_experiment
from experiments.results import (
    ResultWriteContext,
    build_run_metrics_report,
    build_sequence_split_summary,
    prepare_result_paths,
    write_run_outputs,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_PREFECT_LOGGING_CONFIG = load_logging_config(DEFAULT_LOGGING_SETTINGS_PATH)


@dataclass(frozen=True, slots=True)
class _BundleGroupRequest:
    config_path: Path
    bundles: list[ExperimentBundle]
    bundle_indexes: list[int]
    force: bool
    write_predictions: bool
    debug_reporting: bool


class _FutureWithResult(Protocol):
    def result(self) -> Path:
        """Return the future result when the worker completes."""


@dataclass(frozen=True, slots=True)
class RegisteredExperimentRunRequest:
    """Request metadata for a registry-backed experiment run.

    Attributes:
        experiment_name (str): Named registry experiment to execute.
        registry_path (Path): Path to the registry TOML file.
        repo_root (Path | None): Repository root used to resolve relative
            paths.
        force (bool): Whether to replace existing deterministic run outputs.
        write_predictions (bool): Whether to persist `predictions.jsonl`.
        debug_reporting (bool): Whether to keep verbose diagnostics.
        console (bool): Whether to emit console progress through Prefect.
    """

    experiment_name: str
    registry_path: Path = Path("experiments/configs/registry.toml")
    repo_root: Path | None = None
    force: bool = False
    write_predictions: bool = False
    debug_reporting: bool = False
    console: bool = True


class SharedConsoleHandler(logging.Handler):
    """Write formatted log lines through the shared Rich console."""

    def emit(self, record: logging.LogRecord) -> None:
        """Render one log record through the shared console.

        Args:
            record (logging.LogRecord): Log record to render.
        """
        get_shared_console().print(self.format(record), soft_wrap=True)


def build_prefect_standard_formatter() -> PrefectFormatter:
    """Build Prefect's standard formatter from the active logging config.

    Returns:
        PrefectFormatter: Formatter matching Prefect's standard log layout.
    """
    formatter_config = _PREFECT_LOGGING_CONFIG["formatters"]["standard"]
    return PrefectFormatter(
        format=formatter_config["format"],
        datefmt=formatter_config["datefmt"],
        flow_run_fmt=formatter_config["flow_run_fmt"],
        task_run_fmt=formatter_config["task_run_fmt"],
    )


def run_experiment(
    config_path: Path,
    *,
    force: bool = False,
    write_predictions: bool = False,
    debug_reporting: bool = False,
) -> list[Path]:
    """Run one dataset manifest and return all concrete result directories.

    Args:
        config_path (Path): Dataset manifest TOML path to execute.
        force (bool): Whether to replace an existing deterministic result
            directories.
        write_predictions (bool): Whether to persist `predictions.jsonl` for
            each concrete run.
        debug_reporting (bool): Whether to keep the verbose diagnostic payloads
            and logging output in the run artefacts.

    Returns:
        list[Path]: Deterministic run directories containing the written
            artefacts for bundles that completed successfully. Bundle
            failures are logged and skipped so the remaining runs in the group
            can keep going.

    Raises:
        ConfigError: If the manifest does not expand to any concrete runs.
    """
    bundles = load_experiment_bundles(config_path)
    if not bundles:
        msg = f"Manifest {config_path} did not expand to any concrete runs."
        raise ConfigError(msg)
    results: list[Path] = []
    for bundle_indexes in _group_bundle_indexes_by_run_group(bundles):
        results.extend(
            _run_bundle_group(
                _BundleGroupRequest(
                    config_path=config_path,
                    bundles=bundles,
                    bundle_indexes=bundle_indexes,
                    force=force,
                    write_predictions=write_predictions,
                    debug_reporting=debug_reporting,
                ),
            ),
        )
    return results


def run_registered_experiment(request: RegisteredExperimentRunRequest) -> list[Path]:
    """Run one named registry experiment and return its result directories.

    Args:
        request (RegisteredExperimentRunRequest): Registry-backed run settings.

    Returns:
        list[Path]: Deterministic result directories for the selected experiment.
    """
    resolved_repo_root = Path.cwd() if request.repo_root is None else request.repo_root
    resolved = resolve_registry_experiment(
        request.experiment_name,
        registry_path=request.registry_path,
        repo_root=resolved_repo_root,
    )
    return [
        _run_bundle(
            bundle,
            force=request.force,
            write_predictions=request.write_predictions,
            debug_reporting=request.debug_reporting,
            console=request.console,
        )
        for bundle in resolved.bundles
    ]


def _resolve_max_workers(
    *,
    requested_workers: int | str,
    bundle_count: int,
) -> int:
    if requested_workers == "auto":
        return min(bundle_count, os.cpu_count() or 1)
    if not isinstance(requested_workers, int):
        msg = f"Unsupported max_workers value: {requested_workers!r}"
        raise TypeError(msg)
    return min(bundle_count, requested_workers)


def _run_bundle_from_manifest_payload(
    payload: tuple[Path, int, bool, bool, bool],
) -> Path:
    config_path, index, force, write_predictions, debug_reporting = payload
    bundle = load_experiment_bundles(config_path)[index]
    return _run_bundle(
        bundle,
        force=force,
        write_predictions=write_predictions,
        debug_reporting=debug_reporting,
    )


def _group_bundle_indexes_by_run_group(
    bundles: list[ExperimentBundle],
) -> list[list[int]]:
    grouped_indexes: dict[str, list[int]] = {}
    run_group_order: list[str] = []
    for index, bundle in enumerate(bundles):
        run_group = getattr(bundle, "run_group", "default")
        if not isinstance(run_group, str):
            msg = f"Unsupported run_group value: {run_group!r}"
            raise TypeError(msg)
        if run_group not in grouped_indexes:
            run_group_order.append(run_group)
            grouped_indexes[run_group] = []
        grouped_indexes[run_group].append(index)
    return [grouped_indexes[run_group] for run_group in run_group_order]


def _run_bundle_group(
    request: _BundleGroupRequest,
) -> list[Path]:
    grouped_bundles = [request.bundles[index] for index in request.bundle_indexes]
    max_workers = _resolve_max_workers(
        requested_workers=grouped_bundles[0].sweep.max_workers,
        bundle_count=len(grouped_bundles),
    )
    if max_workers == 1 or len(grouped_bundles) == 1:
        results_by_index, failures_by_index = _run_bundle_group_serial(
            grouped_bundles,
            force=request.force,
            write_predictions=request.write_predictions,
            debug_reporting=request.debug_reporting,
        )
    else:
        results_by_index, failures_by_index = _run_bundle_group_parallel(
            request=request,
            grouped_bundles=grouped_bundles,
            max_workers=max_workers,
        )
    if failures_by_index:
        _write_line("One or more runs in this group failed:")
        for index in sorted(failures_by_index):
            failure = failures_by_index[index]
            _write_line(f"  - {failure}")
    return [results_by_index[index] for index in sorted(results_by_index)]


def _run_bundle_group_serial(
    grouped_bundles: list[ExperimentBundle],
    *,
    force: bool,
    write_predictions: bool,
    debug_reporting: bool,
) -> tuple[dict[int, Path], dict[int, str]]:
    results_by_index: dict[int, Path] = {}
    failures_by_index: dict[int, str] = {}
    for index, bundle in enumerate(grouped_bundles):
        result, failure = _run_bundle_with_failure_capture(
            bundle,
            force=force,
            write_predictions=write_predictions,
            debug_reporting=debug_reporting,
        )
        if result is not None:
            results_by_index[index] = result
        if failure is not None:
            failures_by_index[index] = failure
    return results_by_index, failures_by_index


def _run_bundle_group_parallel(
    *,
    request: _BundleGroupRequest,
    grouped_bundles: list[ExperimentBundle],
    max_workers: int,
) -> tuple[dict[int, Path], dict[int, str]]:
    results_by_index: dict[int, Path] = {}
    failures_by_index: dict[int, str] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        submit = getattr(executor, "submit", None)
        if submit is None:
            results = executor.map(
                _run_bundle_from_manifest_payload,
                [
                    (
                        request.config_path,
                        index,
                        request.force,
                        request.write_predictions,
                        request.debug_reporting,
                    )
                    for index in request.bundle_indexes
                ],
            )
            results_by_index.update(dict(enumerate(results)))
        else:
            future_to_index = {
                submit(
                    _run_bundle_from_manifest_payload,
                    (
                        request.config_path,
                        index,
                        request.force,
                        request.write_predictions,
                        request.debug_reporting,
                    ),
                ): index
                for index in request.bundle_indexes
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                bundle = grouped_bundles[index]
                result, failure = _capture_future_result(future, bundle)
                if result is not None:
                    results_by_index[index] = result
                if failure is not None:
                    failures_by_index[index] = failure
    return results_by_index, failures_by_index


def _run_bundle(
    bundle: ExperimentBundle,
    *,
    force: bool = False,
    write_predictions: bool = False,
    debug_reporting: bool = False,
    console: bool = True,
) -> Path:
    """Execute one concrete run derived from a dataset manifest.

    Args:
        bundle (ExperimentBundle): Concrete run bundle to execute.
        force (bool): Whether to replace an existing deterministic result
            directory.
        write_predictions (bool): Whether to persist `predictions.jsonl` for
            the concrete run.
        debug_reporting (bool): Whether to keep verbose diagnostic payloads in
            the run artefacts and logs.
        console (bool): Whether to mirror log output to the shared console.

    Returns:
        Path: Deterministic run directory containing the written artefacts.

    Raises:
        FileExistsError: If the result path exists but is not a directory.
    """
    result_paths = prepare_result_paths(bundle)
    if result_paths.run_dir.exists():
        if not result_paths.run_dir.is_dir():
            msg = f"Result path exists but is not a directory: {result_paths.run_dir}"
            raise FileExistsError(msg)
        if result_paths.metrics_path.is_file() and not force:
            return result_paths.run_dir
        shutil.rmtree(result_paths.run_dir)
    result_paths.run_dir.mkdir(parents=True, exist_ok=True)

    with _experiment_logger(
        result_paths.run_log_path,
        run_name=bundle.concrete_name,
        console=console,
    ) as logger:
        logger.info("Loaded experiment config from %s", bundle.sweep_path)
        logger.info("Using dataset config %s", bundle.dataset_path)
        logger.info("Using model config %s", bundle.model_path)
        logger.info("Running concrete experiment variant %s", bundle.concrete_name)
        if bundle.applied_overrides:
            logger.info("Applied overrides: %s", bundle.applied_overrides)
        dataset_spec = build_dataset_spec(bundle.dataset, repo_root=bundle.repo_root)
        logger.info("Building dataset %s", bundle.dataset.dataset_name)
        templated = dataset_spec.build()
        sequence_view_for_summary = bundle.dataset.sequence.apply(templated)
        logger.info("Dataset ready; starting model run for %s", bundle.model.detector)
        split_counts_hint = sequence_view_for_summary.split_count_hint()
        model_summary = run_model(
            sequence_factory=lambda: iter(
                bundle.dataset.sequence.apply(templated),
            ),
            config=bundle.model,
            prediction_output=PredictionOutputConfig(
                predictions_path=result_paths.predictions_path,
                write_predictions=write_predictions,
            ),
            logger=logger,
            progress_plan=RunProgressPlan(
                train=(
                    None
                    if split_counts_hint is None
                    else ProgressHint(
                        total=split_counts_hint.train_count,
                        unit=sequence_view_for_summary.train_sequence_count_unit_hint(),
                    )
                ),
                score=(
                    None
                    if split_counts_hint is None
                    else ProgressHint(
                        total=split_counts_hint.test_count,
                    )
                ),
            ),
        )
        sequences_for_split_summary = bundle.dataset.sequence.apply(templated)
        split_summary = build_sequence_split_summary(
            sequences_for_split_summary,
            sequence_summary=model_summary.sequence_summary,
        )
        train_on_normal_entities_only = split_summary.train_on_normal_entities_only
        if train_on_normal_entities_only is not None:
            total_sequences = model_summary.sequence_summary.sequence_count
            test_sequences = model_summary.sequence_summary.test_sequence_count
            train_pool_sequences = split_summary.train_pool_sequence_count
            logger.info(
                ("Fixed entity split: train_pool=%s, train=%s, ignored=%s, test=%s"),
                train_pool_sequences,
                split_summary.realised_train_sequence_count,
                split_summary.ignored_sequence_count,
                test_sequences,
            )
        if train_on_normal_entities_only:
            logger.warning(
                "Normal-only training uses the chronological train pool and "
                "excludes ineligible entities from training; requested "
                "train_fraction=%.4f, realised_train=%s, eligible_normals=%s, "
                "train_pool=%s, ineligible_prefix=%s, total=%s",
                split_summary.requested_train_fraction,
                split_summary.realised_train_sequence_count,
                split_summary.eligible_train_sequence_count,
                split_summary.train_pool_sequence_count,
                split_summary.ineligible_train_pool_count,
                total_sequences,
            )
        metric_report = build_run_metrics_report(
            bundle=bundle,
            sequences=bundle.dataset.sequence.apply(templated),
            model_summary=model_summary,
            debug_reporting=debug_reporting,
        )
        _log_metric_report(logger, metric_report, debug_reporting=debug_reporting)
        logger.info(
            "Model run complete with %s sequences",
            model_summary.sequence_summary.sequence_count,
        )
        sequences_for_outputs = bundle.dataset.sequence.apply(templated)
        write_run_outputs(
            context=ResultWriteContext(
                bundle=bundle,
                templated=templated,
                sequences=sequences_for_outputs,
                model_summary=model_summary,
                result_paths=result_paths,
                debug_reporting=debug_reporting,
            ),
        )
        logger.info(
            "Wrote experiment artifacts to %s",
            shlex.quote(str(result_paths.run_dir)),
        )
    return result_paths.run_dir


def _log_metric_report(
    logger: logging.Logger,
    report: dict[str, Any],
    *,
    debug_reporting: bool,
) -> None:
    """Log the selected primary metric scope and notable secondary blocks.

    Args:
        logger (logging.Logger): Experiment-run logger used for output.
        report (dict[str, Any]): Serialised metrics report for the run.
        debug_reporting (bool): Whether verbose diagnostics should be logged.
    """
    _log_primary_metric_report(logger, report, debug_reporting=debug_reporting)
    _log_metric_block_warnings(logger, report, debug_reporting=debug_reporting)


def _log_primary_metric_report(
    logger: logging.Logger,
    report: dict[str, Any],
    *,
    debug_reporting: bool,
) -> None:
    """Log the selected primary metric block details.

    Args:
        logger (logging.Logger): Experiment-run logger used for output.
        report (dict[str, Any]): Serialised metrics report for the run.
        debug_reporting (bool): Whether verbose diagnostics should be logged.
    """
    primary_metric_scope = report.get("primary_metric_scope")
    logger.info("Primary metric scope: %s", primary_metric_scope)
    metric_blocks = report.get("metric_blocks")
    if isinstance(metric_blocks, dict) and isinstance(primary_metric_scope, str):
        primary_metrics = metric_blocks.get(primary_metric_scope)
    else:
        primary_metrics = None
    if isinstance(primary_metrics, dict):
        primary_metrics_mapping: dict[str, Any] = primary_metrics
        status = primary_metrics_mapping.get("status")
        logger.info("Primary metric status: %s", status)
        headline_metrics = primary_metrics_mapping.get("headline_metrics")
        if isinstance(headline_metrics, dict) and headline_metrics:
            logger.info("Primary headline metrics: %s", headline_metrics)
        if debug_reporting:
            diagnostics = primary_metrics_mapping.get("diagnostics")
            if isinstance(diagnostics, dict):
                logger.debug("Primary diagnostics: %s", diagnostics)


def _log_metric_block_warnings(
    logger: logging.Logger,
    report: dict[str, Any],
    *,
    debug_reporting: bool,
) -> None:
    """Log warnings for invalid or diagnostic-only metric blocks.

    Args:
        logger (logging.Logger): Experiment-run logger used for output.
        report (dict[str, Any]): Serialised metrics report for the run.
        debug_reporting (bool): Whether verbose diagnostics should be logged.
    """
    metric_blocks = report.get("metric_blocks")
    if not isinstance(metric_blocks, dict):
        return
    for scope, block in metric_blocks.items():
        if not isinstance(block, dict):
            continue
        block_mapping: dict[str, Any] = block
        status = block_mapping.get("status")
        if status == "invalid":
            reason = block_mapping.get("invalid_reason")
            if reason is not None:
                logger.warning("Metric block %s is %s: %s", scope, status, reason)
            else:
                logger.warning("Metric block %s is %s", scope, status)
        elif status == "diagnostic_only" and debug_reporting:
            logger.info("Metric block %s is diagnostic-only", scope)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Parser for the experiment runner CLI.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--config",
        type=Path,
        help="Path to a dataset manifest TOML file under experiments/configs/datasets.",
    )
    source_group.add_argument(
        "--experiment",
        help=(
            "Registry experiment name to resolve from "
            "experiments/configs/registry.toml."
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("experiments/configs/registry.toml"),
        help="Path to the named experiment registry TOML file.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used to resolve registry-relative paths.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing deterministic result directory.",
    )
    parser.add_argument(
        "--write-predictions",
        action="store_true",
        help="Write predictions.jsonl for each run.",
    )
    parser.add_argument(
        "--debug-reporting",
        action="store_true",
        help="Keep verbose diagnostic fields and logging in the run artefacts.",
    )
    return parser


def main() -> int:
    """Run the CLI entrypoint.

    Returns:
        int: Process exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        if getattr(args, "experiment", None) is not None:
            run_registered_experiment(
                RegisteredExperimentRunRequest(
                    experiment_name=args.experiment,
                    registry_path=getattr(
                        args,
                        "registry",
                        Path("experiments/configs/registry.toml"),
                    ),
                    repo_root=getattr(args, "repo_root", None),
                    force=getattr(args, "force", False),
                    write_predictions=getattr(args, "write_predictions", False),
                    debug_reporting=getattr(args, "debug_reporting", False),
                ),
            )
        else:
            run_experiment(
                args.config,
                force=getattr(args, "force", False),
                write_predictions=getattr(args, "write_predictions", False),
                debug_reporting=getattr(args, "debug_reporting", False),
            )
    except (ConfigError, FileExistsError, ValueError) as exc:
        parser.exit(status=2, message=f"{exc}\n")
    return 0


def _experiment_logger_name(run_name: str) -> str:
    """Return the stable logger name used for one concrete experiment run.

    Args:
        run_name (str): Human-readable concrete experiment variant name.

    Returns:
        str: Logger name displayed by the Prefect-style formatter.
    """
    return f"experiments.run.{run_name}"


@contextmanager
def _experiment_logger(
    log_path: Path,
    *,
    run_name: str,
    console: bool = True,
) -> Iterator[logging.Logger]:
    logger = logging.getLogger(_experiment_logger_name(run_name))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = build_prefect_standard_formatter()
    # Writes log lines for permanent storage
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    if console:
        # Writes log lines to the console
        stream_handler = SharedConsoleHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    try:
        yield logger
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


def _write_line(message: str) -> None:
    sys.stdout.write(message + "\n")


def _run_bundle_with_failure_capture(
    bundle: ExperimentBundle,
    *,
    force: bool,
    write_predictions: bool,
    debug_reporting: bool,
) -> tuple[Path | None, str | None]:
    try:
        return (
            _run_bundle(
                bundle,
                force=force,
                write_predictions=write_predictions,
                debug_reporting=debug_reporting,
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001
        return None, _format_bundle_failure(bundle, exc)


def _capture_future_result(
    future: _FutureWithResult,
    bundle: ExperimentBundle,
) -> tuple[Path | None, str | None]:
    try:
        result = future.result()
    except Exception as exc:  # noqa: BLE001
        return None, _format_bundle_failure(bundle, exc)
    return result, None


def _format_bundle_failure(bundle: ExperimentBundle, exc: Exception) -> str:
    detail = str(exc).strip() or exc.__class__.__name__
    return f"{bundle.concrete_name}: {detail}"


if __name__ == "__main__":
    raise SystemExit(main())
