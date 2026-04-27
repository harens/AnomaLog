"""CLI entrypoint for an AnomaLog experiment sweep."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from prefect.logging.configuration import (
    DEFAULT_LOGGING_SETTINGS_PATH,
    load_logging_config,
)
from prefect.logging.formatters import PrefectFormatter

from anomalog.io_utils import get_shared_console
from anomalog.sequences import EntitySequenceBuilder
from experiments import ConfigError
from experiments.config import ExperimentBundle, load_experiment_bundles
from experiments.datasets import build_dataset_spec
from experiments.models import ProgressHint, RunProgressPlan, run_model
from experiments.models.evaluate import PredictionOutputConfig
from experiments.results import (
    build_sequence_split_summary,
    prepare_result_paths,
    write_run_outputs,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_PREFECT_LOGGING_CONFIG = load_logging_config(DEFAULT_LOGGING_SETTINGS_PATH)


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
) -> list[Path]:
    """Run one sweep config and return all concrete result directories.

    Args:
        config_path (Path): Sweep config TOML path to execute.
        force (bool): Whether to replace an existing deterministic result
            directories.
        write_predictions (bool): Whether to persist `predictions.jsonl` for
            each concrete run.

    Returns:
        list[Path]: Deterministic run directories containing the written
            artifacts.

    Raises:
        ConfigError: If the sweep does not expand to any concrete runs.
    """
    bundles = load_experiment_bundles(config_path)
    if not bundles:
        msg = f"Sweep {config_path} did not expand to any concrete runs."
        raise ConfigError(msg)
    max_workers = _resolve_max_workers(
        requested_workers=bundles[0].sweep.max_workers,
        bundle_count=len(bundles),
    )
    if max_workers == 1 or len(bundles) == 1:
        return [
            _run_bundle(
                bundle,
                force=force,
                write_predictions=write_predictions,
            )
            for bundle in bundles
        ]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(
                _run_bundle_from_sweep_payload,
                [
                    (config_path, index, force, write_predictions)
                    for index in range(len(bundles))
                ],
            ),
        )


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


def _run_bundle_from_sweep_payload(
    payload: tuple[Path, int, bool, bool],
) -> Path:
    config_path, index, force, write_predictions = payload
    bundle = load_experiment_bundles(config_path)[index]
    return _run_bundle(bundle, force=force, write_predictions=write_predictions)


def _run_bundle(
    bundle: ExperimentBundle,
    *,
    force: bool = False,
    write_predictions: bool = False,
) -> Path:
    """Execute one concrete run derived from a sweep.

    Args:
        bundle (ExperimentBundle): Concrete run bundle to execute.
        force (bool): Whether to replace an existing deterministic result
            directory.
        write_predictions (bool): Whether to persist `predictions.jsonl` for
            the concrete run.

    Returns:
        Path: Deterministic run directory containing the written artefacts.

    Raises:
        FileExistsError: If the deterministic result directory already exists
            and `force` is false.
    """
    result_paths = prepare_result_paths(bundle)
    if result_paths.run_dir.exists():
        if not force:
            msg = (
                f"Result directory already exists: {result_paths.run_dir}. "
                "Use --force to replace it."
            )
            raise FileExistsError(msg)
        shutil.rmtree(result_paths.run_dir)
    result_paths.run_dir.mkdir(parents=True, exist_ok=True)

    with _experiment_logger(
        result_paths.run_log_path,
        run_name=bundle.concrete_name,
    ) as logger:
        logger.info("Loaded sweep config from %s", bundle.sweep_path)
        logger.info("Using dataset config %s", bundle.dataset_path)
        logger.info("Using model config %s", bundle.model_path)
        logger.info("Running concrete sweep variant %s", bundle.concrete_name)
        if bundle.applied_overrides:
            logger.info("Applied overrides: %s", bundle.applied_overrides)
        dataset_spec = build_dataset_spec(bundle.dataset, repo_root=bundle.repo_root)
        logger.info("Building dataset %s", bundle.dataset.dataset_name)
        templated = dataset_spec.build()
        sequences = bundle.dataset.sequence.apply(templated)
        logger.info("Dataset ready; starting model run for %s", bundle.model.detector)
        train_sequence_count_hint = sequences.train_sequence_count_hint()
        sequence_count_hint = sequences.sequence_count_hint()
        model_summary = run_model(
            sequence_factory=lambda: iter(sequences),
            config=bundle.model,
            prediction_output=PredictionOutputConfig(
                predictions_path=result_paths.predictions_path,
                write_predictions=write_predictions,
            ),
            logger=logger,
            progress_plan=RunProgressPlan(
                train=(
                    None
                    if train_sequence_count_hint is None
                    else ProgressHint(
                        total=train_sequence_count_hint,
                        unit=sequences.train_sequence_count_unit_hint(),
                    )
                ),
                score=(
                    None
                    if sequence_count_hint is None or train_sequence_count_hint is None
                    else ProgressHint(
                        total=sequence_count_hint - train_sequence_count_hint,
                    )
                ),
            ),
        )
        split_summary = build_sequence_split_summary(
            sequences,
            sequence_summary=model_summary.sequence_summary,
        )
        if isinstance(sequences, EntitySequenceBuilder) and (
            sequences.train_on_normal_entities_only
        ):
            logger.warning(
                "Normal-only training excludes anomalous entities from train; "
                "this run satisfied requested train_fraction=%.4f over the full "
                "entity population (train=%s, eligible_normals=%s, total=%s)",
                split_summary.requested_train_fraction,
                model_summary.sequence_summary.train_sequence_count,
                split_summary.eligible_train_sequence_count,
                model_summary.sequence_summary.sequence_count,
            )
        logger.info(
            "Model run complete with %s sequences",
            model_summary.sequence_summary.sequence_count,
        )
        write_run_outputs(
            bundle=bundle,
            templated=templated,
            sequences=sequences,
            model_summary=model_summary,
            result_paths=result_paths,
        )
        logger.info(
            "Wrote experiment artifacts to %s",
            shlex.quote(str(result_paths.run_dir)),
        )
    return result_paths.run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Parser for the experiment runner CLI.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a sweep config TOML file under experiments/configs/sweeps.",
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
    return parser


def main() -> int:
    """Run the CLI entrypoint.

    Returns:
        int: Process exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        run_experiment(
            args.config,
            force=args.force,
            write_predictions=args.write_predictions,
        )
    except (ConfigError, FileExistsError, ValueError) as exc:
        parser.exit(status=2, message=f"{exc}\n")
    return 0


def _experiment_logger_name(run_name: str) -> str:
    """Return the stable logger name used for one concrete experiment run.

    Args:
        run_name (str): Human-readable concrete sweep variant name.

    Returns:
        str: Logger name displayed by the Prefect-style formatter.
    """
    return f"experiments.run.{run_name}"


@contextmanager
def _experiment_logger(log_path: Path, *, run_name: str) -> Iterator[logging.Logger]:
    logger = logging.getLogger(_experiment_logger_name(run_name))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = build_prefect_standard_formatter()
    # Writes log lines for permanent storage
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    # Writes log lines to the console
    stream_handler = SharedConsoleHandler()
    stream_handler.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    try:
        yield logger
    finally:
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)


if __name__ == "__main__":
    raise SystemExit(main())
