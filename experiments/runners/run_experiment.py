"""CLI entrypoint for a single AnomaLog experiment run."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from anomalog.sequences import EntitySequenceBuilder
from experiments import ConfigError
from experiments.config import load_experiment_bundle
from experiments.datasets import build_dataset_spec
from experiments.models import run_model
from experiments.results import (
    build_sequence_split_summary,
    prepare_result_paths,
    write_run_outputs,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


def run_experiment(config_path: Path, *, force: bool = False) -> Path:
    """Run a single experiment from a TOML config path."""
    bundle = load_experiment_bundle(config_path)
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

    with _experiment_logger(result_paths.run_log_path) as logger:
        logger.info("Loaded run config from %s", bundle.run_path)
        logger.info("Using dataset config %s", bundle.dataset_path)
        logger.info("Using model config %s", bundle.model_path)
        dataset_spec = build_dataset_spec(bundle.dataset, repo_root=bundle.repo_root)
        logger.info("Building dataset %s", bundle.dataset.dataset_name)
        templated = dataset_spec.build()
        sequences = bundle.dataset.sequence.apply(templated)
        logger.info("Dataset ready; starting model run for %s", bundle.model.detector)
        model_summary = run_model(
            sequence_factory=lambda: iter(sequences),
            config=bundle.model,
            predictions_path=result_paths.predictions_path,
            logger=logger,
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
        logger.info("Wrote experiment artifacts to %s", result_paths.run_dir)
    return result_paths.run_dir


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to a run config TOML file under experiments/configs/runs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing deterministic result directory.",
    )
    return parser


def main() -> int:
    """Run the CLI and print the result directory."""
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        run_dir = run_experiment(args.config, force=args.force)
    except (ConfigError, FileExistsError, ValueError) as exc:
        parser.exit(status=2, message=f"{exc}\n")
    sys.stdout.write(f"{run_dir}\n")
    return 0


@contextmanager
def _experiment_logger(log_path: Path) -> Iterator[logging.Logger]:
    logger = logging.getLogger(f"experiments.run.{log_path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stderr)
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
