"""Result-directory management and manifest utilities."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

import msgspec

from experiments.config import serialize_config
from experiments.datasets import dataset_source_summary

if TYPE_CHECKING:
    from pathlib import Path

    from anomalog.parsers.template import TemplatedDataset
    from anomalog.sequences import SequenceBuilder, SequenceSplitSummary
    from experiments.config import ExperimentBundle
    from experiments.models import ModelRunSummary, SequenceSummary


@dataclass(frozen=True, slots=True)
class ResultPaths:
    """Concrete artifact paths inside a single run directory."""

    run_fingerprint: str
    run_dir: Path
    config_path: Path
    dataset_manifest_path: Path
    metrics_path: Path
    predictions_path: Path
    environment_path: Path
    run_log_path: Path

    @classmethod
    def for_bundle(cls, bundle: ExperimentBundle) -> ResultPaths:
        """Create deterministic result paths for the experiment bundle."""
        combined_config = bundle.normalized_config()
        run_fingerprint = stable_fingerprint(combined_config)
        results_root = (
            bundle.repo_root / bundle.run.results_root
            if not bundle.run.results_root.is_absolute()
            else bundle.run.results_root
        )
        run_dir = results_root / bundle.run.name / run_fingerprint[:12]
        return cls(
            run_fingerprint=run_fingerprint,
            run_dir=run_dir,
            config_path=run_dir / "run_config.json",
            dataset_manifest_path=run_dir / "dataset_manifest.json",
            metrics_path=run_dir / "metrics.json",
            predictions_path=run_dir / "predictions.jsonl",
            environment_path=run_dir / "environment.json",
            run_log_path=run_dir / "run.log",
        )


def prepare_result_paths(bundle: ExperimentBundle) -> ResultPaths:
    """Create deterministic result paths for the experiment bundle."""
    return ResultPaths.for_bundle(bundle)


def write_run_outputs(
    *,
    bundle: ExperimentBundle,
    templated: TemplatedDataset,
    sequences: SequenceBuilder,
    model_summary: ModelRunSummary,
    result_paths: ResultPaths,
) -> None:
    """Persist the full experiment result bundle."""
    _write_json(result_paths.config_path, bundle.normalized_config())
    _write_json(
        result_paths.dataset_manifest_path,
        build_dataset_manifest(
            bundle=bundle,
            templated=templated,
            sequences=sequences,
            model_summary=model_summary,
            result_paths=result_paths,
        ),
    )
    _write_json(result_paths.metrics_path, model_summary.metrics)
    _write_json(
        result_paths.environment_path,
        build_environment_metadata(
            bundle=bundle,
            result_paths=result_paths,
        ),
    )


def stable_fingerprint(payload: object) -> str:
    """Return a deterministic fingerprint for a JSON-serializable payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8",
    )
    return hashlib.sha256(encoded).hexdigest()


def build_dataset_manifest(
    *,
    bundle: ExperimentBundle,
    templated: TemplatedDataset,
    sequences: SequenceBuilder,
    model_summary: ModelRunSummary,
    result_paths: ResultPaths,
) -> dict[str, object]:
    """Build a provenance manifest for the preprocessed dataset and sequences."""
    sequence_summary = model_summary.sequence_summary
    raw_logs_path = templated.sink.raw_dataset_path.resolve()
    timestamp_min, timestamp_max = templated.sink.timestamp_bounds()
    dataset_fingerprint = stable_fingerprint(serialize_config(bundle.dataset))
    structured_parser_name = _structured_parser_name(bundle)
    split_summary = build_sequence_split_summary(
        sequences,
        sequence_summary=sequence_summary,
    )
    return {
        "run_fingerprint": result_paths.run_fingerprint,
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_variant": bundle.dataset.name,
        "dataset_name": bundle.dataset.dataset_name,
        "source": dataset_source_summary(bundle.dataset, repo_root=bundle.repo_root),
        "structured_parser": structured_parser_name,
        "template_parser": bundle.dataset.template_parser,
        "cache_paths": {
            "data_root": templated.cache_paths.data_root.resolve().as_posix(),
            "cache_root": templated.cache_paths.cache_root.resolve().as_posix(),
        },
        "raw_logs": {
            "path": raw_logs_path.as_posix(),
            "sha256": sha256_for_file(raw_logs_path),
        },
        "structured_rows": templated.sink.count_rows(),
        "timestamp_bounds": {
            "min_unix_ms": timestamp_min,
            "max_unix_ms": timestamp_max,
        },
        "sequence_config": serialize_config(bundle.dataset.sequence),
        "sequence_split_summary": split_summary.as_dict(),
        "sequence_count": sequence_summary.sequence_count,
        "sequence_split_counts": {
            "train": sequence_summary.train_sequence_count,
            "test": sequence_summary.test_sequence_count,
        },
        "label_counts": {
            "train": sequence_summary.train_label_counts,
            "test": sequence_summary.test_label_counts,
        },
        "model_manifest": msgspec.to_builtins(model_summary.model_manifest),
    }


def build_sequence_split_summary(
    sequences: SequenceBuilder,
    *,
    sequence_summary: SequenceSummary,
) -> SequenceSplitSummary:
    """Describe requested versus effective split semantics for one run."""
    return sequences.build_split_summary(
        sequence_count=sequence_summary.sequence_count,
        train_sequence_count=sequence_summary.train_sequence_count,
        train_label_counts=sequence_summary.train_label_counts,
        test_label_counts=sequence_summary.test_label_counts,
    )


def build_environment_metadata(
    *,
    bundle: ExperimentBundle,
    result_paths: ResultPaths,
) -> dict[str, object]:
    """Capture the local environment for reproducibility and provenance."""
    return {
        "recorded_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "run_fingerprint": result_paths.run_fingerprint,
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
        },
    }


def sha256_for_file(path: Path) -> str:
    """Hash a file without loading it all into memory."""
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
