"""CLI tool to audit dataset/split readiness for DeepLog paper reproduction."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from prefect.testing.utilities import prefect_test_harness

from experiments.audit import audit_dataset_for_deeplog
from experiments.config_loader import (
    _decode_dataset_config,
    _decode_toml_file,
    _resolve_named_config,
)

_DEFAULT_DATASETS = (
    "hdfs_v1_entity_supervised:10",
    "bgl_entity:3",
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _parse_dataset_item(value: str) -> tuple[str, int]:
    dataset_ref, separator, history_size_raw = value.partition(":")
    if not separator:
        msg = (
            "dataset items must use '<dataset-config-ref>:<history_size>', "
            f"got {value!r}."
        )
        raise ValueError(msg)
    dataset_ref = dataset_ref.strip()
    if not dataset_ref:
        msg = f"dataset config reference is empty in item {value!r}."
        raise ValueError(msg)
    try:
        history_size = int(history_size_raw)
    except ValueError as exc:
        msg = f"history_size must be an integer in item {value!r}."
        raise ValueError(msg) from exc
    if history_size < 0:
        msg = f"history_size must be non-negative in item {value!r}."
        raise ValueError(msg)
    return dataset_ref, history_size


def _resolve_dataset_config_path(*, experiments_root: Path, dataset_ref: str) -> Path:
    candidate = Path(dataset_ref)
    if candidate.suffix == ".toml" or "/" in dataset_ref:
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (experiments_root.parent / candidate).resolve()
        if not resolved.exists():
            msg = f"Dataset config file not found: {resolved}"
            raise FileNotFoundError(msg)
        return resolved
    return _resolve_named_config(experiments_root / "configs" / "datasets", dataset_ref)


def _render_markdown(
    *,
    payload: Mapping[str, object],
) -> str:
    lines = [
        "# DeepLog Dataset Audit",
        "",
        f"- generated_at_utc: {payload['generated_at_utc']}",
        "",
    ]
    dataset_reports = _require_object_list(payload["datasets"])
    for raw_report in dataset_reports:
        report = _require_object_dict(raw_report)
        lines.extend(
            [
                f"## {report['dataset_variant']}",
                "",
                f"- dataset_name: {report['dataset_name']}",
                f"- grouping_key: {report['grouping_key']}",
                (
                    "- split_strategy: "
                    f"{json.dumps(report['split_strategy'], sort_keys=True)}"
                ),
                f"- raw_log_entry_count: {report['raw_log_entry_count']}",
                f"- parsed_event_count: {report['parsed_event_count']}",
                f"- parsed_template_count: {report['parsed_template_count']}",
                f"- sequence_count: {report['sequence_count']}",
                f"- train_sequence_count: {report['train_sequence_count']}",
                f"- test_sequence_count: {report['test_sequence_count']}",
                f"- ignored_sequence_count: {report['ignored_sequence_count']}",
                "",
                "### Split/Event Counts",
                "",
                (
                    "| split | sequences | events | normal_sequences | "
                    "anomalous_sequences |"
                ),
                "| --- | ---: | ---: | ---: | ---: |",
            ],
        )
        split_summaries = _require_object_dict(report["split_summaries"])
        for split in ("train", "ignored", "test"):
            split_summary_raw = split_summaries.get(split)
            if not isinstance(split_summary_raw, dict):
                continue
            split_summary = _require_object_dict(split_summary_raw)
            lines.append(
                "| "
                f"{split} | {split_summary['sequence_count']} | "
                f"{split_summary['event_count']} | "
                f"{split_summary['normal_sequence_count']} | "
                f"{split_summary['anomalous_sequence_count']} |",
            )
        sequence_length_summary = _require_object_dict(
            report["sequence_length_summary"],
        )
        warmup = _require_object_dict(report["warmup_overall"])
        no_eligible = _require_object_dict(report["no_eligible_predictions"])
        training_targets = _require_object_dict(report["training_target_summary"])
        raw_entry_split_summary = report.get("raw_entry_split_summary")
        lines.extend(
            [
                "",
                "### Sequence Length Summary",
                "",
                f"- min: {sequence_length_summary['min']}",
                f"- p25: {sequence_length_summary['p25']}",
                f"- median: {sequence_length_summary['median']}",
                f"- p75: {sequence_length_summary['p75']}",
                f"- max: {sequence_length_summary['max']}",
                f"- mean: {sequence_length_summary['mean']}",
                (
                    "- count_lte_history_size: "
                    f"{sequence_length_summary['count_lte_history_size']}"
                ),
                (
                    "- count_gt_history_size: "
                    f"{sequence_length_summary['count_gt_history_size']}"
                ),
                "",
                "### DeepLog Warm-up",
                "",
                f"- events_seen: {warmup['events_seen']}",
                f"- insufficient_history: {warmup['insufficient_history']}",
                f"- events_eligible: {warmup['events_eligible']}",
                f"- insufficient_history_rate: {warmup['insufficient_history_rate']}",
                "",
                "### No Eligible Predictions",
                "",
                f"- sequence_count: {no_eligible['sequence_count']}",
                f"- label_counts: {no_eligible['label_counts']}",
                "",
                "### Training Targets",
                "",
                (
                    "- eligible_normal_event_count: "
                    f"{training_targets['eligible_normal_event_count']}"
                ),
                (
                    "- excluded_anomalous_event_count: "
                    f"{training_targets['excluded_anomalous_event_count']}"
                ),
                (
                    "- excluded_context_event_count: "
                    f"{training_targets['excluded_context_event_count']}"
                ),
                f"- will_train: {training_targets['will_train']}",
                "",
            ],
        )
        if isinstance(raw_entry_split_summary, dict):
            raw_entry_summary = _require_object_dict(raw_entry_split_summary)
            lines.extend(
                [
                    "### Raw Entry Split",
                    "",
                    f"- split_mode: {raw_entry_summary['split_mode']}",
                    f"- application_order: {raw_entry_summary['application_order']}",
                    f"- cutoff_entry_index: {raw_entry_summary['cutoff_entry_index']}",
                    (
                        "- train_raw_entry_count: "
                        f"{raw_entry_summary['train_raw_entry_count']}"
                    ),
                    (
                        "- train_normal_entry_count: "
                        f"{raw_entry_summary['train_normal_entry_count']}"
                    ),
                    (
                        "- train_anomalous_entry_count: "
                        f"{raw_entry_summary['train_anomalous_entry_count']}"
                    ),
                    (
                        "- test_raw_entry_count: "
                        f"{raw_entry_summary['test_raw_entry_count']}"
                    ),
                    (
                        "- test_normal_entry_count: "
                        f"{raw_entry_summary['test_normal_entry_count']}"
                    ),
                    (
                        "- test_anomalous_entry_count: "
                        f"{raw_entry_summary['test_anomalous_entry_count']}"
                    ),
                    (
                        "- ignored_raw_entry_count: "
                        f"{raw_entry_summary['ignored_raw_entry_count']}"
                    ),
                    (
                        "- ignored_normal_entry_count: "
                        f"{raw_entry_summary['ignored_normal_entry_count']}"
                    ),
                    (
                        "- ignored_anomalous_entry_count: "
                        f"{raw_entry_summary['ignored_anomalous_entry_count']}"
                    ),
                    (f"- chunk_count: {report['sequence_count']}"),
                    (
                        "- straddling_group_count: "
                        f"{raw_entry_summary['straddling_group_count']}"
                    ),
                    (
                        "- straddling_group_policy: "
                        f"{raw_entry_summary['straddling_group_policy']}"
                    ),
                    "",
                ],
            )
    return "\n".join(lines).rstrip() + "\n"


def _require_object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"Expected dict payload, got {type(value).__name__}."
        raise TypeError(msg)
    return {str(key): item for key, item in value.items()}


def _require_object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        msg = f"Expected list payload, got {type(value).__name__}."
        raise TypeError(msg)
    return list(value)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        argparse.ArgumentParser: Parsed CLI configuration for the audit tool.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        help=(
            "Dataset config to audit in '<dataset-config-ref>:<history_size>' "
            "format. Defaults to HDFS and BGL DeepLog datasets."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("experiments/results/deeplog_paper_audit.json"),
        help="Where to write the JSON audit payload.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=Path("experiments/results/deeplog_paper_audit.md"),
        help="Where to write a Markdown audit summary.",
    )
    return parser


def main() -> int:
    """Run the audit CLI.

    Returns:
        int: Process exit code.
    """
    parser = build_arg_parser()
    args = parser.parse_args()
    dataset_items = args.dataset or list(_DEFAULT_DATASETS)
    parsed_items = [_parse_dataset_item(item) for item in dataset_items]

    # The audit is an offline local pass; avoid API log-stream retries.
    os.environ.setdefault("PREFECT_LOGGING_TO_API_ENABLED", "false")

    repo_root = Path(__file__).resolve().parents[2]
    experiments_root = repo_root / "experiments"
    reports: list[dict[str, object]] = []
    with prefect_test_harness():
        for dataset_ref, history_size in parsed_items:
            dataset_config_path = _resolve_dataset_config_path(
                experiments_root=experiments_root,
                dataset_ref=dataset_ref,
            )
            dataset_config = _decode_toml_file(
                dataset_config_path,
                decode=_decode_dataset_config,
            )
            report = audit_dataset_for_deeplog(
                config=dataset_config,
                repo_root=repo_root,
                history_size=history_size,
            )
            report_payload = report.to_dict()
            report_payload["dataset_config_path"] = dataset_config_path.as_posix()
            report_payload["history_size"] = history_size
            reports.append(report_payload)

    payload: dict[str, object] = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "repo_root": repo_root.as_posix(),
        "datasets": reports,
    }
    output_json_path: Path = args.output_json
    output_markdown_path: Path = args.output_markdown
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_markdown_path.write_text(
        _render_markdown(payload=payload),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
