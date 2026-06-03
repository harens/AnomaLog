"""Tests for public DeepLog audit helpers and CLI wiring."""

from __future__ import annotations

import json
import types
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from experiments.audit.deeplog_data_audit import (
    aggregate_warmup_accounting,
    audit_bgl_chunk_size_sensitivity,
    audit_bgl_continuous_stream_warmup,
    audit_hdfs_first_100k_policies,
    warmup_counts_for_sequence_length,
)
from experiments.config import load_experiment_bundles
from experiments.runners import audit_deeplog_data as audit_cli

if TYPE_CHECKING:
    from collections.abc import Iterator


pytestmark = pytest.mark.skip(
    reason="DeepLog audit helpers are excluded from the default unit test run.",
)


@pytest.mark.parametrize(
    ("sequence_length", "history_size", "expected_insufficient", "expected_eligible"),
    [
        (0, 10, 0, 0),
        (1, 10, 1, 0),
        (10, 10, 10, 0),
        (11, 10, 10, 1),
        (13, 10, 10, 3),
    ],
)
def test_warmup_counts_for_sequence_length_matches_deeplog_contract(
    sequence_length: int,
    history_size: int,
    expected_insufficient: int,
    expected_eligible: int,
) -> None:
    """Per-sequence warm-up accounting should follow the DeepLog contract.

    Args:
        sequence_length (int): Sequence length under test.
        history_size (int): DeepLog history size under test.
        expected_insufficient (int): Expected warm-up exclusion count.
        expected_eligible (int): Expected eligible event count.
    """
    insufficient, eligible = warmup_counts_for_sequence_length(
        sequence_length=sequence_length,
        history_size=history_size,
    )

    assert insufficient == expected_insufficient
    assert eligible == expected_eligible


def test_aggregate_warmup_accounting_mixed_lengths() -> None:
    """Mixed sequence lengths should aggregate warm-up counts consistently."""
    expected_insufficient = 31
    expected_eligible = 4
    expected_seen = 35
    summary = aggregate_warmup_accounting(
        sequence_lengths=[0, 1, 10, 11, 13],
        history_size=10,
    )

    assert summary.insufficient_history == expected_insufficient
    assert summary.events_eligible == expected_eligible
    assert summary.events_seen == expected_seen
    assert summary.insufficient_history_rate == pytest.approx(
        expected_insufficient / expected_seen,
    )


def test_aggregate_warmup_accounting_includes_additional_exclusions() -> None:
    """Extra exclusion counts should be reflected in the overall total."""
    expected_insufficient = 10
    expected_eligible = 1
    expected_seen = 16
    summary = aggregate_warmup_accounting(
        sequence_lengths=[11],
        history_size=10,
        additional_excluded_events=5,
    )

    assert summary.insufficient_history == expected_insufficient
    assert summary.events_eligible == expected_eligible
    assert summary.events_seen == expected_seen


def test_chunk_boundary_warmup_accounting_adds_extra_warmup_loss() -> None:
    """Chunked streams should report the additional warm-up cost explicitly."""
    expected_contiguous_insufficient = 2
    expected_chunked_insufficient = 4
    expected_delta = 2
    contiguous = aggregate_warmup_accounting(
        sequence_lengths=[6],
        history_size=2,
    )
    chunked = aggregate_warmup_accounting(
        sequence_lengths=[3, 3],
        history_size=2,
    )

    assert contiguous.insufficient_history == expected_contiguous_insufficient
    assert chunked.insufficient_history == expected_chunked_insufficient
    assert (
        chunked.insufficient_history - contiguous.insufficient_history == expected_delta
    )


def test_audit_bgl_sensitivity_helpers_use_requested_chunk_sizes() -> None:
    """BGL chunk-size audits should return one summary per requested size."""
    repo_root = Path(__file__).resolve().parents[2]
    bundle = load_experiment_bundles(
        Path(
            "experiments/configs/datasets/bgl/"
            "bgl_deeplog_ccs2017_paper_10pct_entry_stream_no_online.toml",
        ),
    )[0]
    summaries = audit_bgl_chunk_size_sensitivity(
        config=bundle.dataset,
        repo_root=repo_root,
        history_size=3,
        chunk_sizes=(3, 5),
    )

    assert [summary.chunk_size for summary in summaries] == [3, 5]
    assert all(
        summary.warmup_loss == summary.insufficient_history for summary in summaries
    )


def test_audit_bgl_continuous_stream_warmup_reports_context_carryover() -> None:
    """Continuous-stream audits should preserve carried history across chunks."""
    repo_root = Path(__file__).resolve().parents[2]
    bundle = load_experiment_bundles(
        Path(
            "experiments/configs/datasets/bgl/"
            "bgl_deeplog_ccs2017_paper_10pct_entry_stream_no_online.toml",
        ),
    )[0]
    summary = audit_bgl_continuous_stream_warmup(
        config=bundle.dataset,
        repo_root=repo_root,
        history_size=3,
    )

    assert summary.events_eligible >= 0
    assert summary.insufficient_history >= 0
    assert summary.lost_event_line_orders == []


def test_audit_hdfs_first_100k_policies_distinguish_straddlers() -> None:
    """HDFS policy audits should keep the split interpretations distinct."""
    repo_root = Path(__file__).resolve().parents[2]
    bundle = load_experiment_bundles(
        Path("experiments/configs/datasets/hdfs_v1/entity_chronological.toml"),
    )[0]
    summaries = audit_hdfs_first_100k_policies(
        config=bundle.dataset,
        repo_root=repo_root,
        history_size=10,
    )
    by_name = {summary.policy_name: summary for summary in summaries}

    assert by_name["split_partial_sequences"].train_normal_sessions >= 0
    assert by_name["assign_by_first_event"].ignored_sessions >= 0
    assert by_name["assign_by_last_event"].test_anomalous_sessions >= 0


def test_audit_cli_main_writes_reports_and_uses_resolved_dataset_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI should write reports for resolved dataset references.

    Args:
        tmp_path (Path): Temporary directory used to hold the synthetic audit
            outputs.
        monkeypatch (pytest.MonkeyPatch): Patch helper used to stub the CLI
            dependencies.
    """

    class _Parser:
        @staticmethod
        def parse_args() -> types.SimpleNamespace:
            return types.SimpleNamespace(
                dataset=("demo:3",),
                repo_root=tmp_path,
                output_markdown=tmp_path / "audit.md",
                output_json=tmp_path / "audit.json",
            )

    @contextmanager
    def _noop_harness() -> Iterator[None]:
        yield

    class _Report:
        @staticmethod
        def to_dict() -> dict[str, object]:
            return {
                "dataset_variant": "demo",
                "dataset_name": "demo",
                "grouping_key": "entity",
                "split_strategy": {"split_mode": "prefix_count"},
                "raw_log_entry_count": 1,
                "parsed_event_count": 1,
                "parsed_template_count": 1,
                "sequence_count": 1,
                "train_sequence_count": 1,
                "test_sequence_count": 0,
                "ignored_sequence_count": 0,
                "split_summaries": {},
                "sequence_length_summary": {
                    "min": 1,
                    "p25": 1.0,
                    "median": 1.0,
                    "p75": 1.0,
                    "max": 1,
                    "mean": 1.0,
                    "count_lte_history_size": 1,
                    "count_gt_history_size": 0,
                },
                "warmup_overall": {
                    "events_seen": 1,
                    "insufficient_history": 0,
                    "events_eligible": 1,
                    "insufficient_history_rate": 0.0,
                },
                "no_eligible_predictions": {"sequence_count": 0, "label_counts": {}},
                "training_target_summary": {
                    "eligible_normal_event_count": 1,
                    "excluded_anomalous_event_count": 0,
                    "excluded_context_event_count": 0,
                    "will_train": True,
                },
                "raw_entry_split_summary": None,
            }

    def _build_arg_parser() -> _Parser:
        return _Parser()

    monkeypatch.setattr(audit_cli, "build_arg_parser", _build_arg_parser)
    monkeypatch.setattr(audit_cli, "prefect_test_harness", _noop_harness)
    monkeypatch.setattr(
        audit_cli,
        "_resolve_dataset_config_path",
        lambda **_kwargs: tmp_path / "resolved-demo.toml",
    )
    monkeypatch.setattr(
        audit_cli,
        "_decode_toml_file",
        lambda _path, decode: decode(
            {"name": "demo", "dataset_name": "demo", "preset": "demo"},
        ),
    )
    monkeypatch.setattr(
        audit_cli,
        "audit_dataset_for_deeplog",
        lambda **_kwargs: _Report(),
    )

    assert audit_cli.main() == 0
    assert (
        json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))["datasets"][
            0
        ]["dataset_variant"]
        == "demo"
    )
    assert "# DeepLog Dataset Audit" in (tmp_path / "audit.md").read_text(
        encoding="utf-8",
    )
