# ruff: noqa: PLC2701, PLR2004
"""Tests for DeepLog dataset-audit warm-up accounting helpers."""

from __future__ import annotations

import pytest

from experiments.audit.deeplog_data_audit import (
    HDFSSessionObservation,
    _summarise_hdfs_first_100k_policies,
    aggregate_warmup_accounting,
    warmup_counts_for_sequence_length,
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
    """Per-sequence warm-up accounting should follow min/max DeepLog rules.

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
    """Extra exclusion counts should be reflected in events_seen totals."""
    expected_insufficient = 10
    expected_eligible = 1
    expected_seen = 16
    additional_exclusions = 5
    summary = aggregate_warmup_accounting(
        sequence_lengths=[11],
        history_size=10,
        additional_excluded_events=additional_exclusions,
    )

    assert summary.insufficient_history == expected_insufficient
    assert summary.events_eligible == expected_eligible
    assert summary.events_seen == expected_seen
    assert summary.events_seen == (
        summary.insufficient_history + summary.events_eligible + additional_exclusions
    )


def test_chunk_boundary_warmup_accounting_adds_extra_warmup_loss() -> None:
    """Chunked streams should report the additional warm-up cost explicitly."""
    contiguous = aggregate_warmup_accounting(
        sequence_lengths=[6],
        history_size=2,
    )
    chunked = aggregate_warmup_accounting(
        sequence_lengths=[3, 3],
        history_size=2,
    )

    assert contiguous.insufficient_history == 2
    assert chunked.insufficient_history == 4
    assert chunked.insufficient_history - contiguous.insufficient_history == 2


def test_hdfs_first_100k_policy_summary_distinguishes_straddlers() -> None:
    """The HDFS policy audit should separate partial, first, and last policies."""
    sessions = [
        HDFSSessionObservation(
            entity_id="a",
            first_line_order=0,
            last_line_order=1,
            label=0,
            event_count=2,
            pre_cutoff_event_count=2,
            post_cutoff_event_count=0,
        ),
        HDFSSessionObservation(
            entity_id="b",
            first_line_order=2,
            last_line_order=6,
            label=1,
            event_count=5,
            pre_cutoff_event_count=3,
            post_cutoff_event_count=2,
        ),
        HDFSSessionObservation(
            entity_id="c",
            first_line_order=7,
            last_line_order=8,
            label=0,
            event_count=2,
            pre_cutoff_event_count=0,
            post_cutoff_event_count=2,
        ),
    ]

    summaries = _summarise_hdfs_first_100k_policies(
        sessions=sessions,
        cutoff=5,
        history_size=2,
        template_count=29,
    )
    by_name = {summary.policy_name: summary for summary in summaries}

    assert by_name["split_partial_sequences"].train_normal_sessions == 1
    assert by_name["split_partial_sequences"].ignored_sessions == 1
    assert by_name["split_partial_sequences"].test_anomalous_sessions == 1
    assert by_name["assign_by_first_event"].ignored_sessions == 1
    assert by_name["assign_by_last_event"].test_anomalous_sessions == 1
    assert by_name["normal_complete_sessions"].ignored_sessions == 0
    assert by_name["normal_complete_sessions"].test_anomalous_sessions == 1
