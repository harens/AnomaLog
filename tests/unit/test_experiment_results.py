"""Tests for experiment result manifest helpers."""

from pathlib import Path

import pytest

from anomalog.sequences import EntitySequenceBuilder
from experiments.models.base import SequenceSummary
from experiments.results import build_sequence_split_summary
from tests.unit.helpers import InMemoryStructuredSink, NullStructuredParser


@pytest.mark.allow_no_new_coverage
def test_build_sequence_split_summary_exposes_effective_fraction_for_normal_only() -> (
    None
):
    """Normal-only split summaries should show requested and effective fractions."""
    # This protects experiment-layer manifest metadata outside the configured
    # `anomalog` coverage target.
    expected_requested_train_fraction = 0.5
    expected_sequence_count = 10
    expected_train_sequence_count = 5
    expected_test_sequence_count = 3
    expected_ignored_sequence_count = 2
    expected_eligible_train_sequence_count = 6
    expected_effective_eligible_train_fraction = round(
        expected_train_sequence_count / expected_eligible_train_sequence_count,
        8,
    )
    expected_effective_overall_train_fraction = round(
        expected_train_sequence_count
        / (expected_train_sequence_count + expected_test_sequence_count),
        8,
    )
    summary = build_sequence_split_summary(
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
