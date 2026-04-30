"""Tests for public `SequenceBuilder` behavior."""

from pathlib import Path

import pytest

from anomalog.parsers.structured.contracts import StructuredLine
from anomalog.parsers.template.dataset import (
    ExtractedParameters,
    LogTemplate,
    UntemplatedText,
)
from anomalog.sequences import (
    EntitySequenceBuilder,
    FixedSequenceBuilder,
    SequenceSplitCounts,
    SequenceSplitSummary,
    SplitLabel,
    TemplateSequence,
    TimeSequenceBuilder,
)
from experiments.models.base import SequenceSummary
from tests.unit.helpers import (
    InMemoryStructuredSink,
    NullStructuredParser,
    structured_line,
)


def _sink(*rows: StructuredLine) -> InMemoryStructuredSink:
    return InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=Path("raw.log"),
        parser=NullStructuredParser(),
        rows=list(rows),
    )


def _upper_template(
    text: UntemplatedText,
) -> tuple[LogTemplate, ExtractedParameters]:
    return text.upper(), []


def _upper_template_with_source_param(
    text: UntemplatedText,
) -> tuple[LogTemplate, ExtractedParameters]:
    return text.upper(), [text]


def test_entity_sequences_fractional_split_counts_all_entities_when_not_filtered() -> (
    None
):
    """Entity splits keep a fixed holdout and withhold the train-pool tail."""
    sink = _sink(
        structured_line(
            line_order=0,
            timestamp_unix_ms=100,
            entity_id="a",
            untemplated_message_text="first",
            anomalous=None,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=200,
            entity_id="b",
            untemplated_message_text="second",
            anomalous=None,
        ),
        structured_line(
            line_order=2,
            timestamp_unix_ms=300,
            entity_id="c",
            untemplated_message_text="third",
            anomalous=None,
        ),
    )

    sequences = list(
        EntitySequenceBuilder(
            sink=sink,
            infer_template=_upper_template,
            label_for_group=lambda entity_id: 1 if entity_id == "c" else 0,
            train_frac=0.5,
        ),
    )

    assert [sequence.sole_entity_id for sequence in sequences] == ["a", "b", "c"]
    assert [sequence.split_label for sequence in sequences] == [
        SplitLabel.TRAIN,
        SplitLabel.IGNORED,
        SplitLabel.TEST,
    ]
    assert [sequence.label for sequence in sequences] == [0, 0, 1]


def test_entity_sequences_use_fixed_test_suffix_and_nested_train_prefixes() -> None:
    """Entity splits should keep one fixed test suffix across train fractions."""
    sink = _sink(
        structured_line(
            line_order=0,
            timestamp_unix_ms=300,
            entity_id="m",
            untemplated_message_text="third",
            anomalous=None,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=100,
            entity_id="z",
            untemplated_message_text="first",
            anomalous=None,
        ),
        structured_line(
            line_order=2,
            timestamp_unix_ms=400,
            entity_id="a",
            untemplated_message_text="fourth",
            anomalous=None,
        ),
        structured_line(
            line_order=3,
            timestamp_unix_ms=200,
            entity_id="b",
            untemplated_message_text="second",
            anomalous=None,
        ),
    )

    builder = EntitySequenceBuilder(
        sink=sink,
        infer_template=_upper_template,
        label_for_group=lambda _: 0,
        train_frac=0.5,
    )

    first_pass = list(builder)
    second_pass = list(builder.with_train_fraction(0.75))

    assert [sequence.sole_entity_id for sequence in first_pass] == [
        "z",
        "b",
        "m",
        "a",
    ]
    assert [sequence.split_label for sequence in first_pass] == [
        SplitLabel.TRAIN,
        SplitLabel.TRAIN,
        SplitLabel.IGNORED,
        SplitLabel.TEST,
    ]
    assert [sequence.sole_entity_id for sequence in second_pass] == [
        "z",
        "b",
        "m",
        "a",
    ]
    assert [sequence.split_label for sequence in second_pass] == [
        SplitLabel.TRAIN,
        SplitLabel.TRAIN,
        SplitLabel.TRAIN,
        SplitLabel.TEST,
    ]
    assert [
        sequence.sole_entity_id
        for sequence in first_pass
        if sequence.split_label is SplitLabel.TRAIN
    ] == [
        "z",
        "b",
    ]
    assert [
        sequence.sole_entity_id
        for sequence in second_pass
        if sequence.split_label is SplitLabel.TRAIN
    ] == [
        "z",
        "b",
        "m",
    ]
    assert [
        sequence.sole_entity_id
        for sequence in first_pass
        if sequence.split_label is SplitLabel.TEST
    ] == [
        "a",
    ]
    assert [
        sequence.sole_entity_id
        for sequence in second_pass
        if sequence.split_label is SplitLabel.TEST
    ] == [
        "a",
    ]


@pytest.mark.allow_no_new_coverage
def test_entity_sequences_fractional_split_counts_only_normals() -> None:
    """Normal-only training still targets the requested overall train fraction."""
    sink = _sink(
        structured_line(
            line_order=0,
            timestamp_unix_ms=100,
            entity_id="a",
            untemplated_message_text="first",
            anomalous=None,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=200,
            entity_id="b",
            untemplated_message_text="second",
            anomalous=None,
        ),
        structured_line(
            line_order=2,
            timestamp_unix_ms=300,
            entity_id="c",
            untemplated_message_text="third",
            anomalous=None,
        ),
    )

    sequences = list(
        EntitySequenceBuilder(
            sink=sink,
            infer_template=_upper_template,
            label_for_group=lambda entity_id: 1 if entity_id == "c" else 0,
            train_frac=0.5,
            train_on_normal_entities_only=True,
        ),
    )

    assert [sequence.sole_entity_id for sequence in sequences] == ["a", "b", "c"]
    assert [sequence.split_label for sequence in sequences] == [
        SplitLabel.TRAIN,
        SplitLabel.IGNORED,
        SplitLabel.TEST,
    ]
    assert [sequence.label for sequence in sequences] == [0, 0, 1]


def test_entity_sequences_error_when_normal_only_target_is_impossible() -> None:
    """Normal-only training should fail if the requested overall split is impossible."""
    sink = _sink(
        structured_line(
            line_order=0,
            timestamp_unix_ms=100,
            entity_id="a",
            untemplated_message_text="first",
            anomalous=None,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=200,
            entity_id="b",
            untemplated_message_text="second",
            anomalous=None,
        ),
        structured_line(
            line_order=2,
            timestamp_unix_ms=300,
            entity_id="c",
            untemplated_message_text="third",
            anomalous=None,
        ),
        structured_line(
            line_order=3,
            timestamp_unix_ms=400,
            entity_id="d",
            untemplated_message_text="fourth",
            anomalous=None,
        ),
    )

    builder = EntitySequenceBuilder(
        sink=sink,
        infer_template=_upper_template,
        label_for_group=lambda entity_id: 0 if entity_id == "c" else 1,
        train_frac=0.8,
        train_on_normal_entities_only=True,
    )

    with pytest.raises(
        ValueError,
        match="Requested train fraction is impossible",
    ):
        list(builder)


@pytest.mark.allow_no_new_coverage
def test_entity_sequences_treat_nonzero_group_labels_as_anomalous() -> None:
    """Any non-zero group label should force the entity into the anomaly path."""
    # This locks down the shared anomaly-semantics contract for grouped labels.
    # Nearby uncovered lines are generic helper plumbing, not this regression.
    sink = _sink(
        structured_line(
            line_order=0,
            timestamp_unix_ms=100,
            entity_id="a",
            untemplated_message_text="first",
            anomalous=None,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=200,
            entity_id="b",
            untemplated_message_text="second",
            anomalous=None,
        ),
    )

    sequences = list(
        EntitySequenceBuilder(
            sink=sink,
            infer_template=_upper_template,
            label_for_group=lambda entity_id: 2 if entity_id == "b" else 0,
            train_frac=0.5,
            train_on_normal_entities_only=True,
        ),
    )

    assert [sequence.sole_entity_id for sequence in sequences] == ["a", "b"]
    assert [sequence.split_label for sequence in sequences] == [
        SplitLabel.TRAIN,
        SplitLabel.TEST,
    ]
    assert [sequence.label for sequence in sequences] == [0, 1]


@pytest.mark.allow_no_new_coverage
def test_entity_sequences_train_fraction_one_uses_all_entities() -> None:
    """Without the normal-only filter, `train_frac=1.0` leaves no test entities."""
    # This keeps the `train_frac=1.0` boundary explicit: all eligible entity
    # sequences must stay in train. Nearby uncovered branches are not about
    # this ceil-based cutoff, so covering them would not replace this check.
    sink = _sink(
        structured_line(
            line_order=0,
            timestamp_unix_ms=100,
            entity_id="a",
            untemplated_message_text="first",
            anomalous=None,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=200,
            entity_id="b",
            untemplated_message_text="second",
            anomalous=None,
        ),
    )

    sequences = list(
        EntitySequenceBuilder(
            sink=sink,
            infer_template=_upper_template,
            label_for_group=lambda entity_id: 1 if entity_id == "b" else 0,
            train_frac=1.0,
            test_frac=0.0,
        ),
    )

    assert [sequence.split_label for sequence in sequences] == [
        SplitLabel.TRAIN,
        SplitLabel.TRAIN,
    ]


def test_fixed_window_sequences_use_inline_line_labels_and_positional_split() -> None:
    """Fixed windows label from rows and split by window position."""
    sink = _sink(
        structured_line(
            line_order=0,
            timestamp_unix_ms=100,
            entity_id="a",
            untemplated_message_text="one",
            anomalous=0,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=130,
            entity_id="a",
            untemplated_message_text="two",
            anomalous=0,
        ),
        structured_line(
            line_order=2,
            timestamp_unix_ms=170,
            entity_id="c",
            untemplated_message_text="three",
            anomalous=1,
        ),
    )
    builder = FixedSequenceBuilder(
        sink=sink,
        infer_template=_upper_template_with_source_param,
        label_for_group=lambda _: 0,
        window_size=2,
        step=2,
        train_frac=0.5,
    )

    sequences = list(builder)
    train_window, test_window = sequences

    assert [sequence.split_label for sequence in sequences] == [
        SplitLabel.TRAIN,
        SplitLabel.TEST,
    ]
    assert train_window.entity_ids == ["a"]
    assert train_window.events == [("ONE", ["one"], None), ("TWO", ["two"], 30)]
    assert train_window.label == 0
    assert test_window.entity_ids == ["c"]
    assert test_window.events == [("THREE", ["three"], None)]
    assert test_window.label == 1


def test_represent_with_yields_model_ready_records() -> None:
    """Sequence builders expose a fluent representation stage."""

    class _EventCountRepresentation:
        name = "event_count"

        def represent(self, sequence: TemplateSequence) -> dict[str, int]:
            assert self.name == "event_count"
            return {"event_count": len(sequence.events)}

    sink = _sink(
        structured_line(
            line_order=0,
            timestamp_unix_ms=100,
            entity_id="a",
            untemplated_message_text="one",
            anomalous=0,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=120,
            entity_id="a",
            untemplated_message_text="one",
            anomalous=0,
        ),
        structured_line(
            line_order=2,
            timestamp_unix_ms=150,
            entity_id="b",
            untemplated_message_text="two",
            anomalous=1,
        ),
    )
    builder = EntitySequenceBuilder(
        sink=sink,
        infer_template=_upper_template,
        label_for_group=lambda entity_id: 1 if entity_id == "b" else 0,
        train_frac=0.5,
    )

    represented = list(
        builder.represent_with(
            _EventCountRepresentation(),
        ),
    )

    assert [sample.window_id for sample in represented] == [0, 1]
    assert [sample.entity_ids for sample in represented] == [["a"], ["b"]]
    assert [sample.split_label for sample in represented] == [
        SplitLabel.TRAIN,
        SplitLabel.TEST,
    ]
    assert [sample.label for sample in represented] == [0, 1]
    assert [sample.data for sample in represented] == [
        {"event_count": 2},
        {"event_count": 1},
    ]


def test_represented_sequences_can_stream_as_river_dataset() -> None:
    """Represented sequences expose River-style `(x, y)` examples."""

    class _EventCountRepresentation:
        name = "event_count"

        def represent(self, sequence: TemplateSequence) -> dict[str, int]:
            assert self.name == "event_count"
            return {"event_count": len(sequence.events)}

    sink = _sink(
        structured_line(
            line_order=0,
            timestamp_unix_ms=100,
            entity_id="a",
            untemplated_message_text="one",
            anomalous=0,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=150,
            entity_id="b",
            untemplated_message_text="two",
            anomalous=1,
        ),
    )
    builder = EntitySequenceBuilder(
        sink=sink,
        infer_template=_upper_template,
        label_for_group=lambda entity_id: 1 if entity_id == "b" else 0,
        train_frac=0.5,
    )

    river_examples = list(
        builder.represent_with(_EventCountRepresentation()).iter_labeled_examples(),
    )

    assert river_examples == [({"event_count": 1}, 0), ({"event_count": 1}, 1)]


@pytest.mark.allow_no_new_coverage
def test_template_sequence_exposes_single_entity_only_when_unambiguous() -> None:
    """Template sequences should expose a single entity only when exact."""
    # This protects the single-entity convenience accessor without encoding a
    # misleading default when a window spans multiple entities.
    single_entity_sequence = TemplateSequence(
        events=[("ONE", ["one"], None)],
        label=0,
        entity_ids=["node-a"],
        window_id=7,
        split_label=SplitLabel.TEST,
    )
    multi_entity_sequence = TemplateSequence(
        events=[("ONE", ["one"], None)],
        label=0,
        entity_ids=["node-a", "node-b"],
        window_id=8,
        split_label=SplitLabel.TEST,
    )

    assert single_entity_sequence.sole_entity_id == "node-a"
    assert multi_entity_sequence.sole_entity_id is None


def test_sequence_builder_base_methods_cover_default_helper_paths() -> None:
    """Base builder helpers should expose their default contracts clearly."""
    updated_train_fraction = 0.25
    expected_sequence_count = 3
    expected_eligible_sequence_count = 1
    builder = FixedSequenceBuilder(
        sink=_sink(),
        infer_template=_upper_template,
        label_for_group=lambda _: 0,
        window_size=1,
    )

    assert (
        builder.with_train_fraction(updated_train_fraction).train_frac
        == updated_train_fraction
    )
    assert (
        builder.train_fraction_eligible_sequence_count(
            sequence_summary=SequenceSummary(
                sequence_count=expected_sequence_count,
                train_sequence_count=1,
                test_sequence_count=2,
                train_label_counts={0: 1},
                test_label_counts={1: 2},
            ),
        )
        == expected_eligible_sequence_count
    )
    assert builder.split_count_hint() == SequenceSplitCounts(
        total_count=0,
        train_count=0,
        ignored_count=0,
        test_count=0,
    )
    assert builder.build_split_summary(
        sequence_summary=SequenceSummary(
            sequence_count=expected_sequence_count,
            train_sequence_count=1,
            test_sequence_count=2,
            train_label_counts={0: 1},
            test_label_counts={1: 2},
        ),
    ) == SequenceSplitSummary(
        requested_train_fraction=0.8,
        train_on_normal_entities_only=None,
        eligible_train_sequence_count=expected_eligible_sequence_count,
        ignored_sequence_count=0,
        effective_train_fraction_of_eligible=1.0,
        effective_train_fraction_overall=0.33333333,
    )


def test_fixed_sequence_builder_can_use_a_fixed_test_suffix() -> None:
    """Fixed grouping should support a shared fixed-holdout split contract."""
    expected_train_sequence_count = 1
    expected_test_sequence_count = 1
    expected_ignored_sequence_count = 2
    expected_sequence_count = 4
    sink = _sink(
        structured_line(
            line_order=0,
            timestamp_unix_ms=100,
            entity_id="a",
            untemplated_message_text="one",
            anomalous=0,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=130,
            entity_id="a",
            untemplated_message_text="two",
            anomalous=0,
        ),
        structured_line(
            line_order=2,
            timestamp_unix_ms=160,
            entity_id="b",
            untemplated_message_text="three",
            anomalous=0,
        ),
        structured_line(
            line_order=3,
            timestamp_unix_ms=190,
            entity_id="b",
            untemplated_message_text="four",
            anomalous=1,
        ),
    )

    builder = FixedSequenceBuilder(
        sink=sink,
        infer_template=_upper_template,
        label_for_group=lambda _: 0,
        window_size=1,
        train_frac=0.25,
        test_frac=0.25,
    )

    sequences = list(builder)

    assert [sequence.split_label for sequence in sequences] == [
        SplitLabel.TRAIN,
        SplitLabel.IGNORED,
        SplitLabel.IGNORED,
        SplitLabel.TEST,
    ]
    assert builder.split_count_hint() == SequenceSplitCounts(
        total_count=expected_sequence_count,
        train_count=expected_train_sequence_count,
        ignored_count=expected_ignored_sequence_count,
        test_count=expected_test_sequence_count,
    )


def test_entity_sequence_builder_entity_specific_helpers_cover_public_contract() -> (
    None
):
    """Entity builder helpers should reflect the normal-only training policy."""
    all_eligible_sequences = 5
    expected_train_pool_sequence_count = all_eligible_sequences - 1
    expected_ignored_sequence_count = 2
    normal_eligible_sequences = 3
    builder = EntitySequenceBuilder(
        sink=_sink(
            structured_line(
                line_order=0,
                timestamp_unix_ms=100,
                entity_id="a",
                untemplated_message_text="first",
                anomalous=None,
            ),
            structured_line(
                line_order=1,
                timestamp_unix_ms=200,
                entity_id="b",
                untemplated_message_text="second",
                anomalous=None,
            ),
            structured_line(
                line_order=2,
                timestamp_unix_ms=300,
                entity_id="c",
                untemplated_message_text="third",
                anomalous=None,
            ),
            structured_line(
                line_order=3,
                timestamp_unix_ms=400,
                entity_id="d",
                untemplated_message_text="fourth",
                anomalous=1,
            ),
            structured_line(
                line_order=4,
                timestamp_unix_ms=500,
                entity_id="e",
                untemplated_message_text="fifth",
                anomalous=None,
            ),
        ),
        infer_template=_upper_template,
        label_for_group=lambda entity_id: 1 if entity_id == "d" else 0,
        train_frac=0.5,
    )

    assert builder.with_train_on_normal_entities_only().train_on_normal_entities_only
    assert (
        builder.with_train_on_normal_entities_only(
            enabled=False,
        ).train_on_normal_entities_only
        is False
    )
    assert (
        builder.train_fraction_eligible_sequence_count(
            sequence_summary=SequenceSummary(
                sequence_count=all_eligible_sequences,
                train_sequence_count=2,
                test_sequence_count=1,
                train_label_counts={0: 2},
                test_label_counts={1: 3},
            ),
        )
        == expected_train_pool_sequence_count
    )
    assert (
        builder.with_train_on_normal_entities_only().train_fraction_eligible_sequence_count(
            sequence_summary=SequenceSummary(
                sequence_count=all_eligible_sequences,
                train_sequence_count=2,
                test_sequence_count=1,
                train_label_counts={0: 2},
                test_label_counts={0: 1, 1: 2},
                ignored_label_counts={0: 1, 1: 1},
                ignored_sequence_count=2,
            ),
        )
        == normal_eligible_sequences
    )
    assert builder.split_count_hint() == SequenceSplitCounts(
        total_count=all_eligible_sequences,
        train_count=2,
        ignored_count=expected_ignored_sequence_count,
        test_count=1,
    )


def test_time_sequence_builder_uses_public_windowing_and_preserves_null_deltas() -> (
    None
):
    """Time grouping should use sink windows and keep null timestamps as null deltas."""
    sink = _sink(
        structured_line(
            line_order=0,
            timestamp_unix_ms=100,
            entity_id="a",
            untemplated_message_text="one",
            anomalous=0,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=None,
            entity_id="a",
            untemplated_message_text="two",
            anomalous=0,
        ),
        structured_line(
            line_order=2,
            timestamp_unix_ms=250,
            entity_id="b",
            untemplated_message_text="three",
            anomalous=1,
        ),
    )

    sequences = list(
        TimeSequenceBuilder(
            sink=sink,
            infer_template=_upper_template_with_source_param,
            label_for_group=lambda entity_id: 1 if entity_id == "b" else 0,
            time_span_ms=500,
            train_frac=1.0,
            test_frac=0.0,
        ),
    )

    assert len(sequences) == 1
    assert sequences[0].split_label is SplitLabel.TRAIN
    assert sequences[0].events == [
        ("ONE", ["one"], None),
        ("TWO", ["two"], None),
        ("THREE", ["three"], 150),
    ]
    assert sequences[0].entity_ids == ["a", "b"]
    assert sequences[0].label == 1


def test_fixed_sequence_builder_uses_single_train_window_when_dataset_is_short() -> (
    None
):
    """Fixed grouping should still emit one train window when rows fit in one window."""
    sink = _sink(
        structured_line(
            line_order=0,
            timestamp_unix_ms=100,
            entity_id="a",
            untemplated_message_text="one",
            anomalous=1,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=130,
            entity_id="a",
            untemplated_message_text="two",
            anomalous=1,
        ),
    )

    sequences = list(
        FixedSequenceBuilder(
            sink=sink,
            infer_template=_upper_template,
            label_for_group=lambda _: 0,
            window_size=5,
            train_frac=0.5,
            test_frac=0.0,
        ),
    )

    assert len(sequences) == 1
    assert sequences[0].split_label is SplitLabel.TRAIN
    assert sequences[0].events == [("ONE", [], None), ("TWO", [], 30)]
    assert sequences[0].label == 1
