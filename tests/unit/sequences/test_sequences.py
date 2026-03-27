"""Tests for public `SequenceBuilder` behavior."""

from pathlib import Path

import pytest

from anomalog.sequences import GroupingMode, SequenceBuilder, SplitLabel
from anomalog.structured_parsers.contracts import StructuredLine
from anomalog.template_parsers.templated_dataset import (
    ExtractedParameters,
    LogTemplate,
    UntemplatedText,
)
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


def test_entity_sequences_train_only_normals() -> None:
    """Group labels alone can keep anomalous entities out of the training split."""
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

    builder = SequenceBuilder(
        sink=sink,
        infer_template=_upper_template,
        label_for_group=lambda entity_id: 1 if entity_id == "b" else 0,
        mode=GroupingMode.ENTITY,
        train_frac=1.0,
        train_on_normal_entities_only=True,
    )

    sequences = list(builder)
    entity_counts = sink.count_entities_by_label(builder.label_for_group)

    assert entity_counts.normal_entities == 1
    assert entity_counts.total_entities == len(sequences)
    assert [sequence.entity_id for sequence in sequences] == ["a", "b"]
    assert [sequence.split_label for sequence in sequences] == [
        SplitLabel.TRAIN,
        SplitLabel.TEST,
    ]
    assert [sequence.label for sequence in sequences] == [0, 1]


def test_entity_sequences_fractional_split_counts_all_entities_when_not_filtered() -> (
    None
):
    """Entity splits use entity position when all entities are eligible for train."""
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
        SequenceBuilder(
            sink=sink,
            infer_template=_upper_template,
            label_for_group=lambda entity_id: 1 if entity_id == "c" else 0,
            mode=GroupingMode.ENTITY,
            train_frac=0.5,
        ),
    )

    assert [sequence.entity_id for sequence in sequences] == ["a", "b", "c"]
    assert [sequence.split_label for sequence in sequences] == [
        SplitLabel.TRAIN,
        SplitLabel.TRAIN,
        SplitLabel.TEST,
    ]
    assert [sequence.label for sequence in sequences] == [0, 0, 1]


@pytest.mark.allow_no_new_coverage
def test_entity_sequences_fractional_split_counts_only_normals() -> None:
    """Normal-only training uses the number of normal entities as the train target."""
    # This guards the regression where `train_frac` must be applied to the
    # count of normal entities, not all entities. The nearby uncovered branches
    # are unrelated iterator paths and would not express this split-policy rule.
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
        SequenceBuilder(
            sink=sink,
            infer_template=_upper_template,
            label_for_group=lambda entity_id: 1 if entity_id == "c" else 0,
            mode=GroupingMode.ENTITY,
            train_frac=0.5,
            train_on_normal_entities_only=True,
        ),
    )

    assert [sequence.entity_id for sequence in sequences] == ["a", "b", "c"]
    assert [sequence.split_label for sequence in sequences] == [
        SplitLabel.TRAIN,
        SplitLabel.TEST,
        SplitLabel.TEST,
    ]
    assert [sequence.label for sequence in sequences] == [0, 0, 1]


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
        SequenceBuilder(
            sink=sink,
            infer_template=_upper_template,
            label_for_group=lambda entity_id: 1 if entity_id == "b" else 0,
            mode=GroupingMode.ENTITY,
            train_frac=1.0,
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
    builder = SequenceBuilder(
        sink=sink,
        infer_template=_upper_template_with_source_param,
        label_for_group=lambda _: 0,
        mode=GroupingMode.FIXED,
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
    assert train_window.counts == {"ONE": 1, "TWO": 1}
    assert train_window.label == 0
    assert test_window.entity_ids == ["c"]
    assert test_window.events == [("THREE", ["three"], None)]
    assert test_window.label == 1
