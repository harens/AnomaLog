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
    SplitLabel,
    TemplateSequence,
    TimeSequenceBuilder,
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


@pytest.mark.allow_no_new_coverage
def test_entity_sequences_train_only_normals() -> None:
    """Normal-only training should fail if the overall target cannot be satisfied."""
    # This protects the fail-fast contract for impossible normal-only splits.
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

    builder = EntitySequenceBuilder(
        sink=sink,
        infer_template=_upper_template,
        label_for_group=lambda entity_id: 1 if entity_id == "b" else 0,
        train_frac=1.0,
        train_on_normal_entities_only=True,
    )

    with pytest.raises(
        ValueError,
        match="Requested train fraction is impossible",
    ):
        list(builder)


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
        SplitLabel.TRAIN,
        SplitLabel.TEST,
    ]
    assert [sequence.label for sequence in sequences] == [0, 0, 1]


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
        SplitLabel.TRAIN,
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
    )

    builder = EntitySequenceBuilder(
        sink=sink,
        infer_template=_upper_template,
        label_for_group=lambda entity_id: 1 if entity_id == "c" else 0,
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


@pytest.mark.allow_no_new_coverage
def test_non_entity_builders_do_not_expose_normal_only_training() -> None:
    """Non-entity builders do not expose the entity-only normal-train policy."""
    # This protects the public capability boundary after the builder split.
    # Nearby uncovered lines are internal split-summary helpers, not the API
    # contract this regression test is asserting.
    fixed_builder = FixedSequenceBuilder(
        sink=_sink(),
        infer_template=_upper_template,
        label_for_group=lambda _: 0,
        window_size=1,
    )
    time_builder = TimeSequenceBuilder(
        sink=_sink(),
        infer_template=_upper_template,
        label_for_group=lambda _: 0,
        time_span_ms=1,
    )

    assert not hasattr(fixed_builder, "with_train_on_normal_entities_only")
    assert not hasattr(time_builder, "with_train_on_normal_entities_only")


def test_represent_with_yields_model_ready_records() -> None:
    """Sequence builders expose a fluent representation stage."""

    class _EventCountRepresentation:
        name = "event_count"

        def represent(self, sequence: TemplateSequence) -> dict[str, int]:
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
