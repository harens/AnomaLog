"""Edge-case regressions for the sequence builder helper branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from typing_extensions import override

from anomalog.parsers.structured.parsers import ThunderbirdParser
from anomalog.sequences import (
    EntitySequenceBuilder,
    FixedSequenceBuilder,
    FixedWindowBasis,
    RawEntrySplitMode,
    SequenceSplitCounts,
    SplitApplicationOrder,
    SplitLabel,
    StraddlingGroupPolicy,
)
from tests.unit.helpers import (
    InMemoryStructuredSink,
    NullStructuredParser,
    structured_line,
)
from tests.unit.sequences.test_sequences import _sink, _upper_template

if TYPE_CHECKING:
    from collections.abc import Collection, Iterator
    from pathlib import Path

    from anomalog.parsers.structured.contracts import StructuredLine


@dataclass(frozen=True)
class _EmptyThenFullGroupsBuilder(EntitySequenceBuilder):
    """Entity builder variant that exposes an empty grouped collection first."""

    @override
    def iter_grouped_rows(self) -> Iterator[Collection[StructuredLine]]:
        yield []
        yield [
            structured_line(
                line_order=0,
                timestamp_unix_ms=100,
                entity_id="a",
                untemplated_message_text="first",
                anomalous=0,
            ),
        ]


def test_before_grouping_split_counts_cover_empty_and_empty_group_cases() -> None:
    """Before-grouping split counts should handle empty and skipped groups."""
    empty_builder = EntitySequenceBuilder(
        sink=_sink(),
        infer_template=_upper_template,
        label_for_group=lambda _: 0,
        split_mode=RawEntrySplitMode.PREFIX_COUNT,
        split_application_order=SplitApplicationOrder.BEFORE_GROUPING,
        straddling_group_policy=StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES,
        train_entry_count=1,
    )
    assert empty_builder.split_count_hint() == SequenceSplitCounts(
        total_count=0,
        train_count=0,
        ignored_count=0,
        test_count=0,
    )

    grouped_builder = _EmptyThenFullGroupsBuilder(
        sink=_sink(
            structured_line(
                line_order=0,
                timestamp_unix_ms=100,
                entity_id="a",
                untemplated_message_text="first",
                anomalous=0,
            ),
        ),
        infer_template=_upper_template,
        label_for_group=lambda _: 0,
        split_mode=RawEntrySplitMode.PREFIX_COUNT,
        split_application_order=SplitApplicationOrder.BEFORE_GROUPING,
        straddling_group_policy=StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES,
        train_entry_count=1,
    )
    assert grouped_builder.split_count_hint() == SequenceSplitCounts(
        total_count=1,
        train_count=1,
        ignored_count=0,
        test_count=0,
    )


def test_before_grouping_group_counts_cover_ignored_and_empty_segments() -> None:
    """Before-grouping helper counts should handle ignored and empty segments."""
    rows = [
        structured_line(
            line_order=0,
            timestamp_unix_ms=100,
            entity_id="a",
            untemplated_message_text="first",
            anomalous=0,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=200,
            entity_id="a",
            untemplated_message_text="second",
            anomalous=0,
        ),
        structured_line(
            line_order=2,
            timestamp_unix_ms=300,
            entity_id="a",
            untemplated_message_text="third",
            anomalous=0,
        ),
    ]

    partial_builder = EntitySequenceBuilder(
        sink=_sink(*rows),
        infer_template=_upper_template,
        label_for_group=lambda _: 1,
        split_mode=RawEntrySplitMode.PREFIX_COUNT,
        split_application_order=SplitApplicationOrder.BEFORE_GROUPING,
        straddling_group_policy=StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES,
        train_entry_count=2,
        train_on_normal_entities_only=True,
    )
    assert partial_builder.split_count_hint() == SequenceSplitCounts(
        total_count=2,
        train_count=0,
        ignored_count=1,
        test_count=1,
    )

    resolved_builder = EntitySequenceBuilder(
        sink=_sink(*rows),
        infer_template=_upper_template,
        label_for_group=lambda _: 1,
        split_mode=RawEntrySplitMode.PREFIX_COUNT,
        split_application_order=SplitApplicationOrder.BEFORE_GROUPING,
        straddling_group_policy=StraddlingGroupPolicy.ASSIGN_BY_FIRST_EVENT,
        train_entry_count=2,
        train_on_normal_entities_only=True,
    )
    assert resolved_builder.split_count_hint() == SequenceSplitCounts(
        total_count=1,
        train_count=0,
        ignored_count=1,
        test_count=0,
    )


def test_before_grouping_training_replay_handles_cutoff_and_prefix_entities() -> None:
    """Raw-entry training replay should respect empty cut-offs and prefix entities."""
    sink = _sink(
        structured_line(
            line_order=0,
            timestamp_unix_ms=100,
            entity_id="a",
            untemplated_message_text="first-a",
            anomalous=None,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=200,
            entity_id="b",
            untemplated_message_text="first-b",
            anomalous=None,
        ),
        structured_line(
            line_order=2,
            timestamp_unix_ms=300,
            entity_id="a",
            untemplated_message_text="second-a",
            anomalous=None,
        ),
        structured_line(
            line_order=3,
            timestamp_unix_ms=400,
            entity_id="c",
            untemplated_message_text="first-c",
            anomalous=None,
        ),
    )

    zero_cutoff_builder = EntitySequenceBuilder(
        sink=sink,
        infer_template=_upper_template,
        label_for_group=lambda _: 0,
        split_mode=RawEntrySplitMode.PREFIX_COUNT,
        split_application_order=SplitApplicationOrder.BEFORE_GROUPING,
        straddling_group_policy=StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES,
        train_entry_count=0,
        train_frac=0.5,
        test_frac=0.5,
    )
    assert list(zero_cutoff_builder.iter_training_sequences()) == []

    normal_only_builder = EntitySequenceBuilder(
        sink=sink,
        infer_template=_upper_template,
        label_for_group=lambda entity_id: 1 if entity_id == "a" else 0,
        split_mode=RawEntrySplitMode.PREFIX_COUNT,
        split_application_order=SplitApplicationOrder.BEFORE_GROUPING,
        straddling_group_policy=StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES,
        train_entry_count=2,
        train_frac=0.5,
        test_frac=0.5,
        train_on_normal_entities_only=True,
    )
    normal_only_sequences = list(normal_only_builder.iter_training_sequences())
    assert [sequence.sole_entity_id for sequence in normal_only_sequences] == [
        "b",
    ]
    assert [sequence.split_label for sequence in normal_only_sequences] == [
        SplitLabel.TRAIN,
    ]

    selected_builder = EntitySequenceBuilder(
        sink=sink,
        infer_template=_upper_template,
        label_for_group=lambda entity_id: 1 if entity_id == "b" else 0,
        split_mode=RawEntrySplitMode.PREFIX_COUNT,
        split_application_order=SplitApplicationOrder.BEFORE_GROUPING,
        straddling_group_policy=StraddlingGroupPolicy.ASSIGN_BY_FIRST_EVENT,
        train_entry_count=3,
        train_frac=0.5,
        test_frac=0.5,
    )
    sequences = list(selected_builder.iter_training_sequences())
    assert [sequence.sole_entity_id for sequence in sequences] == ["a", "b"]
    assert [sequence.split_label for sequence in sequences] == [
        SplitLabel.TRAIN,
        SplitLabel.TRAIN,
    ]


def test_before_grouping_test_replay_skips_the_train_prefix() -> None:
    """Raw-entry test replay should resume after the train prefix only once."""
    sink = _sink(
        structured_line(
            line_order=0,
            timestamp_unix_ms=100,
            entity_id="a",
            untemplated_message_text="first-a",
            anomalous=0,
        ),
        structured_line(
            line_order=1,
            timestamp_unix_ms=200,
            entity_id="a",
            untemplated_message_text="second-a",
            anomalous=0,
        ),
        structured_line(
            line_order=2,
            timestamp_unix_ms=300,
            entity_id="a",
            untemplated_message_text="third-a",
            anomalous=0,
        ),
        structured_line(
            line_order=3,
            timestamp_unix_ms=400,
            entity_id="a",
            untemplated_message_text="fourth-a",
            anomalous=0,
        ),
    )
    builder = EntitySequenceBuilder(
        sink=sink,
        infer_template=_upper_template,
        label_for_group=lambda _: 0,
        split_mode=RawEntrySplitMode.PREFIX_COUNT,
        split_application_order=SplitApplicationOrder.BEFORE_GROUPING,
        straddling_group_policy=StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES,
        train_entry_count=2,
        train_frac=0.5,
        test_frac=0.5,
    )

    train_sequences = list(builder.iter_training_sequences())
    test_sequences = list(builder.iter_test_sequences())

    assert [sequence.window_id for sequence in train_sequences] == [0]
    assert [sequence.window_id for sequence in test_sequences] == [1]
    assert [sequence.split_label for sequence in test_sequences] == [
        SplitLabel.TEST,
    ]


def test_fixed_sequence_builder_raw_position_mode_handles_contracts_and_gaps(
    tmp_path: Path,
) -> None:
    """Raw-position windows should validate contracts and preserve ignored gaps.

    Args:
        tmp_path (Path): Temporary directory used to stage the synthetic
            Thunderbird fixture.
    """
    raw_lines = [
        "- 1131566461 2005.11.09 dn228 Nov 9 12:01:01 dn228/dn228 normal-1",
        "- 1131566462 2005.11.09 dn228 Nov 9 12:01:02 dn228/dn228 normal-2",
        "- 1131566463 2005.11.09 dn228 Nov 9 12:01:03 dn228/dn228 normal-3",
        "- 1131566464 2005.11.09 dn228 Nov 9 12:01:04 dn228/dn228 normal-4",
        "- 1131566465 2005.11.09 dn228 Nov 9 12:01:05 dn228/dn228 normal-5",
        "- 1131566466 2005.11.09 dn228 Nov 9 12:01:06 dn228/dn228 normal-6",
        "- 1131566467 2005.11.09 dn228 Nov 9 12:01:07 dn228/dn228 normal-7",
        "- 1131566468 2005.11.09 dn228 Nov 9 12:01:08 dn228/dn228 normal-8",
        "- 1131566469 2005.11.09 dn228 Nov 9 12:01:09 dn228/dn228 normal-9",
        "- 1131566470 2005.11.09 dn228 Nov 9 12:01:10 dn228/dn228 normal-10",
        "- 1131566471 2005.11.09 dn228 Nov 9 12:01:11 dn228/dn228 normal-11",
        "- 1131566472 2005.11.09 dn228 Nov 9 12:01:12 dn228/dn228 normal-12",
        "- 1131566473 2005.11.09 dn228 Nov 9 12:01:13 dn228/dn228 normal-13",
        "- 1131566474 2005.11.09 dn228 Nov 9 12:01:14 dn228/dn228 normal-14",
        "- 1131566475 2005.11.09 dn228 Nov 9 12:01:15 dn228/dn228 normal-15",
        "- 1131566476 2005.11.09 dn228 Nov 9 12:01:16 dn228/dn228 normal-16",
    ]
    raw_path = tmp_path / "Thunderbird.log"
    raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    parser = ThunderbirdParser()
    retained_rows = []
    for line_order, raw_line in enumerate(raw_lines):
        parsed = parser.parse_line(raw_line)
        assert parsed is not None
        retained_rows.append(
            structured_line(
                line_order=line_order,
                timestamp_unix_ms=parsed.timestamp_unix_ms,
                entity_id=parsed.entity_id,
                untemplated_message_text=parsed.untemplated_message_text,
                anomalous=parsed.anomalous,
            ),
        )

    with pytest.raises(
        ValueError,
        match="raw-position fixed windows require step",
    ):
        FixedSequenceBuilder(
            sink=InMemoryStructuredSink(
                dataset_name="thunderbird",
                raw_dataset_path=raw_path,
                parser=parser,
                rows=retained_rows,
            ),
            infer_template=_upper_template,
            label_for_group=lambda _: 0,
            window_size=5,
            step=3,
            window_basis=FixedWindowBasis.RAW_POSITIONS,
            train_frac=0.33,
            test_frac=0.33,
        ).count_windows()

    with pytest.raises(
        ValueError,
        match="window_alignment_offset must be non-negative",
    ):
        FixedSequenceBuilder(
            sink=InMemoryStructuredSink(
                dataset_name="thunderbird",
                raw_dataset_path=raw_path,
                parser=parser,
                rows=retained_rows,
            ),
            infer_template=_upper_template,
            label_for_group=lambda _: 0,
            window_size=5,
            step=5,
            window_basis=FixedWindowBasis.RAW_POSITIONS,
            window_alignment_offset=-1,
            train_frac=0.33,
            test_frac=0.33,
        ).count_windows()

    with pytest.raises(TypeError, match="require ThunderbirdParser"):
        list(
            FixedSequenceBuilder(
                sink=InMemoryStructuredSink(
                    dataset_name="thunderbird",
                    raw_dataset_path=raw_path,
                    parser=NullStructuredParser(),
                    rows=retained_rows,
                ),
                infer_template=_upper_template,
                label_for_group=lambda _: 0,
                window_size=5,
                step=5,
                window_basis=FixedWindowBasis.RAW_POSITIONS,
                train_frac=0.33,
                test_frac=0.33,
            ),
        )

    window_size = 5
    expected_window_count = (len(raw_lines) - 1) // window_size
    raw_position_builder = FixedSequenceBuilder(
        sink=InMemoryStructuredSink(
            dataset_name="thunderbird",
            raw_dataset_path=raw_path,
            parser=parser,
            rows=retained_rows,
        ),
        infer_template=_upper_template,
        label_for_group=lambda _: 0,
        window_size=window_size,
        step=window_size,
        window_basis=FixedWindowBasis.RAW_POSITIONS,
        window_alignment_offset=1,
        train_frac=1 / 3,
        test_frac=1 / 3,
    )
    assert raw_position_builder.count_windows() == expected_window_count
    sequences = list(raw_position_builder)

    assert [sequence.split_label for sequence in sequences] == [
        SplitLabel.TRAIN,
        SplitLabel.IGNORED,
        SplitLabel.TEST,
    ]
