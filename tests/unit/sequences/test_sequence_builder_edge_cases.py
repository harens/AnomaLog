"""Edge-case regressions for the sequence builder helper branches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytest
from typing_extensions import override

from anomalog.parsers.structured.contracts import BaseStructuredLine
from anomalog.parsers.structured.deeplog_preprocessed import (
    DelimitedLabelledEventParser,
)
from anomalog.parsers.structured.parsers import ThunderbirdParser
from anomalog.parsers.template.parsers import IdentityTemplateParser
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
    from collections.abc import Callable, Collection, Iterator

    from prefect.assets import Asset

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


class _DelimitedParser(DelimitedLabelledEventParser):
    """Parser for the temporary raw-source benchmark fixture.

    Attributes:
        name (ClassVar[str]): Registry name used by the test helper.
    """

    name: ClassVar[str] = "delimited"

    @override
    def parse_line(self, raw_line: str) -> BaseStructuredLine | None:
        timestamp_s, entity_id, message, anomalous_s = raw_line.split("|", maxsplit=3)
        return BaseStructuredLine(
            timestamp_unix_ms=int(timestamp_s) if timestamp_s else None,
            entity_id=entity_id or None,
            untemplated_message_text=message,
            anomalous=int(anomalous_s) if anomalous_s else None,
        )


class _UpperTemplateParser(IdentityTemplateParser):
    """Template parser fixture that is intentionally not identity.

    Attributes:
        name (ClassVar[str]): Registry name used by the test helper.
        is_identity_parser (ClassVar[bool]): Explicitly disabled so the raw
            replay path exercises the non-identity branch under test.
    """

    name: ClassVar[str] = "upper"
    is_identity_parser: ClassVar[bool] = False

    @override
    def inference(
        self,
        unstructured_text: str,
    ) -> tuple[str, list[str]]:
        return unstructured_text.upper(), []

    @override
    def train(
        self,
        untemplated_text_iterator: Callable[[], Iterator[str]],
        *,
        asset_deps: list[Asset] | None = None,
    ) -> None:
        del untemplated_text_iterator, asset_deps


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


def test_before_grouping_raw_entry_helpers_count_train_test_and_straddlers() -> None:
    """Raw-entry counting helpers should accumulate per-group diagnostics.

    The helper is exercised through the public split-summary builder so the
    regression stays aligned with the supported API surface.
    """
    expected_train_raw_entry_count = 2
    expected_test_raw_entry_count = 2
    builder = EntitySequenceBuilder(
        sink=_sink(
            structured_line(
                line_order=0,
                timestamp_unix_ms=100,
                entity_id="entity-a",
                untemplated_message_text="train-normal",
                anomalous=0,
            ),
            structured_line(
                line_order=1,
                timestamp_unix_ms=101,
                entity_id="entity-a",
                untemplated_message_text="train-anomalous",
                anomalous=1,
            ),
            structured_line(
                line_order=3,
                timestamp_unix_ms=102,
                entity_id="entity-a",
                untemplated_message_text="test-normal",
                anomalous=0,
            ),
            structured_line(
                line_order=4,
                timestamp_unix_ms=103,
                entity_id="entity-b",
                untemplated_message_text="test-anomalous",
                anomalous=1,
            ),
        ),
        infer_template=_upper_template,
        label_for_group=lambda _: 0,
        split_mode=RawEntrySplitMode.PREFIX_COUNT,
        split_application_order=SplitApplicationOrder.BEFORE_GROUPING,
        straddling_group_policy=StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES,
        train_entry_count=2,
    )

    summary = builder.build_raw_entry_split_summary()

    assert summary is not None
    assert summary.train_raw_entry_count == expected_train_raw_entry_count
    assert summary.train_normal_entry_count == 1
    assert summary.train_anomalous_entry_count == 1
    assert summary.test_raw_entry_count == expected_test_raw_entry_count
    assert summary.test_normal_entry_count == 1
    assert summary.test_anomalous_entry_count == 1
    assert summary.straddling_group_count == 1


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
        template_parser=_UpperTemplateParser("demo"),
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
    assert [sequence.window_id for sequence in test_sequences] == [0]
    assert [sequence.split_label for sequence in test_sequences] == [
        SplitLabel.TEST,
    ]


def test_before_grouping_test_replay_uses_suffix_scan_helper() -> None:
    """Raw-entry test replay should use a suffix-only scan when available."""
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=Path("raw.log"),
        parser=NullStructuredParser(),
        rows=[
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
                entity_id="b",
                untemplated_message_text="first-b",
                anomalous=0,
            ),
        ],
    )
    builder = EntitySequenceBuilder(
        sink=sink,
        infer_template=_upper_template,
        label_for_group=lambda _: 0,
        template_parser=_UpperTemplateParser("demo"),
        split_mode=RawEntrySplitMode.PREFIX_COUNT,
        split_application_order=SplitApplicationOrder.BEFORE_GROUPING,
        straddling_group_policy=StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES,
        train_entry_count=2,
        train_frac=0.5,
        test_frac=0.5,
    )

    test_sequences = list(builder.iter_test_sequences())

    assert [sequence.window_id for sequence in test_sequences] == [0, 1]
    assert [sequence.split_label for sequence in test_sequences] == [
        SplitLabel.TEST,
        SplitLabel.TEST,
    ]
    assert [sequence.templates for sequence in test_sequences] == [
        ["THIRD-A"],
        ["FIRST-B"],
    ]


def test_before_grouping_test_replay_uses_raw_source_resume(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Raw-entry test replay should resume from the raw source cutoff.

    Args:
        tmp_path (Path): Temporary directory that holds the synthetic raw log.
        caplog (pytest.LogCaptureFixture): Capture fixture used to assert the
            resume log line is emitted.
    """
    caplog.set_level("INFO", logger="anomalog.sequences")
    raw_path = tmp_path / "raw.log"
    raw_path.write_text(
        "100|a|first-a|0\n200|a|second-a|0\n300|a|third-a|0\n400|b|first-b|0\n",
        encoding="utf-8",
    )
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=raw_path,
        parser=_DelimitedParser(),
        rows=[],
    )
    builder = EntitySequenceBuilder(
        sink=sink,
        infer_template=_upper_template,
        label_for_group=lambda _: 0,
        template_parser=_UpperTemplateParser("demo"),
        split_mode=RawEntrySplitMode.PREFIX_COUNT,
        split_application_order=SplitApplicationOrder.BEFORE_GROUPING,
        straddling_group_policy=StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES,
        train_entry_count=2,
        train_frac=0.5,
        test_frac=0.5,
    )

    test_sequences = list(builder.iter_test_sequences())

    assert (
        "Resuming before-grouping test replay for demo from raw entry 2" in caplog.text
    )
    assert [sequence.window_id for sequence in test_sequences] == [0, 1]
    assert [sequence.split_label for sequence in test_sequences] == [
        SplitLabel.TEST,
        SplitLabel.TEST,
    ]
    assert [sequence.templates for sequence in test_sequences] == [
        ["THIRD-A"],
        ["FIRST-B"],
    ]


def test_before_grouping_training_replay_caches_raw_source_offset(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Raw-entry training replay should cache the test boundary offset.

    Args:
        tmp_path (Path): Temporary directory that holds the synthetic raw log.
        caplog (pytest.LogCaptureFixture): Capture fixture used to assert the
            cached-offset log line is emitted.
    """
    caplog.set_level("INFO", logger="anomalog.sequences")
    raw_path = tmp_path / "raw.log"
    raw_path.write_text(
        "100|a|first-a|0\n200|a|second-a|0\n300|a|third-a|0\n400|b|first-b|0\n",
        encoding="utf-8",
    )
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=raw_path,
        parser=_DelimitedParser(),
        rows=[],
    )
    builder = EntitySequenceBuilder(
        sink=sink,
        infer_template=_upper_template,
        label_for_group=lambda _: 0,
        template_parser=_UpperTemplateParser("demo"),
        split_mode=RawEntrySplitMode.PREFIX_COUNT,
        split_application_order=SplitApplicationOrder.BEFORE_GROUPING,
        straddling_group_policy=StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES,
        train_entry_count=2,
        train_frac=0.5,
        test_frac=0.5,
    )

    train_sequences = list(builder.iter_training_sequences())
    test_sequences = list(builder.iter_test_sequences())

    assert builder.raw_replay_state.test_start_byte_offset is not None
    assert (
        "Resuming before-grouping test replay for demo from cached raw byte offset"
    ) in caplog.text
    assert [sequence.window_id for sequence in train_sequences] == [0]
    assert [sequence.window_id for sequence in test_sequences] == [0, 1]
    assert [sequence.split_label for sequence in test_sequences] == [
        SplitLabel.TEST,
        SplitLabel.TEST,
    ]


def test_before_grouping_test_replay_uses_dense_raw_source_resume(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Dense raw-session replay should resume without rebuilding the prefix.

    Args:
        tmp_path (Path): Temporary directory that holds the synthetic raw log.
        caplog (pytest.LogCaptureFixture): Capture fixture used to assert the
            cached-offset log line is emitted.
    """
    caplog.set_level("INFO", logger="anomalog.sequences")
    raw_path = tmp_path / "raw.log"
    raw_path.write_text(
        "a\t0\tfirst-a\na\t0\tsecond-a\na\t0\tthird-a\nb\t0\tfirst-b\n",
        encoding="utf-8",
    )
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=raw_path,
        parser=DelimitedLabelledEventParser(),
        rows=[],
    )
    builder = EntitySequenceBuilder(
        sink=sink,
        infer_template=IdentityTemplateParser("demo").inference,
        label_for_group=lambda _: 0,
        template_parser=IdentityTemplateParser("demo"),
        split_mode=RawEntrySplitMode.PREFIX_COUNT,
        split_application_order=SplitApplicationOrder.BEFORE_GROUPING,
        straddling_group_policy=StraddlingGroupPolicy.SPLIT_PARTIAL_SEQUENCES,
        train_entry_count=2,
        train_frac=0.5,
        test_frac=0.5,
    )

    train_sequences = list(builder.iter_training_sequences())
    test_sequences = list(builder.iter_test_sequences())

    assert builder.raw_replay_state.test_start_byte_offset is not None
    assert (
        "Resuming before-grouping test replay for demo from cached raw byte offset"
    ) in caplog.text
    assert [sequence.window_id for sequence in train_sequences] == [0]
    assert [sequence.window_id for sequence in test_sequences] == [0, 1]
    assert [sequence.split_label for sequence in test_sequences] == [
        SplitLabel.TEST,
        SplitLabel.TEST,
    ]
    assert [sequence.templates for sequence in test_sequences] == [
        ["third-a"],
        ["first-b"],
    ]


def test_before_grouping_raw_entry_summary_streams_and_caches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw-entry split summaries should stream once and then reuse the cache.

    Args:
        monkeypatch (pytest.MonkeyPatch): Test double used to observe that the
            summary calculation only scans the grouped stream once.
    """
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
            timestamp_unix_ms=200,
            entity_id="a",
            untemplated_message_text="two",
            anomalous=1,
        ),
        structured_line(
            line_order=2,
            timestamp_unix_ms=300,
            entity_id="a",
            untemplated_message_text="three",
            anomalous=0,
        ),
        structured_line(
            line_order=3,
            timestamp_unix_ms=400,
            entity_id="a",
            untemplated_message_text="four",
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
    )

    summary = builder.build_raw_entry_split_summary()
    assert summary is not None
    expected_train_raw_entry_count = 2
    expected_train_normal_entry_count = 1
    expected_train_anomalous_entry_count = 1
    expected_test_raw_entry_count = 2
    expected_test_normal_entry_count = 2
    expected_test_anomalous_entry_count = 0
    expected_straddling_group_count = 1
    assert summary.train_raw_entry_count == expected_train_raw_entry_count
    assert summary.train_normal_entry_count == expected_train_normal_entry_count
    assert summary.train_anomalous_entry_count == expected_train_anomalous_entry_count
    assert summary.test_raw_entry_count == expected_test_raw_entry_count
    assert summary.test_normal_entry_count == expected_test_normal_entry_count
    assert summary.test_anomalous_entry_count == expected_test_anomalous_entry_count
    assert summary.straddling_group_count == expected_straddling_group_count

    monkeypatch.setattr(
        EntitySequenceBuilder,
        "_build_before_grouping_raw_entry_split_summary",
        lambda _self: (_ for _ in ()).throw(
            AssertionError("raw-entry split summary should be cached"),
        ),
    )

    assert builder.build_raw_entry_split_summary() is summary


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
