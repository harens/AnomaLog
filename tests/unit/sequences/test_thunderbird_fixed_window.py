"""Thunderbird fixed-window regression tests."""

from pathlib import Path

from anomalog.parsers.structured.parsers import ThunderbirdParser
from anomalog.sequences import FixedSequenceBuilder, FixedWindowBasis, SplitLabel
from tests.unit.helpers import InMemoryStructuredSink, structured_line
from tests.unit.sequences.test_sequences import _upper_template


def test_thunderbird_raw_position_windows_preserve_skipped_labels(
    tmp_path: Path,
) -> None:
    """Raw-position windows should keep skipped rows from shifting labels."""
    raw_lines = [
        "- 1131566461 2005.11.09 dn228 Nov 9 12:01:01 dn228/dn228 normal-1",
        "- 1131566462 2005.11.09 dn228 Nov 9 12:01:02 dn228/dn228 normal-2",
        "+ 1131566463 2005.11.09 dn228 Nov 9 12:01:03 dn228/dn228",
        "- 1131566464 2005.11.09 dn228 Nov 9 12:01:04 dn228/dn228 normal-3",
        "- 1131566465 2005.11.09 dn228 Nov 9 12:01:05 dn228/dn228 normal-4",
        "- 1131566466 2005.11.09 dn228 Nov 9 12:01:06 dn228/dn228 normal-5",
        "- 1131566467 2005.11.09 dn228 Nov 9 12:01:07 dn228/dn228 normal-6",
        "- 1131566468 2005.11.09 dn228 Nov 9 12:01:08 dn228/dn228 normal-7",
        "- 1131566469 2005.11.09 dn228 Nov 9 12:01:09 dn228/dn228 normal-8",
        "- 1131566470 2005.11.09 dn228 Nov 9 12:01:10 dn228/dn228 normal-9",
    ]
    window_size = 5
    expected_compacted_window_count = 1
    expected_raw_window_count = 2
    expected_raw_total_count = 2
    raw_path = tmp_path / "Thunderbird.log"
    raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    parser = ThunderbirdParser()
    retained_rows = []
    for line_order, raw_line in enumerate(raw_lines):
        parsed = parser.parse_line(raw_line)
        if parsed is None:
            continue
        retained_rows.append(
            structured_line(
                line_order=line_order,
                timestamp_unix_ms=parsed.timestamp_unix_ms,
                entity_id=parsed.entity_id,
                untemplated_message_text=parsed.untemplated_message_text,
                anomalous=parsed.anomalous,
            ),
        )

    compacted_builder = FixedSequenceBuilder(
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
        train_frac=0.5,
        test_frac=0.5,
    )
    raw_builder = FixedSequenceBuilder(
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
        train_frac=0.5,
        test_frac=0.5,
    )

    compacted_sequences = list(compacted_builder)
    raw_sequences = list(raw_builder)

    assert len(compacted_sequences) == expected_compacted_window_count
    assert compacted_sequences[0].label == 0
    assert compacted_sequences[0].split_label is SplitLabel.TEST
    assert len(compacted_sequences[0].events) == window_size

    assert len(raw_sequences) == expected_raw_window_count
    assert [sequence.split_label for sequence in raw_sequences] == [
        SplitLabel.TRAIN,
        SplitLabel.TEST,
    ]
    assert [sequence.label for sequence in raw_sequences] == [1, 0]
    assert [len(sequence.events) for sequence in raw_sequences] == [4, 5]
    assert raw_builder.split_count_hint().total_count == expected_raw_total_count
