"""Tests for concrete structured parsers."""

import pytest

from anomalog.parsers.structured import (
    resolve_structured_parser,
    structured_parser_names,
)
from anomalog.parsers.structured.deeplog_preprocessed import (
    DelimitedLabelledEventParser,
)
from anomalog.parsers.structured.parsers import (
    BGLParser,
    HDFSV1Parser,
    OpenStackDeepLogParser,
)

HDFS_SAMPLE_TS_MS = 1_226_262_918_000
BGL_FALLBACK_TS_MS = 1_117_838_570_000


def test_hdfs_parser_uses_component_when_block_id_is_missing() -> None:
    """HDFSV1Parser falls back to the component as the entity id."""
    parsed = HDFSV1Parser().parse_line(
        "081109 203518 143 INFO dfs.NameNode: Completed checkpoint successfully",
    )

    assert parsed is not None
    assert parsed.entity_id == "dfs.NameNode"
    assert parsed.timestamp_unix_ms == HDFS_SAMPLE_TS_MS


def test_hdfs_parser_returns_none_for_unparseable_lines() -> None:
    """Malformed HDFS lines are skipped."""
    assert HDFSV1Parser().parse_line("not a real hdfs line") is None


def test_bgl_parser_falls_back_to_epoch_seconds_when_hires_timestamp_is_invalid() -> (
    None
):
    """BGLParser uses the epoch field when the high-resolution timestamp fails."""
    parsed = BGLParser().parse_line(
        "1117838570 2005.06.03 R02-M1-N0-C:J12-U11 "
        "2005-99-03-15.42.50.363779 R02-M1-N0-C:J12-U11 "
        "RAS KERNEL INFO cache parity corrected",
    )

    assert parsed is not None
    assert parsed.timestamp_unix_ms == BGL_FALLBACK_TS_MS
    assert parsed.anomalous == 1


def test_structured_parser_registry_resolves_builtins() -> None:
    """Built-in structured parsers register themselves by config name."""
    assert resolve_structured_parser("bgl") is BGLParser
    assert resolve_structured_parser("delimited_labelled_event") is (
        DelimitedLabelledEventParser
    )
    assert resolve_structured_parser("hdfs_v1") is HDFSV1Parser
    assert resolve_structured_parser("openstack_deeplog") is OpenStackDeepLogParser
    assert set(structured_parser_names()) >= {
        "bgl",
        "delimited_labelled_event",
        "hdfs_v1",
        "openstack_deeplog",
    }


def test_openstack_deeplog_parser_groups_by_split_scoped_minute() -> None:
    """OpenStack parser should create one-minute split-scoped entity buckets."""
    parser = OpenStackDeepLogParser()
    parsed = parser.parse_line(
        "openstack_train\t0\t100 2017-01-01 00:00:30.000 1 INFO nova.compute "
        "[instance-0001] Build complete",
    )

    assert parsed is not None
    assert parsed.entity_id == "openstack_train:2017-01-01 00:00"
    assert parsed.anomalous == 0
    assert parsed.untemplated_message_text == "INFO nova.compute Build complete"


@pytest.mark.parametrize(
    ("raw_line", "expected_entity_id", "expected_message"),
    [
        (
            (
                "openstack_test_normal\t0\t"
                "nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 03:19:45.356 "
                "2931 ERROR oslo_service.periodic_task "
                "Traceback (most recent call last):"
            ),
            "openstack_test_normal:2017-05-16 03:19",
            "ERROR oslo_service.periodic_task Traceback (most recent call last):",
        ),
        (
            (
                "openstack_test_abnormal\t1\t"
                "nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 03:19:45.356 "
                "2931 ERROR oslo_service.periodic_task"
            ),
            "openstack_test_abnormal:2017-05-16 03:19",
            "ERROR oslo_service.periodic_task",
        ),
    ],
)
def test_openstack_deeplog_parser_accepts_rows_without_addresses(
    raw_line: str,
    expected_entity_id: str,
    expected_message: str,
) -> None:
    """OpenStack traceback and terminal error rows should still parse.

    Args:
        raw_line (str): Labelled OpenStack row to parse.
        expected_entity_id (str): Minute-bucket entity id expected from the row.
        expected_message (str): Message text expected after parsing.
    """
    parsed = OpenStackDeepLogParser().parse_line(raw_line)

    assert parsed is not None
    assert parsed.entity_id == expected_entity_id
    assert parsed.untemplated_message_text == expected_message


def test_structured_parser_registry_rejects_unknown_names() -> None:
    """Unknown structured parser names raise a descriptive KeyError."""
    with pytest.raises(KeyError, match="Unsupported structured parser: 'missing'"):
        resolve_structured_parser("missing")
