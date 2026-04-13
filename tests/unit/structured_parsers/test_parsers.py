"""Tests for concrete structured parsers."""

from anomalog.parsers.structured import (
    resolve_structured_parser,
    structured_parser_names,
)
from anomalog.parsers.structured.parsers import BGLParser, HDFSV1Parser

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
    assert resolve_structured_parser("hdfs_v1") is HDFSV1Parser
    assert set(structured_parser_names()) >= {"bgl", "hdfs_v1"}
