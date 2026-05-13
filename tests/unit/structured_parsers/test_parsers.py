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


def test_openstack_deeplog_parser_uses_instance_id_for_entity() -> None:
    """OpenStack parser should namespace the VM instance id by the split."""
    parser = OpenStackDeepLogParser()
    parsed = parser.parse_line(
        "openstack_train\t0\t100 2017-01-01 00:00:30.000 1 INFO nova.compute "
        "[instance: b9000564-fe1a-409b-b8cc-1e88b294cd1d] Build complete",
    )

    assert parsed is not None
    assert parsed.entity_id == "openstack_train:b9000564-fe1a-409b-b8cc-1e88b294cd1d"
    assert parsed.anomalous == 0
    assert parsed.untemplated_message_text == "Build complete"


def test_openstack_deeplog_parser_normalises_numeric_message_tokens() -> None:
    """OpenStack parser should strip session markers before template mining."""
    parser = OpenStackDeepLogParser()
    parsed = parser.parse_line(
        "openstack_train\t0\t"
        "100 2017-01-01 00:00:30.000 1 INFO nova.compute "
        "[instance: 29d7b230-75ab-4140-81d8-7353d8f7e69b] "
        "Attempting claim: memory 2048 MB, disk 20 GB, vcpus 1 CPU",
    )

    assert parsed is not None
    assert parsed.entity_id == "openstack_train:29d7b230-75ab-4140-81d8-7353d8f7e69b"
    assert parsed.untemplated_message_text == (
        "Attempting claim: memory NUM MB, disk NUM GB, vcpus NUM CPU"
    )


def test_openstack_deeplog_parser_collapses_instance_storage_paths() -> None:
    """OpenStack parser should collapse instance-store filenames to one key."""
    parser = OpenStackDeepLogParser()
    parsed = parser.parse_line(
        "openstack_train\t0\t"
        "100 2017-01-01 00:00:30.000 1 INFO nova.compute "
        "[instance: 29d7b230-75ab-4140-81d8-7353d8f7e69b] "
        "Base or swap file too young to remove: "
        "/var/lib/nova/instances/_base/a489c868f0c37da93b76227c91bb03908ac0e742",
    )

    assert parsed is not None
    assert parsed.untemplated_message_text == (
        "Base or swap file too young to remove: INSTANCE_PATH"
    )


def test_openstack_deeplog_parser_preserves_http_path_structure() -> None:
    """OpenStack parser should keep HTTP route structure after normalisation."""
    parser = OpenStackDeepLogParser()
    parsed = parser.parse_line(
        "openstack_train\t0\t"
        "100 2017-01-01 00:00:30.000 1 INFO nova.compute "
        "[instance: 29d7b230-75ab-4140-81d8-7353d8f7e69b] "
        '10.11.10.1 "GET /v2/54fadb412c4e40cdbaed9335e4c35a9e/servers/detail '
        'HTTP/1.1" status: 200 len: 1893 time: 0.2477829',
    )

    assert parsed is not None
    assert parsed.untemplated_message_text == (
        'IP "GET /v2/HEX/servers/detail HTTP/NUM" status: NUM len: NUM time: NUM'
    )


def test_openstack_deeplog_parser_preserves_pending_task_state() -> None:
    """OpenStack parser should keep task-state text available to the template."""
    parser = OpenStackDeepLogParser()
    parsed = parser.parse_line(
        "openstack_test_normal\t0\t"
        "100 2017-01-01 00:00:30.000 1 INFO nova.compute "
        "[instance: 29d7b230-75ab-4140-81d8-7353d8f7e69b] "
        "During sync_power_state the instance has a pending task "
        "(networking). Skip.",
    )

    assert parsed is not None
    assert parsed.untemplated_message_text == (
        "During sync_power_state the instance has a pending task (networking). Skip."
    )


@pytest.mark.parametrize(
    "raw_line",
    [
        (
            "openstack_test_normal\t0\t"
            "nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 03:19:45.356 "
            "2931 ERROR oslo_service.periodic_task Traceback (most recent call last):"
        ),
        (
            "openstack_test_abnormal\t1\t"
            "nova-compute.log.1.2017-05-16_13:55:31 2017-05-16 03:19:45.356 "
            "2931 ERROR oslo_service.periodic_task"
        ),
    ],
)
def test_openstack_deeplog_parser_skips_rows_without_instance_handles(
    raw_line: str,
) -> None:
    """OpenStack rows without an instance id should be skipped.

    Args:
        raw_line (str): Labelled OpenStack row lacking an instance id.
    """
    assert OpenStackDeepLogParser().parse_line(raw_line) is None


def test_structured_parser_registry_rejects_unknown_names() -> None:
    """Unknown structured parser names raise a descriptive KeyError."""
    with pytest.raises(KeyError, match="Unsupported structured parser: 'missing'"):
        resolve_structured_parser("missing")
