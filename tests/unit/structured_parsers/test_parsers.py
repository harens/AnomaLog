"""Tests for concrete structured parsers."""

import re

import pytest

from anomalog.parsers.structured import parsers as structured_parsers
from anomalog.parsers.structured import (
    resolve_structured_parser,
    structured_parser_names,
)
from anomalog.parsers.structured.contracts import StructuredLine
from anomalog.parsers.structured.deeplog_preprocessed import (
    DelimitedLabelledEventParser,
)
from anomalog.parsers.structured.parsers import (
    AITADSParser,
    BGLParser,
    HDFSV1Parser,
    OpenStackDeepLogParser,
    ThunderbirdParser,
)

_PARSE_OPENSTACK_LABELLED_ROW = vars(structured_parsers)[
    "_parse_openstack_labelled_row"
]
_COERCE_OPTIONAL_INT = vars(structured_parsers)["_coerce_optional_int"]
_COERCE_OPTIONAL_INT_EXPECTED = 12

HDFS_SAMPLE_TS_MS = 1_226_262_918_000
BGL_FALLBACK_TS_MS = 1_117_838_570_000
AIT_ADS_SAMPLE_TS_MS = 1_642_410_287_000
THUNDERBIRD_SAMPLE_TS_MS = 1_131_566_461_000
THUNDERBIRD_EXIT_SIGNAL_TS_MS = 1_133_559_328_000
THUNDERBIRD_FIXUP_TS_MS = 1_133_563_453_000


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


def test_hdfs_parser_keeps_timestamp_none_when_header_timestamp_is_invalid() -> None:
    """HDFSV1Parser should continue parsing when the timestamp field is bad."""
    parsed = HDFSV1Parser().parse_line(
        "081109 999999 143 INFO dfs.NameNode: Completed checkpoint successfully",
    )

    assert parsed is not None
    assert parsed.timestamp_unix_ms is None
    assert parsed.entity_id == "dfs.NameNode"


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


def test_bgl_parser_keeps_row_when_epoch_fallback_is_unparseable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BGLParser should preserve the row even when the epoch fallback fails.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the BGL regex so the epoch
            fallback branch can be exercised deterministically.
    """
    monkeypatch.setattr(
        BGLParser,
        "_BGL_RE",
        re.compile(
            r"""
            ^\s*
            (?P<dash>-)?\s*
            (?:(?P<prefix>\d+:\S+)\s+)?(?:\S+\s+)?
            (?P<epoch>\S+)\s+
            (?P<date>\d{4}\.\d{2}\.\d{2})\s+
            (?P<entity>\S+)\s+
            (?P<hires_ts>\d{4}-\d{2}-\d{2}-\d{2}\.\d{2}\.\d{2}\.\d+)\s+
            (?P<entity2>\S+)\s+
            (?P<tail>\S+\s+\S+\s+\S+.*)
            \s*$
            """,
            re.VERBOSE,
        ),
    )

    parsed = BGLParser().parse_line(
        "not-an-epoch 2005.06.03 R02-M1-N0-C:J12-U11 "
        "2005-99-03-15.42.50.363779 R02-M1-N0-C:J12-U11 "
        "RAS KERNEL INFO cache parity corrected",
    )

    assert parsed is not None
    assert parsed.timestamp_unix_ms is None


def test_structured_parser_registry_resolves_builtins() -> None:
    """Built-in structured parsers register themselves by config name."""
    assert resolve_structured_parser("bgl") is BGLParser
    assert resolve_structured_parser("delimited_labelled_event") is (
        DelimitedLabelledEventParser
    )
    assert resolve_structured_parser("hdfs_v1") is HDFSV1Parser
    assert resolve_structured_parser("ait_ads") is AITADSParser
    assert resolve_structured_parser("openstack_deeplog") is OpenStackDeepLogParser
    assert set(structured_parser_names()) >= {
        "ait_ads",
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
    assert isinstance(parsed, StructuredLine)
    assert parsed.entity_id == "openstack_train:29d7b230-75ab-4140-81d8-7353d8f7e69b"
    assert parsed.untemplated_message_text == (
        "Attempting claim: memory NUM MB, disk NUM GB, vcpus NUM CPU"
    )
    assert parsed.raw_parameters == ["2048", "20", "1"]


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


def test_openstack_deeplog_parser_accepts_for_instance_instance_ids() -> None:
    """OpenStack parser should recognise alternate instance-id phrasing."""
    parser = OpenStackDeepLogParser()
    parsed = parser.parse_line(
        "openstack_train\t0\t100 2017-01-01 00:00:30.000 1 INFO nova.compute "
        "[addr] Build complete for instance vm-alpha",
    )

    assert parsed is not None
    assert parsed.entity_id == "openstack_train:vm-alpha"
    assert parsed.untemplated_message_text == "Build complete for instance vm-alpha"


def test_openstack_deeplog_parser_skips_rows_with_invalid_timestamps() -> None:
    """OpenStack parser should skip rows that fail timestamp parsing."""
    parser = OpenStackDeepLogParser()

    assert (
        parser.parse_line(
            "openstack_train\t0\t100 2017-99-01 00:00:30.000 1 INFO nova.compute "
            "[instance: vm-alpha] Build complete",
        )
        is None
    )


def test_openstack_deeplog_parser_skips_rows_with_invalid_label_or_payload() -> None:
    """Malformed labelled OpenStack rows should be rejected early."""
    assert _PARSE_OPENSTACK_LABELLED_ROW("missing tabs") is None
    assert (
        _PARSE_OPENSTACK_LABELLED_ROW(
            "openstack_train\t2\t100 2017-01-01 00:00:30.000 1 INFO nova.compute "
            "[instance: vm-alpha] Build complete",
        )
        is None
    )


def test_coerce_optional_int_handles_non_numeric_values() -> None:
    """Optional integer coercion should accept ints and reject bad payloads."""
    assert _COERCE_OPTIONAL_INT(None) is None
    assert _COERCE_OPTIONAL_INT("12") == _COERCE_OPTIONAL_INT_EXPECTED
    assert _COERCE_OPTIONAL_INT("bad") is None


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


def test_thunderbird_parser_reads_label_timestamp_and_message_body() -> None:
    """ThunderbirdParser should keep the message body and anomaly label."""
    parsed = ThunderbirdParser().parse_line(
        "- 1131566461 2005.11.09 dn228 Nov 9 12:01:01 dn228/dn228 "
        "crond(pam_unix)[2915]: session closed for user root",
    )

    assert parsed is not None
    assert parsed.timestamp_unix_ms == THUNDERBIRD_SAMPLE_TS_MS
    assert parsed.entity_id == "dn228/dn228"
    assert parsed.anomalous == 0
    assert parsed.untemplated_message_text == "session closed for user root"


def test_thunderbird_parser_treats_non_dash_labels_as_anomalous() -> None:
    """ThunderbirdParser should map any non-dash label token to anomaly."""
    parsed = ThunderbirdParser().parse_line(
        "+ 2005.11.09 dn228 Nov 9 12:02:02 dn228/dn228 "
        "sshd[1234]: disk failure on /dev/sda",
    )

    assert parsed is not None
    assert parsed.timestamp_unix_ms is None
    assert parsed.anomalous == 1
    assert parsed.entity_id == "dn228/dn228"


def test_thunderbird_parser_accepts_tails_without_component_separators() -> None:
    """ThunderbirdParser should keep lines that do not use a colon separator."""
    parsed = ThunderbirdParser().parse_line(
        "- 1133559328 2005.12.02 #1# Dec 2 13:35:28 #1#/#1# exiting on signal 15",
    )

    assert parsed is not None
    assert parsed.timestamp_unix_ms == THUNDERBIRD_EXIT_SIGNAL_TS_MS
    assert parsed.entity_id == "#1#/#1#"
    assert parsed.anomalous == 0
    assert parsed.untemplated_message_text == "exiting on signal 15"


def test_thunderbird_parser_strips_trailing_colons_from_message_tails() -> None:
    """ThunderbirdParser should normalise colon-suffixed message tails."""
    parsed = ThunderbirdParser().parse_line(
        "- 1133563453 2005.12.02 tsqe1 Dec 2 14:44:13 tsqe1/tsqe1 mysql_install_db:",
    )

    assert parsed is not None
    assert parsed.timestamp_unix_ms == THUNDERBIRD_FIXUP_TS_MS
    assert parsed.entity_id == "tsqe1/tsqe1"
    assert parsed.untemplated_message_text == "mysql_install_db"


def test_thunderbird_parser_reports_blank_and_malformed_lines() -> None:
    """ThunderbirdParser should skip blank and malformed lines cleanly."""
    parser = ThunderbirdParser()

    assert parser.analyse_line("   \n") == (None, "blank")
    assert parser.analyse_line("not a thunderbird line") == (None, "malformed")


def test_thunderbird_parser_logs_malformed_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ThunderbirdParser should warn when malformed rows are skipped.

    Args:
        caplog (pytest.LogCaptureFixture): Log capture fixture used to assert
            the warning emitted for malformed Thunderbird rows.
    """
    parser = ThunderbirdParser()

    with caplog.at_level("WARNING"):
        assert parser.parse_line("not a thunderbird line") is None

    assert "Cannot parse Thunderbird line (malformed)" in caplog.text


def test_thunderbird_parser_omits_timestamp_when_header_is_missing() -> None:
    """ThunderbirdParser should leave the timestamp unset when it is omitted."""
    parsed = ThunderbirdParser().parse_line(
        "- 2022.01.18 zeus Jan 18 02:39:00 node-1 message body",
    )

    assert parsed is not None
    assert parsed.timestamp_unix_ms is None


def test_thunderbird_parser_skips_empty_message_lines_without_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """ThunderbirdParser should treat header-only records as expected skips.

    Args:
        caplog (pytest.LogCaptureFixture): Log capture fixture used to verify
            the warning suppression.
    """
    parser = ThunderbirdParser()

    with caplog.at_level("WARNING"):
        assert (
            parser.parse_line(
                "- 1147467134 2006.05.12 #9# May 12 13:52:14 #9#/#9#",
            )
            is None
        )

    assert "empty_message" not in caplog.text


def test_ait_ads_parser_reads_canonical_alert_rows() -> None:
    """AIT-ADS parser should map canonical JSON rows into structured lines."""
    parser = AITADSParser()
    parsed = parser.parse_line(
        '{"alert_uid":"fox:fox_aminer.json:0","anomalous":1,'
        '"attack_phase":"service_stop","entity_id":"fox:aminer:172.17.129.140",'
        '"ids_source":"aminer","metadata":{"analysis_component_identifier":3},'
        '"original_timestamp":"1642410287.0","scenario":"fox","source_file":"fox_aminer.json",'
        '"source_line_order":0,"template_key":"aminer|type=NewMatchPathDetector|'
        'name=AMiner: New event type.|key=nmpd",'
        '"timestamp_unix_ms":1642410287000}',
    )

    assert parsed is not None
    assert parsed.timestamp_unix_ms == AIT_ADS_SAMPLE_TS_MS
    assert parsed.entity_id == "fox:aminer:172.17.129.140"
    assert parsed.anomalous == 1
    assert parsed.untemplated_message_text.startswith(
        "aminer|type=NewMatchPathDetector",
    )


def test_ait_ads_parser_coerces_invalid_numeric_fields_to_none() -> None:
    """AIT-ADS parser should ignore malformed optional numeric payload fields."""
    parser = AITADSParser()
    parsed = parser.parse_line(
        '{"entity_id":"fox:aminer:172.17.129.140",'
        '"anomalous":"not-a-number",'
        '"template_key":"aminer|type=NewMatchPathDetector|key=nmpd",'
        '"timestamp_unix_ms":"also-not-a-number"}',
    )

    assert parsed is not None
    assert parsed.timestamp_unix_ms is None
    assert parsed.anomalous is None
    assert parsed.entity_id == "fox:aminer:172.17.129.140"
    assert parsed.untemplated_message_text == (
        "aminer|type=NewMatchPathDetector|key=nmpd"
    )


def test_ait_ads_parser_rejects_blank_and_malformed_rows(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """AIT-ADS parser should skip blank, non-object, and incomplete rows.

    Args:
        caplog (pytest.LogCaptureFixture): Log capture fixture used to assert
            warnings for malformed AIT-ADS rows.
    """
    parser = AITADSParser()

    with caplog.at_level("WARNING"):
        assert parser.parse_line("   ") is None
        assert parser.parse_line("not-json") is None
        assert parser.parse_line("1") is None
        assert parser.parse_line('{"entity_id":"demo"}') is None

    assert "template_key" in caplog.text


def test_structured_parser_registry_rejects_unknown_names() -> None:
    """Unknown structured parser names raise a descriptive KeyError."""
    with pytest.raises(KeyError, match="Unsupported structured parser: 'missing'"):
        resolve_structured_parser("missing")
