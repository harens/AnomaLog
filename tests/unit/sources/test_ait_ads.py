"""Tests for the AIT-ADS source helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from anomalog.sources.ait_ads import (
    AITADSScenarioSource,
    load_ait_ads_label_windows,
    materialise_ait_ads_alert_stream,
)
from anomalog.sources.local import LocalDirSource

if TYPE_CHECKING:
    from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_fixture_source_tree(tmp_path: Path) -> Path:
    source_root = tmp_path / "ait_ads_source"
    source_root.mkdir()
    (source_root / "labels.csv").write_text(
        (
            "scenario,attack,start,end\n"
            "fox,service_stop,1642410286.0,1642410288.0\n"
            "fox,dnsteal,1642410288.0,1642413600.0\n"
            "fox,network_scans,1642507140.0,1642508220.0\n"
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        source_root / "fox_aminer.json",
        [
            {
                "AnalysisComponent": {
                    "AnalysisComponentIdentifier": 3,
                    "AnalysisComponentType": "NewMatchPathDetector",
                    "AnalysisComponentName": "AMiner: New event type.",
                    "Message": "New path(es) detected",
                    "PersistenceFileName": "nmpd",
                },
                "LogData": {
                    "RawLogData": ["first"],
                    "Timestamps": [1642410287.0],
                    "DetectionTimestamp": [1642410287.0],
                    "LogLinesCount": 1,
                    "LogResources": ["/var/log/audit/audit.log"],
                },
                "AMiner": {"ID": "172.17.129.140"},
            },
            {
                "AnalysisComponent": {
                    "AnalysisComponentIdentifier": 19,
                    "AnalysisComponentType": "NewMatchPathValueComboDetector",
                    "AnalysisComponentName": (
                        "AMiner: New user_acct parameter combination in Audit logs."
                    ),
                    "Message": "New value combination(s) detected",
                    "PersistenceFileName": "nmpvcd_user_acct",
                },
                "LogData": {
                    "RawLogData": ["second"],
                    "Timestamps": [1642507140.0],
                    "DetectionTimestamp": [1642507140.0],
                    "LogLinesCount": 1,
                    "LogResources": ["/var/log/audit/audit.log"],
                },
                "AMiner": {"ID": "172.17.129.140"},
            },
        ],
    )
    _write_jsonl(
        source_root / "fox_wazuh.json",
        [
            {
                "agent": {"ip": "172.17.131.81", "name": "wazuh-client", "id": "18"},
                "manager": {"name": "wazuh.manager"},
                "rule": {
                    "id": "52507",
                    "level": 3,
                    "description": "ClamAV database update",
                    "groups": ["clamd", "freshclam", "virus"],
                },
                "decoder": {"name": "freshclam"},
                "full_log": "line",
                "input": {"type": "log"},
                "@timestamp": "2022-01-18T02:39:00.000000Z",
                "location": "/var/log/syslog",
                "id": "1686147193.86593",
            },
            {
                "agent": {"ip": "10.35.32.1", "name": "wazuh-client", "id": "29"},
                "manager": {"name": "wazuh.manager"},
                "data": {
                    "tx_id": "0",
                    "event_type": "alert",
                    "alert": {
                        "severity": "3",
                        "signature_id": "2013504",
                        "rev": "6",
                        "gid": "1",
                        "signature": (
                            "ET POLICY GNU/Linux APT User-Agent Outbound "
                            "likely related to package management"
                        ),
                        "action": "allowed",
                        "category": "Not Suspicious Traffic",
                    },
                    "timestamp": "2022-01-18T02:38:08.500000+0000",
                },
                "rule": {
                    "id": "86601",
                    "level": 3,
                    "description": (
                        "Suricata: Alert - ET POLICY GNU/Linux APT "
                        "User-Agent Outbound likely related to package management"
                    ),
                    "groups": ["ids", "suricata"],
                },
                "decoder": {"name": "json"},
                "input": {"type": "log"},
                "@timestamp": "2022-01-18T02:38:08.500000Z",
                "location": "/var/log/suricata/eve.json",
                "id": "1686151580.101926",
            },
        ],
    )
    return source_root


def _write_multi_scenario_fixture_source_tree(tmp_path: Path) -> Path:
    source_root = _write_fixture_source_tree(tmp_path)
    (source_root / "harrison_aminer.json").write_text(
        (
            '{"AnalysisComponent":{"AnalysisComponentIdentifier":7,'
            '"AnalysisComponentType":"NewMatchPathDetector",'
            '"AnalysisComponentName":"AMiner: Harrison event.",'
            '"Message":"New path(es) detected",'
            '"PersistenceFileName":"nmpd_h"},'
            '"LogData":{"RawLogData":["first"],'
            '"Timestamps":[1642410286.5],'
            '"DetectionTimestamp":[1642410286.5],'
            '"LogLinesCount":1,'
            '"LogResources":["/var/log/audit/audit.log"]},'
            '"AMiner":{"ID":"172.17.129.141"}}\n'
        ),
        encoding="utf-8",
    )
    (source_root / "harrison_wazuh.json").write_text(
        (
            '{"agent":{"ip":"172.17.131.82","name":"wazuh-client",'
            '"id":"19"},'
            '"manager":{"name":"wazuh.manager"},'
            '"rule":{"id":"52508","level":3,'
            '"description":"Harrison Wazuh alert","groups":["clamd"]},'
            '"decoder":{"name":"freshclam"},'
            '"full_log":"line","input":{"type":"log"},'
            '"@timestamp":"2022-01-18T02:38:07.000000Z",'
            '"location":"/var/log/syslog","id":"1686147193.86594"}\n'
        ),
        encoding="utf-8",
    )
    (source_root / "labels.csv").write_text(
        (
            "scenario,attack,start,end\n"
            "fox,service_stop,1642410286.0,1642410288.0\n"
            "harrison,service_stop,1642410286.0,1642410288.0\n"
        ),
        encoding="utf-8",
    )
    return source_root


def _write_boundary_fixture_source_tree(tmp_path: Path) -> Path:
    source_root = tmp_path / "ait_ads_boundary_source"
    source_root.mkdir()
    (source_root / "labels.csv").write_text(
        ("scenario,attack,start,end\nfox,service_stop,0.1,0.2\n"),
        encoding="utf-8",
    )
    _write_jsonl(
        source_root / "fox_aminer.json",
        [
            {
                "AnalysisComponent": {
                    "AnalysisComponentIdentifier": 3,
                    "AnalysisComponentType": "NewMatchPathDetector",
                    "AnalysisComponentName": "AMiner: Boundary event.",
                    "Message": "New path(es) detected",
                    "PersistenceFileName": "nmpd",
                },
                "LogData": {
                    "RawLogData": ["boundary"],
                    "Timestamps": [0.2],
                    "DetectionTimestamp": [0.2],
                    "LogLinesCount": 1,
                    "LogResources": ["/var/log/audit/audit.log"],
                },
                "AMiner": {"ID": "172.17.129.140"},
            },
        ],
    )
    _write_jsonl(
        source_root / "fox_wazuh.json",
        [
            {
                "agent": {"ip": "10.35.32.1", "name": "wazuh-client", "id": "29"},
                "manager": {"name": "wazuh.manager"},
                "data": {
                    "tx_id": "0",
                    "event_type": "alert",
                    "alert": {
                        "severity": "3",
                        "signature_id": "2013504",
                        "rev": "6",
                        "gid": "1",
                        "signature": (
                            "ET POLICY GNU/Linux APT User-Agent Outbound "
                            "likely related to package management"
                        ),
                        "action": "allowed",
                        "category": "Not Suspicious Traffic",
                    },
                    "timestamp": "1970-01-01T00:00:00.000000+0000",
                },
                "rule": {
                    "id": "86601",
                    "level": 3,
                    "description": (
                        "Suricata: Alert - ET POLICY GNU/Linux APT "
                        "User-Agent Outbound likely related to package management"
                    ),
                    "groups": ["ids", "suricata"],
                },
                "decoder": {"name": "json"},
                "input": {"type": "log"},
                "@timestamp": "1970-01-01T00:00:00.150000Z",
                "location": "/var/log/suricata/eve.json",
                "id": "9999999999.999999",
            },
        ],
    )
    return source_root


def _write_equal_timestamp_fixture_source_tree(tmp_path: Path) -> Path:
    source_root = tmp_path / "ait_ads_equal_timestamp_source"
    source_root.mkdir()
    (source_root / "labels.csv").write_text(
        (
            "scenario,attack,start,end\n"
            "fox,service_stop,0.0,1000.0\n"
            "harrison,service_stop,0.0,1000.0\n"
        ),
        encoding="utf-8",
    )
    for scenario, entity in (("fox", "172.17.129.140"), ("harrison", "172.17.129.141")):
        _write_jsonl(
            source_root / f"{scenario}_aminer.json",
            [
                {
                    "AnalysisComponent": {
                        "AnalysisComponentIdentifier": 3,
                        "AnalysisComponentType": "NewMatchPathDetector",
                        "AnalysisComponentName": f"AMiner: {scenario} event.",
                        "Message": "New path(es) detected",
                        "PersistenceFileName": "nmpd",
                    },
                    "LogData": {
                        "RawLogData": [scenario],
                        "Timestamps": [0.1],
                        "DetectionTimestamp": [0.1],
                        "LogLinesCount": 1,
                        "LogResources": ["/var/log/audit/audit.log"],
                    },
                    "AMiner": {"ID": entity},
                },
            ],
        )
        _write_jsonl(
            source_root / f"{scenario}_wazuh.json",
            [
                {
                    "agent": {
                        "ip": "10.35.32.1",
                        "name": "wazuh-client",
                        "id": "29",
                    },
                    "manager": {"name": "wazuh.manager"},
                    "rule": {
                        "id": "86601",
                        "level": 3,
                        "description": f"{scenario} Wazuh alert",
                        "groups": ["ids", "suricata"],
                    },
                    "decoder": {"name": "json"},
                    "input": {"type": "log"},
                    "@timestamp": "1970-01-01T00:00:00.100000Z",
                    "location": "/var/log/suricata/eve.json",
                    "id": "1000000000.000001",
                },
            ],
        )
    return source_root


def test_load_ait_ads_label_windows_sorts_non_overlapping_intervals(
    tmp_path: Path,
) -> None:
    """AIT-ADS label windows should be sorted by time before use."""
    source_root = _write_fixture_source_tree(tmp_path)

    windows = load_ait_ads_label_windows(source_root / "labels.csv")

    assert [window.attack for window in windows["fox"]] == [
        "service_stop",
        "dnsteal",
        "network_scans",
    ]


def test_materialise_ait_ads_alert_stream_assigns_labels_and_sorts(
    tmp_path: Path,
) -> None:
    """The canonical AIT-ADS stream should sort alerts and apply half-open labels."""
    source_root = _write_fixture_source_tree(tmp_path)
    raw_logs_path = tmp_path / "preprocessed" / "ait_ads_alerts.jsonl"
    raw_logs_path.parent.mkdir(parents=True, exist_ok=True)

    materialise_ait_ads_alert_stream(
        source_root=source_root,
        labels_path=source_root / "labels.csv",
        raw_logs_path=raw_logs_path,
        scenarios=("fox",),
    )

    records = [
        json.loads(line)
        for line in raw_logs_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["ids_source"] for record in records] == [
        "aminer",
        "suricata",
        "wazuh",
        "aminer",
    ]
    assert [record["timestamp_unix_ms"] for record in records] == [
        1642410287000,
        1642473488500,
        1642473540000,
        1642507140000,
    ]
    assert records[0]["attack_phase"] == "service_stop"
    assert records[1]["attack_phase"] is None
    assert records[2]["attack_phase"] is None
    assert records[3]["attack_phase"] == "network_scans"
    assert records[0]["anomalous"] == 1
    assert records[1]["anomalous"] == 0
    assert records[2]["anomalous"] == 0
    assert records[3]["anomalous"] == 1
    assert records[1]["template_key"].startswith("suricata|signature_id=")
    assert records[2]["template_key"].startswith("wazuh|rule_id=")
    assert records[3]["template_key"].startswith("aminer|type=")


def test_materialise_ait_ads_alert_stream_uses_half_open_end_boundaries(
    tmp_path: Path,
) -> None:
    """AIT-ADS label windows should stay half-open at the end boundary."""
    source_root = _write_boundary_fixture_source_tree(tmp_path)
    raw_logs_path = tmp_path / "preprocessed" / "ait_ads_alerts.jsonl"
    raw_logs_path.parent.mkdir(parents=True, exist_ok=True)

    materialise_ait_ads_alert_stream(
        source_root=source_root,
        labels_path=source_root / "labels.csv",
        raw_logs_path=raw_logs_path,
        scenarios=("fox",),
    )

    records = [
        json.loads(line)
        for line in raw_logs_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["timestamp_unix_ms"] for record in records] == [
        150,
        200,
    ]
    assert [record["anomalous"] for record in records] == [1, 0]
    assert records[0]["original_timestamp"] == "1970-01-01T00:00:00.150000Z"
    assert records[1]["original_timestamp"] == "0.2"
    assert records[0]["attack_phase"] == "service_stop"
    assert records[1]["attack_phase"] is None


def test_materialise_ait_ads_alert_stream_orders_equal_timestamps_stably(
    tmp_path: Path,
) -> None:
    """AIT-ADS sorting should stay deterministic when timestamps tie."""
    source_root = _write_equal_timestamp_fixture_source_tree(tmp_path)
    raw_logs_path = tmp_path / "preprocessed" / "ait_ads_alerts.jsonl"
    raw_logs_path.parent.mkdir(parents=True, exist_ok=True)

    materialise_ait_ads_alert_stream(
        source_root=source_root,
        labels_path=source_root / "labels.csv",
        raw_logs_path=raw_logs_path,
        scenarios=("fox", "harrison"),
    )

    records = [
        json.loads(line)
        for line in raw_logs_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["scenario"] for record in records] == [
        "fox",
        "fox",
        "harrison",
        "harrison",
    ]
    assert [record["ids_source"] for record in records] == [
        "aminer",
        "suricata",
        "aminer",
        "suricata",
    ]
    assert [record["timestamp_unix_ms"] for record in records] == [
        100,
        100,
        100,
        100,
    ]


def test_materialise_ait_ads_alert_stream_sorts_across_scenarios(
    tmp_path: Path,
) -> None:
    """AIT-ADS materialisation should preserve global chronology across scenarios."""
    source_root = _write_multi_scenario_fixture_source_tree(tmp_path)
    raw_logs_path = tmp_path / "preprocessed" / "ait_ads_alerts.jsonl"
    raw_logs_path.parent.mkdir(parents=True, exist_ok=True)

    materialise_ait_ads_alert_stream(
        source_root=source_root,
        labels_path=source_root / "labels.csv",
        raw_logs_path=raw_logs_path,
        scenarios=("fox", "harrison"),
    )

    records = [
        json.loads(line)
        for line in raw_logs_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [record["scenario"] for record in records] == [
        "harrison",
        "fox",
        "harrison",
        "fox",
        "fox",
        "fox",
    ]
    assert [record["timestamp_unix_ms"] for record in records] == [
        1642410286500,
        1642410287000,
        1642473487000,
        1642473488500,
        1642473540000,
        1642507140000,
    ]


def test_ait_ads_source_uses_local_dir_sources_for_tests(tmp_path: Path) -> None:
    """AITADSScenarioSource should be reusable with a local extracted fixture."""
    source_root = _write_fixture_source_tree(tmp_path)
    dataset_root = tmp_path / "materialised"
    source = AITADSScenarioSource(
        scenario_names=("fox",),
        base_source=LocalDirSource(path=source_root),
        labels_md5_checksum="8bc6f42a54490b43c2a0c6e7fb7532cf",
    )

    materialised_root = source.materialise(dst_dir=dataset_root)

    assert materialised_root == source_root
    raw_logs_path = source.raw_logs_path(
        dataset_name="AIT_ADS_FOX",
        dataset_root=materialised_root,
    )
    assert raw_logs_path.exists()
