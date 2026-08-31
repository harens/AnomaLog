"""Tests for experiment result manifest helpers outside the unit harness."""

from dataclasses import dataclass
from pathlib import Path

import pytest

import experiments.results as experiment_results
from anomalog.cache import CachePathsConfig
from anomalog.labels import AnomalyLabelLookup
from anomalog.parsers.structured.contracts import BaseStructuredLine
from anomalog.parsers.template import IdentityTemplateParser, TemplatedDataset
from experiments.config_types import DatasetVariantConfig
from experiments.models.base import ModelManifest, ModelRunSummary, SequenceSummary
from tests.unit.helpers import (
    InMemoryStructuredSink,
    NullStructuredParser,
    structured_line,
)


@dataclass(frozen=True)
class _Bundle:
    dataset: DatasetVariantConfig


@dataclass(frozen=True)
class _Context:
    templated: TemplatedDataset
    model_summary: ModelRunSummary


def _make_context(
    *,
    raw_dataset_path: Path,
    rows: list[BaseStructuredLine],
    window_count: int = 4,
) -> _Context:
    sink = InMemoryStructuredSink(
        dataset_name="thunderbird",
        raw_dataset_path=raw_dataset_path,
        parser=NullStructuredParser(),
        rows=[],
    )
    sink.rows.extend(
        structured_line(
            line_order=line_order,
            timestamp_unix_ms=row.timestamp_unix_ms,
            entity_id=row.entity_id,
            untemplated_message_text=row.untemplated_message_text,
            anomalous=row.anomalous,
        )
        for line_order, row in enumerate(rows)
    )
    return _Context(
        templated=TemplatedDataset(
            sink=sink,
            cache_paths=CachePathsConfig(
                data_root=raw_dataset_path.parent,
                cache_root=raw_dataset_path.parent,
            ),
            template_parser=IdentityTemplateParser(dataset_name="thunderbird"),
            anomaly_labels=AnomalyLabelLookup(
                label_for_line=lambda _line_order: None,
                label_for_group=lambda _group_id: None,
            ),
        ),
        model_summary=ModelRunSummary(
            metrics={},
            model_manifest=ModelManifest(
                detector="test",
                train_sequence_count=1,
                test_sequence_count=2,
                train_label_counts={0: 1},
                test_label_counts={1: 1},
                ignored_sequence_count=1,
            ),
            sequence_summary=SequenceSummary(
                sequence_count=window_count,
                train_sequence_count=1,
                test_sequence_count=2,
                train_label_counts={0: 1},
                test_label_counts={1: 1},
                ignored_sequence_count=1,
            ),
        ),
    )


def test_build_dataset_statistics_keeps_combined_ait_ads_summary(
    tmp_path: Path,
) -> None:
    """AIT-ADS result manifests should still emit dataset-level statistics.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for the local fixture.
    """
    alerts = (
        '{"scenario":"fox","ids_source":"aminer","timestamp_unix_ms":1,"anomalous":0}',
        '{"scenario":"fox","ids_source":"wazuh","timestamp_unix_ms":2,"anomalous":1}',
        '{"scenario":"harrison","ids_source":"aminer","timestamp_unix_ms":3,"anomalous":0}',
    )
    raw_logs_path = tmp_path / "ait_ads_alerts.jsonl"
    raw_logs_path.write_text("\n".join(alerts) + "\n", encoding="utf-8")
    expected_total_alerts = len(alerts)
    expected_anomalous_alerts = sum(1 for alert in alerts if '"anomalous":1' in alert)
    context = _make_context(
        raw_dataset_path=raw_logs_path,
        rows=[],
    )

    stats = experiment_results._build_dataset_statistics(  # ruff: ignore[private-member-access]
        bundle=_Bundle(
            dataset=DatasetVariantConfig(
                name="ait_ads",
                dataset_name="AIT_ADS",
                preset="ait_ads",
            ),
        ),
        context=context,
    )

    assert stats is not None
    assert stats["total_alerts_parsed"] == expected_total_alerts
    assert stats["anomalous_alert_count"] == expected_anomalous_alerts
    assert stats["anomalous_alert_fraction"] == pytest.approx(
        expected_anomalous_alerts / expected_total_alerts,
    )
    assert stats["missing_timestamp_alert_count"] == 0
    assert stats["total_alerts_per_scenario"] == {
        "fox": 2,
        "harrison": 1,
    }
    assert stats["total_alerts_per_ids_source"] == {
        "aminer": 2,
        "wazuh": 1,
    }


def test_build_dataset_statistics_reports_thunderbird_parse_counts(
    tmp_path: Path,
) -> None:
    """Thunderbird manifests should expose parse and template vocabulary stats.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for the local fixture.
    """
    expected_total_lines = 4
    expected_total_events = 2
    expected_skipped_lines = 2
    expected_template_vocabulary_size = 2
    expected_sequence_window_count = 2
    expected_train_sequence_count = 1
    expected_test_sequence_count = 2
    expected_ignored_sequence_count = 1
    raw_logs_path = tmp_path / "Thunderbird.log"
    raw_logs_path.write_text(
        (
            "- 1131566461 2005.11.09 dn228 Nov 9 12:01:01 dn228/dn228 "
            "crond(pam_unix)[2915]: session closed for user root\n"
            "+ 1131566522 2005.11.09 dn228 Nov 9 12:02:02 dn228/dn228 "
            "sshd[1234]: disk failure on /dev/sda\n"
            "\n"
            "malformed line without Thunderbird header\n"
        ),
        encoding="utf-8",
    )
    context = _make_context(
        raw_dataset_path=raw_logs_path,
        rows=[
            BaseStructuredLine(
                timestamp_unix_ms=1_131_566_461_000,
                entity_id="dn228/dn228",
                untemplated_message_text="session closed for user root",
                anomalous=0,
            ),
            BaseStructuredLine(
                timestamp_unix_ms=1_131_566_522_000,
                entity_id="dn228/dn228",
                untemplated_message_text="disk failure on /dev/sda",
                anomalous=1,
            ),
        ],
        window_count=2,
    )

    stats = experiment_results._build_dataset_statistics(  # ruff: ignore[private-member-access]
        bundle=_Bundle(
            dataset=DatasetVariantConfig(
                name="thunderbird",
                dataset_name="THUNDERBIRD",
                preset="thunderbird",
            ),
        ),
        context=context,
    )

    assert stats is not None
    assert stats["total_lines_parsed"] == expected_total_lines
    assert stats["total_events_emitted"] == expected_total_events
    assert stats["normal_event_count"] == 1
    assert stats["anomalous_event_count"] == 1
    assert stats["anomalous_event_fraction"] == pytest.approx(0.5)
    assert stats["missing_timestamp_event_count"] == 0
    assert stats["skipped_line_count"] == expected_skipped_lines
    assert stats["skipped_line_reasons"] == {
        "blank": 1,
        "malformed": 1,
    }
    assert stats["template_vocabulary_size"] == expected_template_vocabulary_size
    assert stats["sequence_window_count"] == expected_sequence_window_count
    assert stats["train_sequence_count"] == expected_train_sequence_count
    assert stats["test_sequence_count"] == expected_test_sequence_count
    assert stats["ignored_sequence_count"] == expected_ignored_sequence_count


def test_ait_ads_suppresses_sequence_level_headline_metrics() -> None:
    """AIT-ADS should report alert-level metrics without sequence-level heads."""
    assert (
        experiment_results._should_emit_sequence_level_detection(  # ruff: ignore[private-member-access]
            DatasetVariantConfig(
                name="ait_ads",
                dataset_name="AIT_ADS",
                preset="ait_ads",
            ),
        )
        is False
    )
    assert (
        experiment_results._should_emit_sequence_level_detection(  # ruff: ignore[private-member-access]
            DatasetVariantConfig(
                name="bgl",
                dataset_name="BGL",
                preset="bgl",
            ),
        )
        is True
    )
