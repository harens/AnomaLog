"""Tests for experiment result manifest helpers outside the unit harness."""

from dataclasses import dataclass
from pathlib import Path

import pytest

import experiments.results as experiment_results
from experiments.config_types import DatasetVariantConfig


@dataclass(frozen=True)
class _Bundle:
    dataset: DatasetVariantConfig


def test_build_dataset_statistics_keeps_combined_ait_ads_summary(
    tmp_path: Path,
) -> None:
    """AIT-ADS result manifests should still emit dataset-level statistics."""
    alerts = (
        '{"scenario":"fox","ids_source":"aminer","timestamp_unix_ms":1,"anomalous":0}',
        '{"scenario":"fox","ids_source":"wazuh","timestamp_unix_ms":2,"anomalous":1}',
        '{"scenario":"harrison","ids_source":"aminer","timestamp_unix_ms":3,"anomalous":0}',
    )
    raw_logs_path = tmp_path / "ait_ads_alerts.jsonl"
    raw_logs_path.write_text("\n".join(alerts) + "\n", encoding="utf-8")
    expected_total_alerts = len(alerts)
    expected_anomalous_alerts = sum(1 for alert in alerts if '"anomalous":1' in alert)

    stats = experiment_results._build_dataset_statistics(  # noqa: SLF001
        bundle=_Bundle(
            dataset=DatasetVariantConfig(
                name="ait_ads",
                dataset_name="AIT_ADS",
                preset="ait_ads",
            ),
        ),
        raw_logs_path=raw_logs_path,
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


def test_ait_ads_suppresses_sequence_level_headline_metrics() -> None:
    """AIT-ADS should report alert-level metrics without sequence-level heads."""
    assert (
        experiment_results._should_emit_sequence_level_detection(  # noqa: SLF001
            DatasetVariantConfig(
                name="ait_ads",
                dataset_name="AIT_ADS",
                preset="ait_ads",
            ),
        )
        is False
    )
    assert (
        experiment_results._should_emit_sequence_level_detection(  # noqa: SLF001
            DatasetVariantConfig(
                name="bgl",
                dataset_name="BGL",
                preset="bgl",
            ),
        )
        is True
    )
