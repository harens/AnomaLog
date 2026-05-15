"""Built-in dataset presets exposed through the public builder API."""

from __future__ import annotations

from functools import partial
from pathlib import Path

from anomalog.dataset import DatasetSpec
from anomalog.labels import CSVReader
from anomalog.parsers import (
    AITADSParser,
    BGLParser,
    HDFSV1Parser,
    OpenStackDeepLogParser,
)
from anomalog.parsers.structured import DelimitedLabelledEventParser
from anomalog.parsers.template import IdentityTemplateParser, SpellTemplateParser
from anomalog.sources import AITADSScenarioSource, PostProcessedSource, RemoteZipSource
from anomalog.sources.ait_ads import AIT_ADS_SCENARIOS
from anomalog.sources.deeplog_preprocessed import (
    materialise_labelled_raw_stream,
    materialise_labelled_session_stream,
)

# See https://github.com/logpai/loghub/issues/61
# Datasets could have mistakes in labeling.

# See LogHub: https://zenodo.org/records/8196385
# Originally tried using LogHub-2.0 (https://zenodo.org/record/8275861),
# but HDFS does not appear to be annotated there.

hdfs_v1 = (
    DatasetSpec("HDFS_V1")
    .from_source(
        RemoteZipSource(
            url="https://zenodo.org/records/8196385/files/HDFS_v1.zip",
            md5_checksum="76a24b4d9a6164d543fb275f89773260",
            raw_logs_relpath=Path("HDFS.log"),
        ),
    )
    .parse_with(HDFSV1Parser())
    .label_with(
        CSVReader(
            relative_path=Path("preprocessed/anomaly_label.csv"),
            entity_column="BlockId",
            label_column="Label",
            anomalous_value="Anomaly",
            normal_value="Normal",
        ),
    )
)

bgl = (
    DatasetSpec("BGL")
    .from_source(
        RemoteZipSource(
            url="https://zenodo.org/records/8196385/files/BGL.zip",
            md5_checksum="4452953c470f2d95fcb32d5f6e733f7a",
            raw_logs_relpath=Path("BGL.log"),
        ),
    )
    .parse_with(BGLParser())
)

hdfs_wuyifan18_deeplog_preprocessed = (
    DatasetSpec("HDFS_WUYIFAN18_DEEPLOG_PREPROCESSED")
    .from_source(
        PostProcessedSource(
            base_source=RemoteZipSource(
                url="https://github.com/wuyifan18/DeepLog/archive/refs/heads/master.zip",
            ),
            post_process=partial(
                materialise_labelled_session_stream,
                split_files=(
                    ("hdfs_train", 0),
                    ("hdfs_test_normal", 0),
                    ("hdfs_test_abnormal", 1),
                ),
            ),
            raw_logs_relpath=Path("preprocessed/hdfs_events.log"),
        ),
    )
    .parse_with(DelimitedLabelledEventParser())
    .template_with(IdentityTemplateParser)
)

openstack_deeplog_preprocessed = (
    DatasetSpec("OPENSTACK_DEEPLOG_PREPROCESSED")
    .from_source(
        PostProcessedSource(
            base_source=RemoteZipSource(
                url="https://zenodo.org/records/8196385/files/OpenStack.tar.gz",
                md5_checksum="66bd42c07837a094d9b0ea2d036b5713",
            ),
            post_process=partial(
                materialise_labelled_raw_stream,
                split_files=(
                    ("openstack_normal1.log", "openstack_train", 0),
                    ("openstack_normal2.log", "openstack_test_normal", 0),
                    ("openstack_abnormal.log", "openstack_test_abnormal", 1),
                ),
            ),
            raw_logs_relpath=Path("preprocessed/openstack_labelled_raw.log"),
        ),
    )
    .parse_with(OpenStackDeepLogParser())
    .template_with(SpellTemplateParser)
)


def _ait_ads_preset_name(scenario: str) -> str:
    return f"ait_ads_{scenario}"


def _ait_ads_dataset_name(scenario: str) -> str:
    return f"AIT_ADS_{scenario.upper()}"


def _build_ait_ads_preset(scenario: str) -> DatasetSpec:
    return (
        DatasetSpec(_ait_ads_dataset_name(scenario))
        .from_source(AITADSScenarioSource((scenario,)))
        .parse_with(AITADSParser())
        .template_with(IdentityTemplateParser)
    )


_AIT_ADS_PRESETS = {
    _ait_ads_preset_name(scenario): _build_ait_ads_preset(scenario)
    for scenario in AIT_ADS_SCENARIOS
}

_PRESETS: dict[str, DatasetSpec] = {
    "bgl": bgl,
    "hdfs_v1": hdfs_v1,
    "hdfs_wuyifan18_deeplog_preprocessed": hdfs_wuyifan18_deeplog_preprocessed,
    "openstack_deeplog_preprocessed": openstack_deeplog_preprocessed,
    **_AIT_ADS_PRESETS,
}


def resolve_preset(name: str) -> DatasetSpec:
    """Resolve a built-in dataset preset by name.

    Args:
        name (str): Registered preset name to resolve.

    Returns:
        DatasetSpec: Registered preset dataset spec.

    Raises:
        KeyError: If `name` does not match a built-in preset.
    """
    try:
        return _PRESETS[name]
    except KeyError as exc:
        msg = f"Unsupported preset: {name!r}"
        raise KeyError(msg) from exc


def preset_names() -> tuple[str, ...]:
    """Return the registered built-in preset names.

    Returns:
        tuple[str, ...]: Preset names in registration order.
    """
    return tuple(_PRESETS)
