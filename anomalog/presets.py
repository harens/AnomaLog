"""Built-in dataset presets exposed through the public builder API."""

from __future__ import annotations

from pathlib import Path

from anomalog.dataset import DatasetSpec
from anomalog.labels import CSVReader
from anomalog.parsers import BGLParser, HDFSV1Parser
from anomalog.sources import RemoteZipSource

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

_PRESETS: dict[str, DatasetSpec] = {
    "bgl": bgl,
    "hdfs_v1": hdfs_v1,
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
