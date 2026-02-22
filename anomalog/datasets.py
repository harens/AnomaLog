from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from prefect import flow

from anomalog.anomaly_label_reader import CSVReader
from anomalog.sources import RemoteZipSource
from anomalog.sources.raw_dataset import RawDataset
from anomalog.structured_parsers.parsers import BGLParser, HDFSV1Parser
from anomalog.template_parsers import Drain3Parser

if TYPE_CHECKING:  # pragma: no cover
    from anomalog.template_parsers.templated_dataset import TemplatedDataset

# See https://github.com/logpai/loghub/issues/61
# Datasets could have mistakes in labeling

# # See LogHub: https://zenodo.org/records/8196385
# # Originally tried using LogHub-2.0 (https://zenodo.org/record/8275861),
# # but HDFS doesn't seem to be annotated


# Raw dataset definitions
hdfs_v1 = RawDataset(
    dataset_name="HDFS_V1",
    raw_logs_relpath=Path("HDFS.log"),
    source=RemoteZipSource(
        url="https://zenodo.org/records/8196385/files/HDFS_v1.zip",
        md5_checksum="76a24b4d9a6164d543fb275f89773260",
    ),
    structured_parser=HDFSV1Parser(),
    anomaly_label_reader=CSVReader(
        relative_path=Path("preprocessed/anomaly_label.csv"),
        entity_column="BlockId",
        label_column="Label",
    ),
)

bgl = RawDataset(
    dataset_name="BGL",
    source=RemoteZipSource(
        url="https://zenodo.org/records/8196385/files/BGL.zip",
        md5_checksum="4452953c470f2d95fcb32d5f6e733f7a",
    ),
    structured_parser=BGLParser(),
)


@flow
def build_bgl_dataset() -> TemplatedDataset:
    """Fetch, parse, and template the BGL dataset."""

    return (
        bgl.fetch_if_needed()
        .extract_structured_components()
        .mine_templates_with(Drain3Parser("BGL"))
    )


@flow
def build_hdfs_v1_dataset() -> TemplatedDataset:
    """Fetch, parse, and template the HDFS v1 dataset."""

    return (
        hdfs_v1.fetch_if_needed()
        .extract_structured_components()
        .mine_templates_with(Drain3Parser("HDFS_V1"))
    )
