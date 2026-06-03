"""Guarded Thunderbird integration smoke test."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from anomalog.cache import CachePathsConfig
from anomalog.dataset import DatasetSpec
from anomalog.parsers import ThunderbirdParser
from anomalog.sources import LocalDirSource, RemoteZipSource
from anomalog.sources.raw_prefix import materialise_raw_log_prefix

THUNDERBIRD_URL = (
    "https://zenodo.org/records/8196385/files/Thunderbird.tar.gz?download=1"
)
THUNDERBIRD_MD5 = "0891b048df2919dc78c99c4428686b44"


@pytest.mark.skipif(
    os.environ.get("RUN_THUNDERBIRD_INTEGRATION") != "1",
    reason="Thunderbird integration smoke test is disabled by default.",
)
def test_thunderbird_prefix_smoke_writes_a_manifest(tmp_path: Path) -> None:
    """Thunderbird should download, prefix, parse, and group under a guard.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for the integration run.
    """
    source = RemoteZipSource(
        url=THUNDERBIRD_URL,
        md5_checksum=THUNDERBIRD_MD5,
        raw_logs_relpath=Path("Thunderbird.log"),
    )
    dataset_root = source.materialise(dst_dir=tmp_path / "Thunderbird")

    line_limit = 10_000_000 if os.environ.get("RUN_THUNDERBIRD_FULL") == "1" else 50_000
    prefix_path = tmp_path / "preprocessed" / "Thunderbird_prefix.log"
    materialise_raw_log_prefix(
        dataset_root,
        prefix_path,
        source_log_relpath=Path("Thunderbird.log"),
        line_limit=line_limit,
    )

    dataset = (
        DatasetSpec("THUNDERBIRD_PREFIX_SMOKE")
        .from_source(
            LocalDirSource(
                prefix_path.parent,
                raw_logs_relpath=Path(prefix_path.name),
            ),
        )
        .parse_with(ThunderbirdParser())
        .with_cache_paths(
            CachePathsConfig(
                data_root=tmp_path / "data",
                cache_root=tmp_path / ".cache",
            ),
        )
        .build()
    )
    sequences = list(dataset.group_by_chronological_stream(chunk_size=10_000))
    manifest = {
        "dataset_root": dataset_root.as_posix(),
        "prefix_path": prefix_path.as_posix(),
        "line_limit": line_limit,
        "source_url": THUNDERBIRD_URL,
        "source_md5": THUNDERBIRD_MD5,
        "parsed_rows": dataset.sink.count_rows(),
        "sequence_count": len(sequences),
        "anomalous_rows": sum(
            1
            for row in dataset.sink.iter_structured_lines()()
            if row.anomalous not in {None, 0}
        ),
    }
    manifest_path = tmp_path / "thunderbird_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    assert manifest_path.exists()
    assert manifest["parsed_rows"] > 0
    assert manifest["sequence_count"] > 0
