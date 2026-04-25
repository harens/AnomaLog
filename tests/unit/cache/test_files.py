"""Tests for file-based cache helpers."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from prefect.assets import Asset, AssetProperties

from anomalog.cache import asset_from_local_path
from anomalog.cache.files import AssetDepsFingerprintPolicy
from tests.unit.helpers import task_run_context


@dataclass(frozen=True)
class _AssetContext:
    direct_asset_dependencies: list[Asset]


def test_asset_deps_fingerprint_policy_returns_none_without_asset_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No asset context means no additional cache key contribution.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces asset-context lookup for the
            duration of the test.
    """
    monkeypatch.setattr("anomalog.cache.files.AssetContext.get", lambda: None)

    key = AssetDepsFingerprintPolicy().compute_key(
        task_run_context(),
        {},
        {},
    )

    assert key is None


def test_asset_deps_fingerprint_policy_handles_file_and_nonfile_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dependency order does not affect mixed file/non-file asset fingerprints.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local asset files.
        monkeypatch (pytest.MonkeyPatch): Replaces asset-context lookup for the
            duration of the test.
    """
    local_file = tmp_path / "input.txt"
    local_file.write_text("payload", encoding="utf-8")
    file_asset = asset_from_local_path(local_file)
    remote_asset = Asset(
        key="storage://bucket/object",
        properties=AssetProperties(url="s3://bucket/object"),
    )

    monkeypatch.setattr(
        "anomalog.cache.files.AssetContext.get",
        lambda: _AssetContext(direct_asset_dependencies=[remote_asset, file_asset]),
    )
    ordered_key = AssetDepsFingerprintPolicy().compute_key(
        task_run_context(),
        {},
        {},
    )

    monkeypatch.setattr(
        "anomalog.cache.files.AssetContext.get",
        lambda: _AssetContext(direct_asset_dependencies=[file_asset, remote_asset]),
    )
    reversed_key = AssetDepsFingerprintPolicy().compute_key(
        task_run_context(),
        {},
        {},
    )

    assert ordered_key is not None
    assert ordered_key == reversed_key
