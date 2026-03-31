"""Additional tests for cache helper functions."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest
from prefect.assets import Asset, AssetProperties
from prefect.logging import disable_run_logger

from anomalog.cache import asset_from_local_path, materialize
from anomalog.cache.files import (
    AssetDepsFingerprintPolicy,
    _asset_file_path,
    _try_file_path_from_asset_url,
)
from tests.unit.helpers import task_run_context

ZeroArgFn = Callable[[], str]
MaterializeDecorator = Callable[[ZeroArgFn], ZeroArgFn]


@dataclass(frozen=True)
class _AssetContext:
    direct_asset_dependencies: list[Asset]


class _FallbackAsset(Asset):
    url: str


class _MissingUrlAsset(Asset):
    url: str | None = None


def _skip_materialize(*_args: object, **_kwargs: object) -> MaterializeDecorator:
    def _decorate(_func: ZeroArgFn) -> ZeroArgFn:
        def _skip() -> str:
            return "cached"

        return _skip

    return _decorate


def test_try_file_path_from_asset_url_decodes_localhost_and_spaces() -> None:
    """File URLs should decode path segments and accept the localhost variant."""
    path = _try_file_path_from_asset_url("file://localhost/tmp/a%20b.txt")

    assert path is not None
    assert path.name == "a b.txt"
    assert path.parent.name == "tmp"


def test_asset_file_path_reads_properties_url_and_ignores_non_file_assets(
    tmp_path: Path,
) -> None:
    """Asset path resolution should only succeed for file-backed assets."""
    local_path = tmp_path / "demo.txt"
    asset = asset_from_local_path(local_path)
    fallback_asset = _FallbackAsset(
        key="localfs://fallback",
        properties=AssetProperties(url=None),
        url=local_path.as_uri(),
    )
    missing_url_asset = _MissingUrlAsset(
        key="localfs://missing",
        properties=AssetProperties(url=None),
    )
    remote = Asset(
        key="s3://bucket/demo.txt",
        properties=AssetProperties(url="s3://bucket/demo.txt"),
    )

    assert _asset_file_path(asset) == local_path
    assert _asset_file_path(fallback_asset) == local_path
    assert _asset_file_path(missing_url_asset) is None
    assert _asset_file_path(remote) is None


def test_asset_deps_fingerprint_policy_uses_placeholder_for_no_upstream_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty upstream set should still contribute a deterministic key."""
    monkeypatch.setattr(
        "anomalog.cache.files.AssetContext.get",
        lambda: _AssetContext(direct_asset_dependencies=[]),
    )

    key = AssetDepsFingerprintPolicy().compute_key(
        task_run_context(),
        {},
        {},
    )

    assert key is not None


def test_asset_deps_fingerprint_policy_changes_when_local_file_metadata_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local file metadata should affect the fingerprint contribution."""
    local_file = tmp_path / "input.txt"
    local_file.write_text("first", encoding="utf-8")
    asset = asset_from_local_path(local_file)
    remote = Asset(
        key="s3://bucket/demo.txt",
        properties=AssetProperties(url="s3://bucket/demo.txt"),
    )
    monkeypatch.setattr(
        "anomalog.cache.files.AssetContext.get",
        lambda: _AssetContext(direct_asset_dependencies=[remote, asset]),
    )

    first = AssetDepsFingerprintPolicy().compute_key(
        task_run_context(),
        {},
        {},
    )
    local_file.unlink()
    second = AssetDepsFingerprintPolicy().compute_key(
        task_run_context(),
        {},
        {},
    )

    assert first is not None
    assert second is not None
    assert first != second


def test_materialize_reruns_function_when_output_path_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local-output materialization should recover from stale Prefect cache hits."""
    output_path = tmp_path / "artifact.txt"

    monkeypatch.setattr("anomalog.cache.materialize", _skip_materialize)

    @materialize(output_path)
    def _build() -> str:
        output_path.write_text("hello", encoding="utf-8")
        return "rebuilt"

    with disable_run_logger():
        assert _build() == "rebuilt"
    assert output_path.read_text(encoding="utf-8") == "hello"
