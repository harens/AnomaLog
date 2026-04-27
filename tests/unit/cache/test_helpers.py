"""Additional tests for cache helper functions."""

from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from time import monotonic, sleep

import pytest
from filelock import Timeout
from prefect.assets import Asset, AssetProperties
from prefect.logging import disable_run_logger

from anomalog.cache import CachePathsConfig, asset_from_local_path, materialize
from anomalog.cache import files as cache_files
from anomalog.cache.core import dataset_build_lock, dataset_build_lock_path
from anomalog.cache.files import AssetDepsFingerprintPolicy
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


def _hold_dataset_build_lock(
    dataset_name: str,
    data_root: Path,
    cache_root: Path,
    ready_path: Path,
    release_path: Path,
) -> None:
    cache_paths = CachePathsConfig(data_root=data_root, cache_root=cache_root)
    with dataset_build_lock(dataset_name, cache_paths=cache_paths):
        ready_path.touch()
        while not release_path.exists():
            sleep(0.01)


def test_try_file_path_from_asset_url_decodes_localhost_and_spaces() -> None:
    """File URLs should decode path segments and accept the localhost variant."""
    try_file_path_from_asset_url = vars(cache_files)["_try_file_path_from_asset_url"]
    path = try_file_path_from_asset_url("file://localhost/tmp/a%20b.txt")

    assert path is not None
    assert path.name == "a b.txt"
    assert path.parent.name == "tmp"


def test_asset_file_path_reads_properties_url_and_ignores_non_file_assets(
    tmp_path: Path,
) -> None:
    """Asset path resolution should only succeed for file-backed assets.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local asset paths.
    """
    asset_file_path = vars(cache_files)["_asset_file_path"]
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

    assert asset_file_path(asset) == local_path
    assert asset_file_path(fallback_asset) == local_path
    assert asset_file_path(missing_url_asset) is None
    assert asset_file_path(remote) is None


def test_asset_deps_fingerprint_policy_uses_placeholder_for_no_upstream_assets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty upstream set should still contribute a deterministic key.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces asset-context lookup for the
            duration of the test.
    """
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
    """Local file metadata should affect the fingerprint contribution.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local asset files.
        monkeypatch (pytest.MonkeyPatch): Replaces asset-context lookup for the
            duration of the test.
    """
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
    """Local-output materialization should recover from stale Prefect cache hits.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local outputs.
        monkeypatch (pytest.MonkeyPatch): Replaces Prefect materialization with a
            cache-hit stub so the fallback path can be exercised directly.
    """
    output_path = tmp_path / "artifact.txt"

    monkeypatch.setattr("anomalog.cache.materialize", _skip_materialize)

    @materialize(output_path)
    def _build() -> str:
        output_path.write_text("hello", encoding="utf-8")
        return "rebuilt"

    with disable_run_logger():
        assert _build() == "rebuilt"
    assert output_path.read_text(encoding="utf-8") == "hello"


def test_dataset_build_lock_path_changes_with_cache_namespace(tmp_path: Path) -> None:
    """Dataset build locks should be scoped to dataset name and cache roots.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for cache namespace paths.
    """
    first_cache_paths = CachePathsConfig(
        data_root=tmp_path / "data-a",
        cache_root=tmp_path / "cache-a",
    )
    second_cache_paths = CachePathsConfig(
        data_root=tmp_path / "data-b",
        cache_root=tmp_path / "cache-a",
    )

    first_path = dataset_build_lock_path("demo", cache_paths=first_cache_paths)
    second_path = dataset_build_lock_path("demo", cache_paths=second_cache_paths)

    assert first_path.parent == first_cache_paths.cache_root / "dataset_build_locks"
    assert first_path != second_path
    with pytest.raises(ValueError, match="non-empty dataset name"):
        dataset_build_lock_path("", cache_paths=first_cache_paths)


def test_dataset_build_lock_blocks_other_processes_for_same_namespace(
    tmp_path: Path,
) -> None:
    """Concurrent builds in one dataset namespace should serialize.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for cache namespace paths.
    """
    cache_paths = CachePathsConfig(
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
    )
    ready_path = tmp_path / "ready"
    release_path = tmp_path / "release"
    process = Process(
        target=_hold_dataset_build_lock,
        args=(
            "demo",
            cache_paths.data_root,
            cache_paths.cache_root,
            ready_path,
            release_path,
        ),
    )
    process.start()
    deadline = monotonic() + 5
    while monotonic() < deadline and not ready_path.exists():
        sleep(0.01)

    try:
        assert ready_path.exists()
        lock = dataset_build_lock("demo", cache_paths=cache_paths)

        with pytest.raises(Timeout), lock.acquire(timeout=0.05):
            pass
    finally:
        release_path.touch()
        process.join(timeout=5)

    assert process.exitcode == 0
