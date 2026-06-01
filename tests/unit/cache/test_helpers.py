"""Additional tests for cache helper functions."""

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path
from time import monotonic, sleep

import pytest
from filelock import Timeout
from prefect.assets import Asset, AssetProperties
from prefect.logging import disable_run_logger

from anomalog.cache import CachePathsConfig, asset_from_local_path, materialize
from anomalog.cache import core as cache_core
from anomalog.cache import files as cache_files
from anomalog.cache.core import dataset_build_lock, dataset_build_lock_path
from anomalog.cache.core import task as cache_task
from anomalog.cache.files import AssetDepsFingerprintPolicy
from tests.unit.helpers import task_run_context

ZeroArgFn = Callable[[], str]
MaterializeDecorator = Callable[[ZeroArgFn], ZeroArgFn]


@dataclass(frozen=True)
class _AssetContext:
    direct_asset_dependencies: list[Asset]


@dataclass(frozen=True)
class _GroupedFailureError(RuntimeError):
    """Minimal group-like wrapper for stale-cache regression tests.

    Attributes:
        message (str): Human-readable wrapper message.
        exceptions (tuple[BaseException, ...]): Wrapped failures exposed via a
            group-like attribute so the cache helper can recurse through them.
    """

    message: str
    exceptions: tuple[BaseException, ...]

    def __post_init__(self) -> None:
        super().__init__(self.message)


class _FallbackAsset(Asset):
    url: str


class _MissingUrlAsset(Asset):
    url: str | None = None


@dataclass(frozen=True)
class _BasepathResultStorage:
    basepath: str | Path | None


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

    monkeypatch.setattr("anomalog.cache.core._prefect_materialize", _skip_materialize)

    @materialize(output_path)
    def _build() -> str:
        output_path.write_text("hello", encoding="utf-8")
        return "rebuilt"

    with disable_run_logger():
        assert _build() == "rebuilt"
    assert output_path.read_text(encoding="utf-8") == "hello"


def test_materialize_reruns_function_when_prefect_cached_result_is_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local-output materialization should recover from stale Prefect cache paths.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local outputs.
        monkeypatch (pytest.MonkeyPatch): Replaces Prefect materialization with a
            cache-hit stub that simulates a stale local filesystem result.
    """

    def _raise_stale_path_error(
        *_args: object,
        **_kwargs: object,
    ) -> Callable[[ZeroArgFn], ZeroArgFn]:
        def _decorate(_func: ZeroArgFn) -> ZeroArgFn:
            def _raise() -> str:
                message = (
                    "Provided path /old/run/storage is outside of the base path "
                    "/new/run/storage."
                )
                raise ValueError(message)

            return _raise

        return _decorate

    output_path = tmp_path / "artifact.txt"
    monkeypatch.setattr(
        "anomalog.cache.core._prefect_materialize",
        _raise_stale_path_error,
    )

    @materialize(output_path)
    def _build() -> str:
        output_path.write_text("hello", encoding="utf-8")
        return "rebuilt"

    with disable_run_logger():
        assert _build() == "rebuilt"
    assert output_path.read_text(encoding="utf-8") == "hello"


def test_materialize_reruns_function_when_prefect_cached_result_is_stale_in_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Local-output materialisation should unwrap grouped stale cache failures.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for local outputs.
        monkeypatch (pytest.MonkeyPatch): Replaces Prefect materialization with a
            cache-hit stub that simulates a wrapped stale filesystem result.
    """

    def _raise_stale_path_group(
        *_args: object,
        **_kwargs: object,
    ) -> Callable[[ZeroArgFn], ZeroArgFn]:
        def _decorate(_func: ZeroArgFn) -> ZeroArgFn:
            def _raise() -> str:
                message = (
                    "Provided path /old/run/storage is outside of the base path "
                    "/new/run/storage."
                )
                group_message = "task run failed"
                raise _GroupedFailureError(
                    group_message,
                    (
                        RuntimeError("Task run failed with exception"),
                        ValueError(message),
                    ),
                )

            return _raise

        return _decorate

    output_path = tmp_path / "artifact.txt"
    monkeypatch.setattr(
        "anomalog.cache.core._prefect_materialize",
        _raise_stale_path_group,
    )

    @materialize(output_path)
    def _build() -> str:
        output_path.write_text("hello", encoding="utf-8")
        return "rebuilt"

    with disable_run_logger():
        assert _build() == "rebuilt"
    assert output_path.read_text(encoding="utf-8") == "hello"


def test_materialize_and_task_use_shared_result_storage_base(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prefect result storage should be pinned to the shared cache namespace.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for the fabricated asset.
        monkeypatch (pytest.MonkeyPatch): Replaces Prefect materialisation so
            the test can inspect the forwarded task options directly.
    """
    expected_basepath = CachePathsConfig().cache_root / "prefect" / "storage"
    result_storage = cache_task.keywords["result_storage"]

    assert Path(result_storage) == expected_basepath

    captured: dict[str, object] = {}

    def _capture_materialize(
        *_args: object,
        **kwargs: object,
    ) -> Callable[[ZeroArgFn], ZeroArgFn]:
        captured.update(kwargs)

        def _decorate(func: ZeroArgFn) -> ZeroArgFn:
            return func

        return _decorate

    monkeypatch.setattr(
        "anomalog.cache.core._prefect_materialize",
        _capture_materialize,
    )

    @materialize(tmp_path / "demo.txt")
    def _build() -> str:
        return "demo"

    _build()

    captured_result_storage = captured["result_storage"]
    assert isinstance(captured_result_storage, (str, Path))
    assert Path(captured_result_storage) == expected_basepath


def test_cache_paths_config_uses_cluster_root_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment overrides should redirect the default cache roots.

    Args:
        tmp_path (Path): Temporary directory used to host the synthetic root
            paths.
        monkeypatch (pytest.MonkeyPatch): Environment helper used to inject
            cluster roots.
    """
    monkeypatch.setenv("ANOMALOG_DATA_ROOT", (tmp_path / "data").as_posix())
    monkeypatch.setenv("ANOMALOG_CACHE_ROOT", (tmp_path / "cache").as_posix())

    cache_paths = CachePathsConfig()

    assert cache_paths.data_root == tmp_path / "data"
    assert cache_paths.cache_root == tmp_path / "cache"


def test_result_storage_cache_policy_changes_with_basepath(tmp_path: Path) -> None:
    """Result storage moves should force Prefect to miss stale cached states.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for fabricated result
            storage roots.
    """
    build_cache_policy = vars(cache_core)["_cache_policy_for_result_storage"]
    first_policy = build_cache_policy(tmp_path / "one" / "storage")
    second_policy = build_cache_policy(tmp_path / "two" / "storage")

    first_key = first_policy.compute_key(task_run_context(), {}, {})
    second_key = second_policy.compute_key(task_run_context(), {}, {})

    assert first_key is not None
    assert second_key is not None
    assert first_key != second_key


def test_resolve_result_storage_basepath_prefers_explicit_basepath(
    tmp_path: Path,
) -> None:
    """Result-storage objects with a basepath should resolve that path directly.

    Args:
        tmp_path (Path): Per-test filesystem sandbox used to fabricate the
            storage base path.
    """
    resolve_result_storage_basepath = vars(cache_core)[
        "_resolve_result_storage_basepath"
    ]
    storage = _BasepathResultStorage(basepath=tmp_path / "storage")

    assert resolve_result_storage_basepath(storage) == (tmp_path / "storage").resolve()


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
    deadline = monotonic() + 15
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


def test_dataset_build_lock_allows_reentrant_acquisition_for_same_namespace(
    tmp_path: Path,
) -> None:
    """The namespace lock should be reusable within one build flow.

    Args:
        tmp_path (Path): Per-test filesystem sandbox for cache namespace paths.
    """
    cache_paths = CachePathsConfig(
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
    )

    with ExitStack() as stack:
        stack.enter_context(dataset_build_lock("demo", cache_paths=cache_paths))
        stack.enter_context(dataset_build_lock("demo", cache_paths=cache_paths))
        assert True
