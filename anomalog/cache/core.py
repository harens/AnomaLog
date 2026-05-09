"""Cache utilities and Prefect helpers for AnomaLog flows."""

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any, ParamSpec, TypedDict, TypeVar

from filelock import FileLock
from platformdirs import user_cache_dir, user_data_dir
from prefect import task as _task
from prefect.assets import Asset, AssetProperties
from prefect.assets import materialize as _prefect_materialize
from prefect.cache_policies import INPUTS, TASK_SOURCE, CacheKeyFnPolicy
from typing_extensions import Unpack

from anomalog.cache.classes import cache_class_key_fn

from .files import _ALLOWED, AssetDepsFingerprintPolicy

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class CachePathsConfig:
    """Resolved locations for data and cache storage.

    The project keeps source materialisation under `data_root` and derived,
    reproducible build artifacts under `cache_root`. Carrying both roots together
    makes dataset-scoped cleanup and path derivation deterministic across the
    builder, runtime, and experiment layers.

    Attributes:
        data_root (Path): Root directory for raw or materialised dataset inputs.
        cache_root (Path): Root directory for derived local artifacts and Prefect
            cache storage.
    """

    data_root: Path = field(default_factory=lambda: Path(user_data_dir("anomalog")))
    cache_root: Path = field(default_factory=lambda: Path(user_cache_dir("anomalog")))


class MaterializeTaskOptions(TypedDict, total=False):
    """Task options accepted by the local materialisation helper.

    Attributes:
        name (str | None): Optional task name.
        description (str | None): Optional task description.
        tags (list[str] | None): Optional task tags.
        version (str | None): Optional task version.
        cache_key_fn (Any): Optional cache key function override.
        cache_expiration (Any): Optional cache expiration policy.
        task_run_name (Any): Optional task run name override.
        retries (int | None): Retry count.
        retry_delay_seconds (float | int | list[float] | Any): Retry delay
            policy.
        retry_jitter_factor (float | None): Retry jitter factor.
        result_storage (Any): Result storage override.
        result_serializer (Any): Result serializer override.
        result_storage_key (str | None): Result storage key.
        cache_result_in_memory (bool): Whether to keep cached results in memory.
        timeout_seconds (int | float | None): Task timeout.
        log_prints (bool | None): Whether print statements are logged.
        refresh_cache (bool | None): Whether to ignore cached results.
        on_completion (Any): Completion hooks.
        on_failure (Any): Failure hooks.
        on_running (Any): Running hooks.
        on_rollback (Any): Rollback hooks.
        on_commit (Any): Commit hooks.
        retry_condition_fn (Any): Retry condition callback.
        viz_return_value (Any): Value surfaced to the visualiser.
    """

    name: str | None
    description: str | None
    tags: list[str] | None
    version: str | None
    cache_key_fn: Any
    cache_expiration: Any
    task_run_name: Any
    retries: int | None
    retry_delay_seconds: float | int | list[float] | Any
    retry_jitter_factor: float | None
    result_storage: Any
    result_serializer: Any
    result_storage_key: str | None
    cache_result_in_memory: bool
    timeout_seconds: int | float | None
    log_prints: bool | None
    refresh_cache: bool | None
    on_completion: Any
    on_failure: Any
    on_running: Any
    on_rollback: Any
    on_commit: Any
    retry_condition_fn: Any
    viz_return_value: Any


def _unique_dataset_roots(
    dataset_name: str,
    cache_paths: CachePathsConfig,
) -> tuple[Path, ...]:
    roots = (
        cache_paths.data_root / dataset_name,
        cache_paths.cache_root / dataset_name,
    )
    return tuple(dict.fromkeys(roots))


def clear_dataset_cache(
    dataset_name: str,
    *,
    cache_paths: CachePathsConfig,
) -> None:
    """Delete all local cached artifacts for a dataset.

    This removes the dataset source materialisation under `data_root` and all
    derived dataset-scoped artifacts under `cache_root`.

    Args:
        dataset_name (str): Dataset identifier whose cached artifacts should be
            removed.
        cache_paths (CachePathsConfig): Resolved data and cache root locations.

    Raises:
        ValueError: If `dataset_name` is empty.
    """
    if not dataset_name:
        msg = "clear_dataset_cache() requires a non-empty dataset name."
        raise ValueError(msg)

    for root in _unique_dataset_roots(dataset_name, cache_paths):
        if root.is_dir():
            shutil.rmtree(root)
            continue
        if root.exists():
            root.unlink()


def dataset_build_lock_path(
    dataset_name: str,
    *,
    cache_paths: CachePathsConfig,
) -> Path:
    """Return the coarse dataset-build lock path for one cache namespace.

    The dataset build pipeline writes shared source and derived artifacts under
    `<data_root>/<dataset_name>` and `<cache_root>/<dataset_name>`. Builds that
    target that namespace must therefore be serialised even when they differ in
    parser or templating configuration, because the on-disk outputs would still
    collide.

    Args:
        dataset_name (str): Dataset identifier whose build namespace should be
            locked.
        cache_paths (CachePathsConfig): Resolved data and cache root locations.

    Returns:
        Path: Lock file path scoped to the dataset/cache namespace.

    Raises:
        ValueError: If `dataset_name` is empty.
    """
    if not dataset_name:
        msg = "dataset_build_lock_path() requires a non-empty dataset name."
        raise ValueError(msg)

    resolved_data_root = cache_paths.data_root.expanduser().resolve()
    resolved_cache_root = cache_paths.cache_root.expanduser().resolve()
    namespace = (
        f"{dataset_name}\n"
        f"{resolved_data_root.as_posix()}\n"
        f"{resolved_cache_root.as_posix()}"
    )
    lock_digest = sha256(namespace.encode("utf-8")).hexdigest()[:16]
    safe_dataset_name = _ALLOWED.sub("_", dataset_name).strip("_") or "dataset"
    lock_dir = resolved_cache_root / "dataset_build_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / f"{safe_dataset_name}-{lock_digest}.lock"


def dataset_build_lock(
    dataset_name: str,
    *,
    cache_paths: CachePathsConfig,
) -> FileLock:
    """Return the coarse cross-process lock for one dataset build namespace.

    Args:
        dataset_name (str): Dataset identifier whose build namespace should be
            locked.
        cache_paths (CachePathsConfig): Resolved data and cache root locations.

    Returns:
        FileLock: Lock guarding all dataset build work for that namespace.
    """
    return FileLock(dataset_build_lock_path(dataset_name, cache_paths=cache_paths))


def asset_from_local_path(path: Path) -> Asset:
    """Create a Prefect Asset from a local filesystem path.

    - Asset key is a sanitised identifier derived from the path
    - Real path is stored in Asset.properties.url
    - Deterministic: same path -> same key

    Args:
        path (Path): Local filesystem path to expose as a Prefect asset.

    Examples:
        >>> asset = asset_from_local_path(Path("/tmp/demo.txt"))
        >>> asset.properties.url.endswith("/tmp/demo.txt")
        True

    Returns:
        Asset: Asset describing the resolved local filesystem path.
    """
    path = path.expanduser().resolve()

    # Build a deterministic, Prefect-safe key
    # NOTE: this is an IDENTIFIER, not a real path
    safe_key = _ALLOWED.sub("_", path.as_posix())

    return Asset(
        key=f"localfs://{safe_key}",
        properties=AssetProperties(
            name=path.name,
            url=path.as_uri(),
        ),
    )


# TODO(harens): Allow users to set this, move into CachePathsConfig
CACHE_POLICY = (
    INPUTS
    + TASK_SOURCE
    + AssetDepsFingerprintPolicy()
    + CacheKeyFnPolicy(cache_key_fn=cache_class_key_fn)
).configure(key_storage=CachePathsConfig().cache_root / "prefect")

task = partial(_task, persist_result=True, cache_policy=CACHE_POLICY)


def materialize(
    output_path: Path,
    *,
    asset_deps: list[Asset | str] | None = None,
    **task_kwargs: Unpack[MaterializeTaskOptions],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap Prefect materialsation with a local output existence check.

    Prefect can reuse a cached completed state without re-checking whether the
    local output path still exists. This helper reruns the wrapped function
    directly if the expected local file or directory is missing after Prefect
    returns, or if Prefect returns a stale cached result whose stored file
    path no longer matches the active local storage base.

    Args:
        output_path (Path): Local path that must exist after Prefect returns.
        asset_deps (list[Asset | str] | None): Upstream asset dependencies for
            the wrapped task materialisation.
        **task_kwargs (Unpack[MaterializeTaskOptions]): Additional Prefect task
            options forwarded to `prefect.assets.materialise`.

    Returns:
        Callable[[Callable[P, R]], Callable[P, R]]: Decorator that materialises
            the wrapped function and falls back to rerunning it when the local
            output path is missing or the cached result is no longer readable.
    """

    def _decorate(func: Callable[P, R]) -> Callable[P, R]:
        materialized = _prefect_materialize(
            asset_from_local_path(output_path),
            persist_result=True,
            cache_policy=CACHE_POLICY,
            asset_deps=asset_deps,
            **task_kwargs,
        )(func)

        def _run(*args: P.args, **kwargs: P.kwargs) -> R:
            try:
                result = materialized(*args, **kwargs)
            except ValueError as exc:
                if not _is_stale_materialize_cache_error(exc):
                    raise
                return func(*args, **kwargs)
            if output_path.exists():
                return result
            return func(*args, **kwargs)

        return _run

    return _decorate


def _is_stale_materialize_cache_error(exc: ValueError) -> bool:
    """Return whether Prefect rejected a cached result from another base path.

    Prefect local filesystem result storage raises `ValueError` when a cached
    result points outside the active storage base path. That can happen after a
    checkout moves or when old cache metadata survives a storage-root change.

    Args:
        exc (ValueError): Exception raised while reading the cached result.

    Returns:
        bool: Whether the error should trigger a direct rerun of the wrapped
            function.
    """
    return "outside of the base path" in str(exc)
