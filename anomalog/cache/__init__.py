"""Cache utilities and Prefect helpers for AnomaLog flows."""

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypeVar

from platformdirs import user_cache_dir, user_data_dir
from prefect import task as _task
from prefect.assets import Asset, AssetProperties
from prefect.assets import materialize as _prefect_materialize
from prefect.cache_policies import INPUTS, TASK_SOURCE, CacheKeyFnPolicy
from typing_extensions import Unpack

from anomalog.cache.classes import cache_class_key_fn

from .files import _ALLOWED, AssetDepsFingerprintPolicy

if TYPE_CHECKING:
    from prefect.tasks import TaskOptions

P = ParamSpec("P")
R = TypeVar("R")


@dataclass(frozen=True, slots=True)
class CachePathsConfig:
    """Resolved locations for data and cache storage."""

    data_root: Path = field(default_factory=lambda: Path(user_data_dir("anomalog")))
    cache_root: Path = field(default_factory=lambda: Path(user_cache_dir("anomalog")))


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

    This removes the dataset source materialization under `data_root` and all
    derived dataset-scoped artifacts under `cache_root`.
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


def asset_from_local_path(path: Path) -> Asset:
    """Create a Prefect Asset from a local filesystem path.

    - Asset key is a sanitized identifier derived from the path
    - Real path is stored in Asset.properties.url
    - Deterministic: same path -> same key

    >>> asset = asset_from_local_path(Path("/tmp/demo.txt"))
    >>> asset.properties.url.endswith("/tmp/demo.txt")
    True
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
    asset_deps: list[Asset] | None = None,
    **task_kwargs: Unpack["TaskOptions"],
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Wrap Prefect materialization with a local output existence check.

    Prefect can reuse a cached completed state without re-checking whether the
    local output path still exists. This helper reruns the wrapped function
    directly if the expected local file or directory is missing after Prefect
    returns.
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
            result = materialized(*args, **kwargs)
            if output_path.exists():
                return result
            return func(*args, **kwargs)

        return _run

    return _decorate
