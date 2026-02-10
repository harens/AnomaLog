from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

from platformdirs import user_cache_dir, user_data_dir
from prefect import task as _task
from prefect.assets import Asset, AssetProperties
from prefect.assets import materialize as _materialize
from prefect.cache_policies import INPUTS, TASK_SOURCE, CacheKeyFnPolicy

from anomalog.cache.classes import cache_class_key_fn

from .files import _ALLOWED, AssetDepsFingerprintPolicy


@dataclass(frozen=True, slots=True)
class CachePathsConfig:
    data_root: Path = field(default_factory=lambda: Path(user_data_dir("anomalog")))
    cache_root: Path = field(default_factory=lambda: Path(user_cache_dir("anomalog")))


def asset_from_local_path(path: Path) -> Asset:
    """Create a Prefect Asset from a local filesystem path.

    - Asset key is a sanitized identifier derived from the path
    - Real path is stored in Asset.properties.url
    - Deterministic: same path -> same key
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
materialize = partial(_materialize, persist_result=True, cache_policy=CACHE_POLICY)
