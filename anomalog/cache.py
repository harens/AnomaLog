import re
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from platformdirs import user_cache_dir, user_data_dir
from prefect import task as _task
from prefect.assets import Asset, AssetProperties
from prefect.assets import materialize as _materialize
from prefect.cache_policies import INPUTS, TASK_SOURCE, CachePolicy
from prefect.context import AssetContext, TaskRunContext
from prefect.utilities.hashing import hash_objects


@dataclass(frozen=True, slots=True)
class CachePathsConfig:
    data_root: Path = field(default_factory=lambda: Path(user_data_dir("anomalog")))
    cache_root: Path = field(default_factory=lambda: Path(user_cache_dir("anomalog")))


_ALLOWED = re.compile(r"[^A-Za-z0-9._/-]")


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


def _try_file_path_from_asset_url(asset_url: str | None) -> Path | None:
    """Accepts URLs like:
      - "file:///abs/path/to/data.csv"
      - "file://localhost/abs/path/to/data.csv"  (common variant).

    Returns a local Path if it looks like a file URI; otherwise None.

    Notes:
    - Prefect asset keys cannot contain spaces or '%' so you should not depend on keys
      being reversible file URIs. Use Asset.properties.url (or Asset.url if present).
    - We URL-decode percent escapes (e.g. %20) so paths with spaces work.

    """
    if not asset_url:
        return None

    parsed = urlparse(asset_url)
    if parsed.scheme != "file":
        return None

    # urlparse("file:///...") -> path="/..."
    # urlparse("file://localhost/...") -> netloc="localhost", path="/..."
    # Decode %XX escapes (e.g. %20 -> space)
    path = unquote(parsed.path)
    if not path:
        return None

    return Path(path)


def _file_fingerprint(path: Path) -> tuple[str, int, int, int]:
    """(exists_flag, mtime_ns, size_bytes, inode).

    inode helps detect atomic-save editors that replace the file.
    """
    try:
        st = path.stat()
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        ino = int(getattr(st, "st_ino", 0))
        return ("1", mtime_ns, int(st.st_size), ino)
    except FileNotFoundError:
        return ("0", 0, 0, 0)


def _asset_file_path(asset: Asset) -> Path | None:
    """Resolve a local file Path from an Asset by looking at properties.url (preferred)
    or asset.url (fallback). Returns None for non-file assets / missing urls.
    """
    asset_url = None
    props = getattr(asset, "properties", None)
    if props is not None:
        asset_url = getattr(props, "url", None)
    if not asset_url:
        asset_url = getattr(asset, "url", None)

    if not asset_url:
        return None

    parsed = urlparse(asset_url)
    if parsed.scheme != "file":
        return None

    path = unquote(parsed.path)
    return Path(path) if path else None


@dataclass
class AssetDepsFingerprintPolicy(CachePolicy):
    """Cache key component based on:
      - upstream asset dependencies (upstream_assets + direct_asset_dependencies)
      - downstream produced assets (downstream_assets), if they exist.

    Rationale:
      - If outputs drift (deleted/modified), cache key changes
        -> task reruns -> repairs outputs.
    """

    include_outputs: bool = True

    def compute_key(
        self,
        task_ctx: TaskRunContext,  # noqa: ARG002 - not used, but part of the interface
        inputs: dict[str, Any],  # noqa: ARG002 - not used, but part of the interface
        flow_parameters: dict[str, Any],  # noqa: ARG002 - not used, but part of the interface
        **kwargs: object,  # noqa: ARG002 - not used, but part of the interface
    ) -> str | None:
        asset_ctx = AssetContext.get()
        if not asset_ctx:
            return None

        upstream = asset_ctx.upstream_assets | asset_ctx.direct_asset_dependencies

        if not upstream:
            return None

        payload = []

        for asset in sorted(upstream, key=lambda a: a.key):
            p = _asset_file_path(asset)
            if p is None:
                payload.append(("upstream", asset.key, ("nonfile",)))
            else:
                payload.append(("upstream", asset.key, _file_fingerprint(p)))

        return hash_objects(payload, raise_on_failure=True)


# TODO(harens): Allow users to set this
CACHE_POLICY = INPUTS + TASK_SOURCE + AssetDepsFingerprintPolicy()

task = partial(_task, persist_result=True, cache_policy=CACHE_POLICY)
materialize = partial(_materialize, persist_result=True, cache_policy=CACHE_POLICY)
