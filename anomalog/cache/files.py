"""File-based cache helpers for Prefect assets."""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias
from urllib.parse import unquote, urlparse

from prefect.assets import Asset
from prefect.cache_policies import CachePolicy
from prefect.context import AssetContext, TaskRunContext
from prefect.utilities.hashing import hash_objects
from typing_extensions import override

_ALLOWED = re.compile(r"[^A-Za-z0-9._/-]")
CachePolicyKwarg: TypeAlias = str | int | float | bool | None


def _try_file_path_from_asset_url(asset_url: str | None) -> Path | None:
    """Return a local Path if the URL looks like a file URI.

    Accepts URLs like:
    - "file:///abs/path/to/data.csv"
    - "file://localhost/abs/path/to/data.csv" (common variant).

    Notes:
    - Prefect asset keys cannot contain spaces or '%' so you should not depend on keys
      being reversible file URIs. Use Asset.properties.url (or Asset.url if present).
    - We URL-decode percent escapes (e.g. %20) so paths with spaces work.

    Examples:
        >>> _try_file_path_from_asset_url("file:///tmp/example.txt").name
        'example.txt'
        >>> _try_file_path_from_asset_url("http://example.com") is None
        True

    Args:
        asset_url (str | None): Asset URL to interpret as a local file URI.

    Returns:
        Path | None: Local path for file URIs, or `None` for non-file URLs.
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

    Args:
        path (Path): File path to fingerprint.

    Examples:
        >>> tmp = Path("/tmp/anomalog_fp.txt")
        >>> _ = tmp.write_text("hi")
        >>> fp = _file_fingerprint(tmp)
        >>> fp[0] == "1" and fp[2] == 2
        True
        >>> tmp.unlink()
        >>> _file_fingerprint(tmp)[0]  # missing file
        '0'

    Returns:
        tuple[str, int, int, int]: Existence flag, nanosecond mtime, byte size,
            and inode for the file path.
    """
    try:
        st = path.stat()
        mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))
        ino = int(getattr(st, "st_ino", 0))
        return ("1", mtime_ns, int(st.st_size), ino)
    except FileNotFoundError:
        return ("0", 0, 0, 0)


def _asset_file_path(asset: Asset) -> Path | None:
    """Resolve a local file path from an Asset URL.

    Returns None for non-file assets or missing URLs.

    Args:
        asset (Asset): Prefect asset whose file URL should be inspected.

    Examples:
        >>> from anomalog.cache import asset_from_local_path
        >>> asset = asset_from_local_path(Path("/tmp/example"))
        >>> _asset_file_path(asset).as_posix().endswith('/tmp/example')
        True

    Returns:
        Path | None: Local file path extracted from the asset, if available.
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
    """Cache key component based on upstream asset dependencies.

    Prefect's default cache policies do not include local file metadata for asset
    dependencies. This policy adds a deterministic fingerprint of the direct
    upstream assets so cache reuse breaks when an upstream file is replaced or
    edited in place.

    Attributes:
        include_outputs (bool): Prefect cache-policy compatibility flag retained
            on the policy instance. The current implementation fingerprints direct
            upstream dependencies regardless of this value.
    """

    include_outputs: bool = True

    @override
    def compute_key(
        self,
        task_ctx: TaskRunContext,
        inputs: dict[str, Any],
        flow_parameters: dict[str, Any],
        **kwargs: CachePolicyKwarg,
    ) -> str | None:
        """Fingerprint upstream assets (including file metadata) for cache keys.

        Args:
            task_ctx (TaskRunContext): Prefect task run context supplied by the
                cache-policy interface.
            inputs (dict[str, Any]): Task input mapping supplied by Prefect.
            flow_parameters (dict[str, Any]): Flow parameter mapping supplied by
                Prefect.
            **kwargs (CachePolicyKwarg): Additional cache-policy keyword data
                supplied by Prefect.

        Returns:
            str | None: Stable hash for direct asset dependencies, or `None` when
                no asset context is active.
        """
        del task_ctx, inputs, flow_parameters, kwargs
        asset_ctx = AssetContext.get()
        if not asset_ctx:
            return None

        upstream = asset_ctx.direct_asset_dependencies

        if not upstream:
            # Return a deterministic placeholder so the overall cache key
            # still incorporates other policies (INPUTS, TASK_SOURCE, etc.)
            return hash_objects([("no_upstream_assets",)], raise_on_failure=True)

        payload = []

        for asset in sorted(upstream, key=lambda a: a.key):
            p = _asset_file_path(asset)
            if p is None:
                payload.append(("upstream", asset.key, ("nonfile",)))
            else:
                payload.append(("upstream", asset.key, _file_fingerprint(p)))

        return hash_objects(payload, raise_on_failure=True)
