import hashlib
import importlib
import inspect
from pathlib import Path

from prefect.context import TaskRunContext


def class_source(cls: type) -> str:
    try:
        return inspect.getsource(cls)
    except OSError:
        # Fallback: hash the defining module's file contents (best-effort)
        mod = importlib.import_module(cls.__module__)
        file = getattr(mod, "__file__", None)
        return (
            Path(file).read_text(encoding="utf-8")
            if file
            else f"{cls.__module__}.{cls.__qualname__}"
        )


def cache_class_key_fn(context: TaskRunContext, params: dict[str, object]) -> str:  # noqa: ARG001 - context is not used, but part of the interface
    # If any classes are in the params, include their source code hash in the cache key
    class_hashes = []
    for v in params.values():
        if inspect.isclass(v):
            source = class_source(v)
            class_hashes.append(hashlib.sha256(source.encode("utf-8")).hexdigest())
    return hashlib.sha256("".join(class_hashes).encode("utf-8")).hexdigest()
