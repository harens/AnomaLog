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
    class_sources: list[str] = []
    for v in params.values():
        # Skip functions/methods/modules/etc.
        if inspect.isroutine(v) or inspect.ismodule(v):
            continue

        cls = type(v)

        # Skip built-in classes like function/int/str/list/...
        if cls.__module__ == "builtins":
            continue

        class_sources.append(class_source(cls))
    combined = "\n".join(sorted(class_sources))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
