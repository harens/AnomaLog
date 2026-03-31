"""Helpers for hashing class definitions to build cache keys."""

import hashlib
import importlib
import inspect
from pathlib import Path
from typing import Any

from prefect.context import TaskRunContext


def class_source(cls: type) -> str:
    """Return source code string for a class or a fallback identifier.

    >>> class _Tmp:
    ...     def foo(self): ...
    ...
    >>> src = class_source(_Tmp)
    >>> isinstance(src, str) and len(src) > 0
    True
    """
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


def cache_class_key_fn(context: TaskRunContext, params: dict[str, Any]) -> str:  # noqa: ARG001 - context is not used, but part of the interface
    """Build a stable hash for cache keying based on class definitions.

    Builtins and routines are ignored, and the result is independent of
    parameter ordering.

    >>> class _A:
    ...     pass
    ...
    >>> class _B:
    ...     pass
    ...
    >>> first = cache_class_key_fn(None, {"a": _A(), "b": _B(), "n": 1, "fn": len})
    >>> second = cache_class_key_fn(None, {"b": _B(), "a": _A()})
    >>> first == second
    True
    >>> len(first)
    64
    """
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
