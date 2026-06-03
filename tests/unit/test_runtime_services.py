"""Tests for internal runtime service helpers."""

from __future__ import annotations

import importlib
import os
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch


def _load_runtime_services() -> ModuleType:
    return importlib.import_module("anomalog._runtime.services")


@pytest.mark.allow_no_new_coverage
def test_build_templated_dataset_initialises_prefect_startup_timeout(
    monkeypatch: MonkeyPatch,
) -> None:
    """Dataset builds should raise Prefect's ephemeral startup timeout locally.

    Args:
        monkeypatch (MonkeyPatch): Fixture used to isolate environment and
            runtime module behaviour.
    """
    runtime_services: Any = _load_runtime_services()
    monkeypatch.delenv(
        "PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS",
        raising=False,
    )
    monkeypatch.setattr(runtime_services, "flow", lambda func: func)
    monkeypatch.setattr(
        runtime_services,
        "dataset_build_lock",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        runtime_services,
        "_build_templated_dataset",
        lambda _request: "built",
    )

    result = runtime_services.build_templated_dataset(
        SimpleNamespace(
            dataset_name="demo",
            cache_paths=SimpleNamespace(),
        ),
    )

    assert result == "built"
    assert os.environ["PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS"] == "120"


@pytest.mark.allow_no_new_coverage
def test_build_templated_dataset_preserves_an_explicit_override(
    monkeypatch: MonkeyPatch,
) -> None:
    """An explicit Prefect timeout should not be overwritten by the helper.

    Args:
        monkeypatch (MonkeyPatch): Fixture used to isolate environment and
            runtime module behaviour.
    """
    runtime_services: Any = _load_runtime_services()
    monkeypatch.setenv("PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS", "900")
    monkeypatch.setattr(runtime_services, "flow", lambda func: func)
    monkeypatch.setattr(
        runtime_services,
        "dataset_build_lock",
        lambda *_args, **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        runtime_services,
        "_build_templated_dataset",
        lambda _request: "built",
    )

    result = runtime_services.build_templated_dataset(
        SimpleNamespace(
            dataset_name="demo",
            cache_paths=SimpleNamespace(),
        ),
    )

    assert result == "built"
    assert os.environ["PREFECT_SERVER_EPHEMERAL_STARTUP_TIMEOUT_SECONDS"] == "900"
