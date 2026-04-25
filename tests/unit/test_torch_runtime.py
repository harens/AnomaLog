"""Tests for shared torch experiment runtime helpers."""

from __future__ import annotations

import pytest
import torch

from experiments.models.torch_runtime import resolve_torch_device


class _AvailableMpsBackend:
    """Tiny stand-in for ``torch.backends.mps`` in device-selection tests."""

    @staticmethod
    def is_available() -> bool:
        """Return that the fake MPS backend is available.

        Returns:
            bool: Always `True` so tests can exercise the MPS branch deterministically.
        """
        return True


class _UnavailableMpsBackend:
    """Tiny stand-in for an unavailable ``torch.backends.mps`` backend."""

    @staticmethod
    def is_available() -> bool:
        """Return that the fake MPS backend is unavailable.

        Returns:
            bool: Always `False` so tests can exercise fallback/error branches
                deterministically.
        """
        return False


def test_resolve_torch_device_auto_prefers_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto device selection should prefer CUDA when it is available.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces torch backend availability
            probes to make accelerator priority deterministic.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends, "mps", _AvailableMpsBackend())

    device = resolve_torch_device("auto")

    assert device.type == "cuda"


# This locks down accelerator preference order even on machines whose
# integration tests already exercise the MPS branch through auto-selection.
@pytest.mark.allow_no_new_coverage
def test_resolve_torch_device_auto_uses_mps_after_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto device selection should use MPS when CUDA is unavailable.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces torch backend availability
            probes to make accelerator priority deterministic.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", _AvailableMpsBackend())

    device = resolve_torch_device("auto")

    assert device.type == "mps"


def test_resolve_torch_device_auto_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto device selection should fall back to CPU without accelerators.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces torch backend availability
            probes to make accelerator priority deterministic.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends, "mps", _UnavailableMpsBackend())

    device = resolve_torch_device("auto")

    assert device.type == "cpu"


def test_resolve_torch_device_rejects_unavailable_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit MPS selection should fail when MPS is unavailable.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces torch backend availability to
            force the explicit-MPS error branch.
    """
    monkeypatch.setattr(torch.backends, "mps", _UnavailableMpsBackend())

    with pytest.raises(ValueError, match="MPS is not available"):
        resolve_torch_device("mps")
