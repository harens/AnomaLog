"""Shared torch runtime helpers for experiment models."""

from __future__ import annotations

import random
from typing import Literal

import torch

TorchDeviceName = Literal["auto", "cpu", "cuda", "mps"]
TORCH_DEVICE_NAME_ORDER = ("auto", "cpu", "cuda", "mps")


def set_torch_seed(seed: int) -> None:
    """Set deterministic random seeds for torch-backed experiment models.

    Args:
        seed (int): Random seed used by Python and torch.
    """
    random.seed(seed)
    torch.manual_seed(seed)


def resolve_torch_device(device_name: str) -> torch.device:
    """Resolve a configured torch device name.

    Args:
        device_name (str): Configured device name.

    Returns:
        torch.device: Resolved torch device.

    Raises:
        ValueError: If the requested accelerator is unavailable or unsupported.
    """
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if _mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "cuda":
        if not torch.cuda.is_available():
            msg = "Requested device='cuda' but CUDA is not available."
            raise ValueError(msg)
        return torch.device("cuda")

    if device_name == "mps":
        if not _mps_is_available():
            msg = "Requested device='mps' but MPS is not available."
            raise ValueError(msg)
        return torch.device("mps")

    if device_name == "cpu":
        return torch.device("cpu")

    msg = f"Unsupported torch device: {device_name!r}."
    raise ValueError(msg)


def torch_device_names_display() -> str:
    """Return supported torch device names for config error messages.

    Returns:
        str: Device names in user-facing order.
    """
    return ", ".join(TORCH_DEVICE_NAME_ORDER)


def _mps_is_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return mps_backend is not None and torch.backends.mps.is_available()
