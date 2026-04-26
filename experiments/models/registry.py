"""Registry helpers for experiment model configs."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import import_module

from experiments import ConfigError
from experiments.models.base import ExperimentModelConfig


@dataclass(frozen=True, slots=True)
class _ModelRegistration:
    """Lazy import metadata for one registered experiment detector.

    Attributes:
        module_path (str): Module path containing the detector config class.
        config_type_name (str): Attribute name of the detector config class.
    """

    module_path: str
    config_type_name: str


_MODEL_REGISTRATIONS: dict[str, _ModelRegistration] = {
    "deepcase": _ModelRegistration(
        module_path="experiments.models.deepcase",
        config_type_name="DeepCaseModelConfig",
    ),
    "deeplog": _ModelRegistration(
        module_path="experiments.models.deeplog",
        config_type_name="DeepLogModelConfig",
    ),
    "naive_bayes": _ModelRegistration(
        module_path="experiments.models.naive_bayes",
        config_type_name="NaiveBayesModelConfig",
    ),
    "river": _ModelRegistration(
        module_path="experiments.models.river",
        config_type_name="RiverModelConfig",
    ),
    "template_frequency": _ModelRegistration(
        module_path="experiments.models.template_frequency",
        config_type_name="TemplateFrequencyModelConfig",
    ),
}

_MODEL_INSTALL_HINTS: dict[str, str] = {
    "deepcase": "uv sync --extra experiments --extra deepcase",
    "deeplog": "uv sync --extra experiments --extra deeplog",
    "river": "uv sync --extra experiments --extra river",
}


@cache
def resolve_model_config_type(name: str) -> type[ExperimentModelConfig]:
    """Resolve a built-in model-config type by detector name.

    Args:
        name (str): Registered detector name.

    Returns:
        type[ExperimentModelConfig]: Registered config type for the detector.

    Raises:
        KeyError: If `name` does not match a built-in detector.
        ConfigError: If the detector module is present but its optional
            backend is not installed.
        ModuleNotFoundError: If the detector module fails to import for a
            reason unrelated to the registered optional dependencies.
    """
    try:
        registration = _MODEL_REGISTRATIONS[name]
    except KeyError as exc:
        msg = f"Unsupported detector: {name!r}"
        raise KeyError(msg) from exc

    try:
        module = import_module(registration.module_path)
    except ModuleNotFoundError as exc:
        install_hint = _MODEL_INSTALL_HINTS.get(name)
        if install_hint is not None:
            msg = (
                f"Detector {name!r} requires optional backend dependencies. "
                f"Install them with `{install_hint}`."
            )
            raise ConfigError(msg) from exc
        raise

    config_type = getattr(module, registration.config_type_name)
    if not issubclass(config_type, ExperimentModelConfig):
        msg = (
            f"{registration.module_path}.{registration.config_type_name} is not "
            "a valid experiment model config."
        )
        raise ConfigError(msg)
    return config_type


def model_names() -> tuple[str, ...]:
    """Return supported built-in detector/model names.

    Returns:
        tuple[str, ...]: Detector names in registration order.
    """
    return tuple(_MODEL_REGISTRATIONS)
