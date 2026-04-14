"""Registry helpers for experiment model configs."""

from experiments.models.base import ExperimentModelConfig
from experiments.models.naive_bayes import NaiveBayesModelConfig
from experiments.models.river import RiverModelConfig
from experiments.models.template_frequency import TemplateFrequencyModelConfig

_MODEL_CONFIG_TYPES: dict[str, type[ExperimentModelConfig]] = {
    "naive_bayes": NaiveBayesModelConfig,
    "river": RiverModelConfig,
    "template_frequency": TemplateFrequencyModelConfig,
}


def resolve_model_config_type(name: str) -> type[ExperimentModelConfig]:
    """Resolve a built-in model-config type by detector name.

    Args:
        name (str): Registered detector name.

    Returns:
        type[ExperimentModelConfig]: Registered config type for the detector.

    Raises:
        KeyError: If `name` does not match a built-in detector.
    """
    try:
        return _MODEL_CONFIG_TYPES[name]
    except KeyError as exc:
        msg = f"Unsupported detector: {name!r}"
        raise KeyError(msg) from exc


def model_names() -> tuple[str, ...]:
    """Return supported built-in detector/model names.

    Returns:
        tuple[str, ...]: Detector names in registration order.
    """
    return tuple(_MODEL_CONFIG_TYPES)
