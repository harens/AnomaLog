"""Experiment model runtime exports."""

from experiments.models.base import (
    ExperimentDetector,
    ExperimentModelConfig,
    ModelManifest,
    ModelRunSummary,
    PredictionOutcome,
    SequencePrediction,
    SequenceSummary,
)
from experiments.models.evaluate import run_model
from experiments.models.naive_bayes import NaiveBayesDetector, NaiveBayesModelConfig
from experiments.models.river import RiverDetector, RiverModelConfig
from experiments.models.template_frequency import (
    TemplateFrequencyDetector,
    TemplateFrequencyModelConfig,
)

_MODEL_CONFIG_TYPES: dict[str, type[ExperimentModelConfig]] = {
    "naive_bayes": NaiveBayesModelConfig,
    "river": RiverModelConfig,
    "template_frequency": TemplateFrequencyModelConfig,
}


def resolve_model_config_type(name: str) -> type[ExperimentModelConfig]:
    """Resolve a built-in model-config type by detector name."""
    try:
        return _MODEL_CONFIG_TYPES[name]
    except KeyError as exc:
        msg = f"Unsupported detector: {name!r}"
        raise KeyError(msg) from exc


def model_names() -> tuple[str, ...]:
    """Return supported built-in detector/model names."""
    return tuple(_MODEL_CONFIG_TYPES)


__all__ = [
    "ExperimentDetector",
    "ExperimentModelConfig",
    "ModelManifest",
    "ModelRunSummary",
    "NaiveBayesDetector",
    "NaiveBayesModelConfig",
    "PredictionOutcome",
    "RiverDetector",
    "RiverModelConfig",
    "SequencePrediction",
    "SequenceSummary",
    "TemplateFrequencyDetector",
    "TemplateFrequencyModelConfig",
    "model_names",
    "resolve_model_config_type",
    "run_model",
]
