"""DeepCASE paper-readiness audit helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from experiments import ConfigError
from experiments.config import (
    DatasetVariantConfig,
    EntitySequenceConfig,
)
from experiments.models.deepcase.detector import DeepCaseModelConfig

if TYPE_CHECKING:
    from experiments.models.base import ExperimentModelConfig


_DEEPCASE_CONTEXT_LENGTH = 10
_DEEPCASE_TIMEOUT_SECONDS = 86_400
_DEEPCASE_HIDDEN_SIZE = 128
_DEEPCASE_LABEL_SMOOTHING_DELTA = 0.1
_DEEPCASE_CONFIDENCE_THRESHOLD = 0.2
_DEEPCASE_EPS = 0.1
_DEEPCASE_MIN_SAMPLES = 5
_DEEPCASE_EPOCHS = 100
_DEEPCASE_ITERATIONS = 100
_DEEPCASE_TRAIN_FRACTION = 0.2
_DEEPCASE_TEST_FRACTION = 0.8


def validate_deepcase_hdfs_table_iv_config(
    *,
    dataset_config: DatasetVariantConfig,
    model_config: ExperimentModelConfig | None,
) -> None:
    """Validate the HDFS DeepCASE Table IV reproduction contract.

    Args:
        dataset_config (DatasetVariantConfig): HDFS dataset configuration to
            validate.
        model_config (ExperimentModelConfig | None): DeepCASE model
            configuration to validate.
    """
    _validate_deepcase_hdfs_dataset_config(dataset_config=dataset_config)
    _validate_deepcase_paper_model_config(model_config=model_config)


def validate_deepcase_hdfs_table_x_config(
    *,
    dataset_config: DatasetVariantConfig,
    model_config: ExperimentModelConfig | None,
) -> None:
    """Validate the HDFS DeepCASE Table X reproduction contract.

    Args:
        dataset_config (DatasetVariantConfig): HDFS dataset configuration to
            validate.
        model_config (ExperimentModelConfig | None): DeepCASE model
            configuration to validate.
    """
    _validate_deepcase_hdfs_dataset_config(dataset_config=dataset_config)
    _validate_deepcase_paper_model_config(model_config=model_config)


def validate_deepcase_bgl_extension_config(
    *,
    dataset_config: DatasetVariantConfig,
    model_config: ExperimentModelConfig | None,
) -> None:
    """Validate the BGL DeepCASE extension contract.

    Args:
        dataset_config (DatasetVariantConfig): BGL dataset configuration to
            validate.
        model_config (ExperimentModelConfig | None): DeepCASE model
            configuration to validate.
    """
    _validate_deepcase_bgl_extension_dataset_config(dataset_config=dataset_config)
    _validate_deepcase_paper_model_config(model_config=model_config)


def _validate_deepcase_paper_model_config(
    *,
    model_config: ExperimentModelConfig | None,
) -> None:
    if model_config is None:
        msg = "DeepCASE paper configs require a model config."
        raise ConfigError(msg)
    if not isinstance(model_config, DeepCaseModelConfig):
        msg = "DeepCASE paper configs must use the DeepCASE model."
        raise TypeError(msg)
    _require_equal(
        model_config.context_length,
        _DEEPCASE_CONTEXT_LENGTH,
        "DeepCASE paper configs must use context_length = 10.",
    )
    _require_close(
        model_config.timeout_seconds,
        _DEEPCASE_TIMEOUT_SECONDS,
        "DeepCASE paper configs must use timeout_seconds = 86400.",
    )
    _require_equal(
        model_config.hidden_size,
        _DEEPCASE_HIDDEN_SIZE,
        "DeepCASE paper configs must use hidden_size = 128.",
    )
    _require_close(
        model_config.label_smoothing_delta,
        _DEEPCASE_LABEL_SMOOTHING_DELTA,
        "DeepCASE paper configs must use label_smoothing_delta = 0.1.",
    )
    _require_close(
        model_config.confidence_threshold,
        _DEEPCASE_CONFIDENCE_THRESHOLD,
        "DeepCASE paper configs must use confidence_threshold = 0.2.",
    )
    _require_close(
        model_config.eps,
        _DEEPCASE_EPS,
        "DeepCASE paper configs must use eps = 0.1.",
    )
    _require_equal(
        model_config.min_samples,
        _DEEPCASE_MIN_SAMPLES,
        "DeepCASE paper configs must use min_samples = 5.",
    )
    _require_equal(
        model_config.epochs,
        _DEEPCASE_EPOCHS,
        "DeepCASE paper configs must use epochs = 100.",
    )
    _require_equal(
        model_config.iterations,
        _DEEPCASE_ITERATIONS,
        "DeepCASE paper configs must use iterations = 100.",
    )


def _validate_deepcase_hdfs_dataset_config(
    *,
    dataset_config: DatasetVariantConfig,
) -> None:
    sequence = dataset_config.sequence
    if not isinstance(sequence, EntitySequenceConfig):
        msg = "HDFS DeepCASE paper configs must use entity grouping."
        raise TypeError(msg)
    _require_close(
        sequence.train_fraction,
        _DEEPCASE_TRAIN_FRACTION,
        "HDFS DeepCASE paper configs must use train_fraction = 0.2.",
    )
    _require_close(
        sequence.test_fraction,
        _DEEPCASE_TEST_FRACTION,
        "HDFS DeepCASE paper configs must use test_fraction = 0.8.",
    )
    if sequence.train_on_normal_entities_only:
        msg = (
            "HDFS DeepCASE paper configs must not restrict training to normal "
            "entities only."
        )
        raise ValueError(msg)


def _validate_deepcase_bgl_extension_dataset_config(
    *,
    dataset_config: DatasetVariantConfig,
) -> None:
    sequence = dataset_config.sequence
    if not isinstance(sequence, EntitySequenceConfig):
        msg = "BGL DeepCASE extension configs must use entity grouping."
        raise TypeError(msg)
    _require_close(
        sequence.train_fraction,
        _DEEPCASE_TRAIN_FRACTION,
        "BGL DeepCASE extension configs must use train_fraction = 0.2.",
    )
    _require_close(
        sequence.test_fraction,
        _DEEPCASE_TEST_FRACTION,
        "BGL DeepCASE extension configs must use test_fraction = 0.8.",
    )
    if sequence.train_on_normal_entities_only:
        msg = (
            "BGL DeepCASE extension configs must not restrict training to "
            "normal entities only."
        )
        raise ValueError(msg)


def _require_equal(
    actual: int | str,
    expected: int | str,
    message: str,
) -> None:
    """Raise when two scalar audit values differ.

    Args:
        actual (int | str): Observed configuration value.
        expected (int | str): Paper-aligned configuration value.
        message (str): Error message to raise on mismatch.

    Raises:
        ValueError: If the values do not match.
    """
    if actual != expected:
        raise ValueError(message)


_FLOAT_TOLERANCE = 1e-9


def _require_close(actual: float, expected: float, message: str) -> None:
    """Raise when two floating-point audit values differ materially.

    Args:
        actual (float): Observed configuration value.
        expected (float): Paper-aligned configuration value.
        message (str): Error message to raise on mismatch.

    Raises:
        ValueError: If the values differ by more than the configured
            tolerance.
    """
    if abs(actual - expected) > _FLOAT_TOLERANCE:
        raise ValueError(message)
