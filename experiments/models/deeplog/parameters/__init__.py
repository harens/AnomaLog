"""DeepLog parameter-model pipeline.

This package breaks the parameter path into the same phases a human reader
uses to understand the algorithm:

1. infer a stable schema for each template
2. build normalised `history -> next vector` datasets
3. train one LSTM and Gaussian threshold per template
4. score inference-time events with those fitted models

The public imports here are the detector-facing entry points used by
`detector.py` and the existing tests. The lower-level modules keep the
implementation details close to their phase of the pipeline.
"""

from experiments.models.deeplog.parameters.dataset import (
    ParameterDatasetSplit,
    ParameterTrainingPair,
    RawParameterVector,
    build_parameter_datasets,
    fit_gaussian_threshold,
    masked_mse,
    masked_regression_loss,
)
from experiments.models.deeplog.parameters.schema import (
    build_parameter_schemas,
    parameter_covered_event_count,
    parameter_model_input_size,
    raw_parameter_vector_for_event,
)
from experiments.models.deeplog.parameters.scoring import (
    normalized_parameter_score,
    parameter_anomaly_score,
    score_parameter_sequence,
)
from experiments.models.deeplog.parameters.training import fit_parameter_models

__all__ = [
    "ParameterDatasetSplit",
    "ParameterTrainingPair",
    "RawParameterVector",
    "build_parameter_datasets",
    "build_parameter_schemas",
    "fit_gaussian_threshold",
    "fit_parameter_models",
    "masked_mse",
    "masked_regression_loss",
    "normalized_parameter_score",
    "parameter_anomaly_score",
    "parameter_covered_event_count",
    "parameter_model_input_size",
    "raw_parameter_vector_for_event",
    "score_parameter_sequence",
]
