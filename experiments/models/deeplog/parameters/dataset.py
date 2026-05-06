"""Dataset construction for DeepLog parameter models.

This module turns the raw per-template time series into the exact
`history -> next vector` examples consumed by the parameter LSTM.

The important ideas live here:

- each template is modeled as its own ordered time series
- validation uses the temporal tail, not a random split
- history windows overlap exactly as they do at deployment time
- missing target positions are carried forward in `target_mask` so later
  training and scoring can ignore them
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import TYPE_CHECKING, TypeAlias

import torch
from torch import nn
from typing_extensions import override

from experiments.models.deeplog.parameters.schema import (
    RawParameterVector,
    normalisation_for_raw_series,
    normalize_vector,
    raw_parameter_vector_for_event,
)
from experiments.models.deeplog.shared import (
    EPSILON,
    MIN_TEMPORAL_PARAMETER_PAIRS,
    GaussianThreshold,
    NormalisationStats,
    ParameterFeatureSchema,
    training_event_mask_for_sequence,
)

ParameterSeriesSplit: TypeAlias = tuple[
    list[list[RawParameterVector]],
    list[list[RawParameterVector]],
    list[list[bool]],
    list[list[bool]],
]

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from anomalog.sequences import TemplateSequence


@dataclass(frozen=True, slots=True)
class ParameterTrainingPair:
    """One normalised `history -> next vector` example.

    `history_inputs` and `target` are the actual model inputs and outputs.
    `raw_target` and `target_mask` are retained so loss computation and
    anomaly scoring can ignore parameter positions that were not observed in
    the underlying log event.

    Attributes:
        history_inputs (list[list[float]]): Normalised history window.
        target (list[float]): Normalised target vector.
        raw_target (list[float]): Unnormalised observed target values.
        target_mask (list[bool]): Observation mask for the target vector.
    """

    history_inputs: list[list[float]]
    target: list[float]
    raw_target: list[float]
    target_mask: list[bool]


ParameterDatasetSplit = tuple[
    list[ParameterTrainingPair],
    list[ParameterTrainingPair],
    NormalisationStats,
]


def build_parameter_datasets(
    *,
    normal_sequences: Iterable[TemplateSequence],
    template: str,
    schema: ParameterFeatureSchema,
    history_size: int,
    validation_fraction: float,
) -> ParameterDatasetSplit:
    """Build train and validation examples for one template model.

    A reader should be able to understand the whole parameter data path from
    this function:

    1. extract the ordered subsequence of events for one template
    2. split that per-template series into a train prefix and validation tail
    3. fit normalisation on the train prefix only
    4. convert both splits into sliding-window training pairs while keeping
       per-event eligibility masks aligned with the raw series

    Args:
        normal_sequences (Iterable[TemplateSequence]): Normal training sequences.
        template (str): Template whose parameter stream is being modeled.
        schema (ParameterFeatureSchema): Stable feature schema for the template.
        history_size (int): Number of prior template events per example.
        validation_fraction (float): Fraction of examples reserved for validation.

    Returns:
        ParameterDatasetSplit: Train pairs, validation pairs, and normalisation.
    """
    raw_series: list[list[RawParameterVector]] = []
    raw_series_target_masks: list[list[bool]] = []
    for sequence in normal_sequences:
        sequence_target_mask = training_event_mask_for_sequence(sequence)
        vectors = [
            raw_parameter_vector_for_event(
                parameters=parameters,
                dt_prev_ms=dt_prev_ms,
                schema=schema,
            )
            for event_template, parameters, dt_prev_ms in sequence.events
            if event_template == template
        ]
        if vectors:
            eligible_target_mask = [
                sequence_target_mask[event_index]
                for event_index, (event_template, _, _) in enumerate(sequence.events)
                if event_template == template
            ]
            raw_series.append(vectors)
            raw_series_target_masks.append(eligible_target_mask)

    (
        train_raw_series,
        validation_raw_series,
        train_raw_series_target_masks,
        validation_raw_series_target_masks,
    ) = split_parameter_series_temporally(
        raw_series=raw_series,
        raw_series_target_masks=raw_series_target_masks,
        history_size=history_size,
        validation_fraction=validation_fraction,
    )
    normalisation = normalisation_for_raw_series(raw_series=train_raw_series)
    train_pairs = list(
        iter_parameter_pairs(
            raw_series=train_raw_series,
            raw_series_target_masks=train_raw_series_target_masks,
            normalisation=normalisation,
            history_size=history_size,
        ),
    )
    validation_pairs = list(
        iter_parameter_pairs(
            raw_series=validation_raw_series,
            raw_series_target_masks=validation_raw_series_target_masks,
            normalisation=normalisation,
            history_size=history_size,
        ),
    )
    if not train_pairs and not validation_pairs:
        return [], [], normalisation
    return train_pairs, validation_pairs, normalisation


def split_parameter_series_temporally(
    *,
    raw_series: list[list[RawParameterVector]],
    raw_series_target_masks: list[list[bool]],
    history_size: int,
    validation_fraction: float,
) -> ParameterSeriesSplit:
    """Split each template-specific series into train prefix and validation tail.

    The validation slice overlaps the training slice by `history_size` items on
    purpose. Those overlapping vectors are not validation targets; they are the
    recent history needed to predict the first held-out target, which is how
    DeepLog uses the model at deployment time.

    Args:
        raw_series (list[list[RawParameterVector]]): Per-sequence template vectors.
        raw_series_target_masks (list[list[bool]]): Target eligibility masks
            aligned with `raw_series`.
        history_size (int): Number of prior template events per example.
        validation_fraction (float): Fraction of examples reserved for validation.

    Returns:
        ParameterSeriesSplit:
            Train-prefix series, validation-tail series, and the aligned target
            eligibility masks for both splits.
    """
    train_raw_series: list[list[RawParameterVector]] = []
    validation_raw_series: list[list[RawParameterVector]] = []
    train_raw_series_target_masks: list[list[bool]] = []
    validation_raw_series_target_masks: list[list[bool]] = []

    for series, target_mask in zip(
        raw_series,
        raw_series_target_masks,
        strict=True,
    ):
        pair_count = len(series) - history_size
        if pair_count < MIN_TEMPORAL_PARAMETER_PAIRS:
            continue

        validation_pair_count = max(1, math.ceil(pair_count * validation_fraction))
        validation_pair_count = min(validation_pair_count, pair_count - 1)
        train_target_count = pair_count - validation_pair_count
        train_prefix_length = history_size + train_target_count

        train_raw_series.append(series[:train_prefix_length])
        validation_raw_series.append(series[train_prefix_length - history_size :])
        train_raw_series_target_masks.append(target_mask[:train_prefix_length])
        validation_raw_series_target_masks.append(
            target_mask[train_prefix_length - history_size :],
        )

    return (
        train_raw_series,
        validation_raw_series,
        train_raw_series_target_masks,
        validation_raw_series_target_masks,
    )


def iter_parameter_pairs(
    *,
    raw_series: list[list[RawParameterVector]],
    raw_series_target_masks: list[list[bool]],
    normalisation: NormalisationStats,
    history_size: int,
) -> Iterator[ParameterTrainingPair]:
    """Yield normalized parameter-model training pairs.

    Example with `history_size = 2`:

    - history `[v0, v1]` predicts target `v2`
    - history `[v1, v2]` predicts target `v3`

    That is the multivariate time-series formulation the paper uses for each
    template-specific parameter stream.

    Args:
        raw_series (list[list[RawParameterVector]]): Template-specific raw
            series.
        raw_series_target_masks (list[list[bool]]): Target eligibility masks
            aligned with `raw_series`.
        normalisation (NormalisationStats): Fitted normalisation statistics.
        history_size (int): Number of prior template events per example.

    Yields:
        ParameterTrainingPair: Normalised training pair for the template
            series.
    """
    for series, target_mask in zip(raw_series, raw_series_target_masks, strict=True):
        if len(series) <= history_size:
            continue
        normalized_series = [
            normalize_vector(vector=vector, normalisation=normalisation)
            for vector in series
        ]
        for start in range(len(series) - history_size):
            target_vector = series[start + history_size]
            if not target_mask[start + history_size]:
                continue
            if not any(target_vector.mask):
                continue
            yield ParameterTrainingPair(
                history_inputs=normalized_series[start : start + history_size],
                target=normalize_vector(
                    vector=target_vector,
                    normalisation=normalisation,
                ),
                raw_target=target_vector.values,
                target_mask=target_vector.mask,
            )


class _MaskedRegressionLoss(nn.Module):
    """Mean squared loss over observed regression targets only."""

    @override
    def forward(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_float = mask.to(dtype=outputs.dtype)
        squared_error = (outputs - targets) ** 2
        observed_error = squared_error * mask_float
        observed_count = mask_float.sum()
        if observed_count.item() == 0:
            return observed_error.sum() * 0.0
        return observed_error.sum() / observed_count


def masked_regression_loss(
    *,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Return MSE over observed target positions only.

    Args:
        outputs (torch.Tensor): Predicted target values.
        targets (torch.Tensor): Observed target values.
        mask (torch.Tensor): Observation mask for the target values.

    Returns:
        torch.Tensor: Mean squared error over observed positions only.
    """
    return _MaskedRegressionLoss()(outputs, targets, mask)


def masked_mse(
    *,
    observed: list[float],
    predicted: list[float],
    mask: list[bool],
) -> float:
    """Compute MSE over observed parameter positions only.

    Args:
        observed (list[float]): Observed target vector.
        predicted (list[float]): Predicted target vector.
        mask (list[bool]): Positions present in the original event.

    Returns:
        float: Mean squared error over observed positions.
    """
    squared_errors = [
        (obs - pred) ** 2
        for obs, pred, is_observed in zip(observed, predicted, mask, strict=True)
        if is_observed
    ]
    if not squared_errors:
        return 0.0
    return sum(squared_errors) / len(squared_errors)


def fit_gaussian_threshold(
    *,
    residuals: list[float],
    confidence: float,
) -> GaussianThreshold:
    """Fit Gaussian threshold bounds from validation residuals.

    Args:
        residuals (list[float]): Validation residuals.
        confidence (float): Desired Gaussian confidence level.

    Returns:
        GaussianThreshold: Fitted Gaussian statistics and acceptance bounds.
    """
    mean = sum(residuals) / len(residuals)
    variance = sum((residual - mean) ** 2 for residual in residuals) / len(residuals)
    stddev = math.sqrt(max(variance, EPSILON))
    z_score = NormalDist().inv_cdf((1.0 + confidence) / 2.0)
    return GaussianThreshold(
        mean=mean,
        stddev=stddev,
        lower_bound=max(0.0, mean - (z_score * stddev)),
        upper_bound=mean + (z_score * stddev),
    )
