"""Schema extraction and vector normalisation for DeepLog parameters.

This module answers the first question in the parameter pipeline:
"for a given template, what numeric vector are we actually modeling?"

The paper assumes each log key has a parameter-value vector. In a real parsed
dataset, that requires a few concrete choices:

- which parameter positions are stable enough to model?
- whether elapsed time should be included for this template
- how missing values should be represented without becoming fake signal
- how to normalise each feature before training the LSTM
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from experiments.models.deeplog.shared import (
    DT_FEATURE_NAME,
    EPSILON,
    NormalisationStats,
    ParameterFeatureSchema,
    ParameterModelState,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anomalog.sequences import TemplateSequence


@dataclass(frozen=True, slots=True)
class RawParameterVector:
    """One unnormalised parameter vector plus its observation mask.

    `values` always has the schema's fixed width. Missing positions are filled
    with `0.0` only to preserve tensor shape. `mask` records which positions
    were truly observed so normalisation, loss, and residual scoring can ignore
    the padded placeholders.

    Attributes:
        values (list[float]): Raw feature values, with missing positions filled
            by `0.0`.
        mask (list[bool]): Observation mask for the raw feature vector.
    """

    values: list[float]
    mask: list[bool]


def build_parameter_schemas(
    *,
    normal_sequences: Iterable[TemplateSequence],
    include_elapsed_time: bool,
    all_templates: Iterable[str] | None = None,
) -> dict[str, ParameterFeatureSchema]:
    """Infer stable numeric feature schemas per template.

    The paper talks about one parameter-value vector per log key. In this
    repository, "stable vector" means:

    - include `dt_prev_ms` only if enabled and actually observed in normal data
    - include a parameter position only if every observed value at that
      position is numeric for that template
    - drop mixed numeric/non-numeric positions entirely so the model does not
      learn from an unstable feature definition

    Args:
        normal_sequences (Iterable[TemplateSequence]): Normal training sequences.
        include_elapsed_time (bool): Whether elapsed time may become a feature.
        all_templates (Iterable[str] | None): Optional complete template set.

    Returns:
        dict[str, ParameterFeatureSchema]: Stable feature schema per template.
    """
    numeric_positions_by_template: dict[str, set[int]] = defaultdict(set)
    non_numeric_positions_by_template: dict[str, set[int]] = defaultdict(set)
    saw_elapsed_by_template: dict[str, bool] = defaultdict(bool)

    for sequence in normal_sequences:
        for template, parameters, dt_prev_ms in sequence.events:
            if include_elapsed_time and dt_prev_ms is not None:
                saw_elapsed_by_template[template] = True
            for position, value in enumerate(parameters):
                if try_parse_numeric(value) is None:
                    non_numeric_positions_by_template[template].add(position)
                    continue
                numeric_positions_by_template[template].add(position)

    template_names = (
        set(all_templates)
        if all_templates is not None
        else (
            set(numeric_positions_by_template)
            | set(non_numeric_positions_by_template)
            | set(saw_elapsed_by_template)
        )
    )

    schemas: dict[str, ParameterFeatureSchema] = {}
    for template in sorted(template_names):
        numeric_positions = sorted(
            numeric_positions_by_template.get(template, set())
            - non_numeric_positions_by_template.get(template, set()),
        )
        saw_elapsed = bool(saw_elapsed_by_template.get(template))
        feature_names: list[str] = []
        if include_elapsed_time and saw_elapsed:
            feature_names.append(DT_FEATURE_NAME)
        feature_names.extend(f"param_{position}" for position in numeric_positions)
        schemas[template] = ParameterFeatureSchema(
            feature_names=feature_names,
            numeric_parameter_positions=numeric_positions,
            include_elapsed_time=include_elapsed_time and saw_elapsed,
            dropped_parameter_positions=sorted(
                non_numeric_positions_by_template.get(template, set()),
            ),
        )
    return schemas


def parameter_model_input_size(*, feature_count: int) -> int:
    """Return the parameter-model input width.

    For paper fidelity, each history step is just the normalized parameter
    vector itself. We do not append extra missingness channels.

    Args:
        feature_count (int): Number of features in the parameter vector.

    Returns:
        int: Input width for the parameter model.
    """
    return feature_count


def parameter_covered_event_count(
    *,
    sequences: Iterable[TemplateSequence],
    parameter_models: dict[str, ParameterModelState],
) -> int:
    """Count events whose template has a model and at least one observed feature.

    Args:
        sequences (Iterable[TemplateSequence]): Sequences to inspect.
        parameter_models (dict[str, ParameterModelState]): Fitted parameter models.

    Returns:
        int: Covered event count.
    """
    covered_event_count = 0
    for sequence in sequences:
        for template, parameters, dt_prev_ms in sequence.events:
            state = parameter_models.get(template)
            if state is None:
                continue
            raw_vector = raw_parameter_vector_for_event(
                parameters=parameters,
                dt_prev_ms=dt_prev_ms,
                schema=state.schema,
            )
            if any(raw_vector.mask):
                covered_event_count += 1
    return covered_event_count


def raw_parameter_vector_for_event(
    *,
    parameters: list[str],
    dt_prev_ms: int | None,
    schema: ParameterFeatureSchema,
) -> RawParameterVector:
    """Convert one event payload into the schema's fixed-width raw vector.

    Args:
        parameters (list[str]): Raw parameter strings from the event.
        dt_prev_ms (int | None): Elapsed milliseconds since the previous event.
        schema (ParameterFeatureSchema): Stable schema for the event template.

    Returns:
        RawParameterVector: Raw values and observation mask for one event.
    """
    values: list[float] = []
    mask: list[bool] = []

    if schema.include_elapsed_time:
        if dt_prev_ms is None:
            values.append(0.0)
            mask.append(False)
        else:
            values.append(float(dt_prev_ms))
            mask.append(True)

    for position in schema.numeric_parameter_positions:
        numeric_value = (
            try_parse_numeric(parameters[position])
            if position < len(parameters)
            else None
        )
        if numeric_value is None:
            values.append(0.0)
            mask.append(False)
            continue
        values.append(numeric_value)
        mask.append(True)
    return RawParameterVector(values=values, mask=mask)


def normalisation_for_raw_series(
    *,
    raw_series: list[list[RawParameterVector]],
) -> NormalisationStats:
    """Compute per-feature normalisation statistics from observed values only.

    Args:
        raw_series (list[list[RawParameterVector]]): Training vectors.

    Returns:
        NormalisationStats: Mean and standard deviation per feature.
    """
    if not raw_series:
        return NormalisationStats(means=[], stddevs=[])

    feature_count = len(raw_series[0][0].values)
    observed_values: list[list[float]] = [[] for _ in range(feature_count)]
    for series in raw_series:
        for vector in series:
            for index, is_observed in enumerate(vector.mask):
                if is_observed:
                    observed_values[index].append(vector.values[index])

    means: list[float] = []
    stddevs: list[float] = []
    for values in observed_values:
        if not values:
            means.append(0.0)
            stddevs.append(1.0)
            continue
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        stddevs.append(math.sqrt(max(variance, EPSILON)))
    return NormalisationStats(means=means, stddevs=stddevs)


def normalize_vector(
    *,
    vector: RawParameterVector,
    normalisation: NormalisationStats,
) -> list[float]:
    """Normalize one raw parameter vector.

    Missing positions become `0.0` after normalisation as well. That value is
    only a placeholder for tensor shape; downstream mask-aware code makes sure
    those positions do not contribute to learning or anomaly scoring.

    Args:
        vector (RawParameterVector): Raw vector to normalize.
        normalisation (NormalisationStats): Fitted normalisation statistics.

    Returns:
        list[float]: Normalized vector with masked positions zero-filled.
    """
    normalized: list[float] = []
    for index, value in enumerate(vector.values):
        if not vector.mask[index]:
            normalized.append(0.0)
            continue
        normalized.append(
            (value - normalisation.means[index]) / normalisation.stddevs[index],
        )
    return normalized


def denormalize_vector(
    *,
    normalized_values: list[float],
    normalisation: NormalisationStats,
) -> list[float]:
    """Map one normalized model output vector back to the raw feature space.

    Args:
        normalized_values (list[float]): Normalized model output.
        normalisation (NormalisationStats): Fitted normalisation statistics.

    Returns:
        list[float]: Denormalized feature values.
    """
    return [
        (normalized_value * normalisation.stddevs[index]) + normalisation.means[index]
        for index, normalized_value in enumerate(normalized_values)
    ]


def masked_optional_values(
    values: list[float],
    mask: list[bool],
) -> list[float | None]:
    """Replace unobserved positions with `None` for JSON-friendly output.

    Args:
        values (list[float]): Feature values.
        mask (list[bool]): Positions present in the original event.

    Returns:
        list[float | None]: Optionalized feature values.
    """
    return [
        value if is_observed else None
        for value, is_observed in zip(values, mask, strict=True)
    ]


def most_anomalous_feature(
    *,
    feature_names: list[str],
    observed: list[float],
    predicted: list[float],
    mask: list[bool],
) -> str | None:
    """Return the observed feature with the largest absolute prediction error.

    Args:
        feature_names (list[str]): Feature names in vector order.
        observed (list[float]): Observed raw feature values.
        predicted (list[float]): Predicted raw feature values.
        mask (list[bool]): Positions present in the original event.

    Returns:
        str | None: Feature name with the largest error, if any.
    """
    best_feature: str | None = None
    best_error = -1.0
    for feature_name, observed_value, predicted_value, is_observed in zip(
        feature_names,
        observed,
        predicted,
        mask,
        strict=True,
    ):
        if not is_observed:
            continue
        error = abs(observed_value - predicted_value)
        if error > best_error:
            best_feature = feature_name
            best_error = error
    return best_feature


def try_parse_numeric(raw_value: str) -> float | None:
    """Parse a parameter string into a finite float if possible.

    Args:
        raw_value (str): Raw parameter string.

    Returns:
        float | None: Parsed finite float, or `None` if not modelable.
    """
    try:
        parsed = float(raw_value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed
