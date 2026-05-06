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
from collections.abc import Sized
from dataclasses import dataclass
from typing import TYPE_CHECKING

from experiments.models.deeplog.shared import (
    DT_FEATURE_NAME,
    EPSILON,
    NormalisationStats,
    ParameterFeatureSchema,
    ParameterModelState,
    training_event_index_mask,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rich.progress import Progress, TaskID

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
    progress: Progress | None = None,
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
        progress (Progress | None): Optional progress reporter for schema
            preparation.

    Returns:
        dict[str, ParameterFeatureSchema]: Stable feature schema per template.
    """
    numeric_positions_by_template: dict[str, set[int]] = defaultdict(set)
    non_numeric_positions_by_template: dict[str, set[int]] = defaultdict(set)
    saw_elapsed_by_template: dict[str, bool] = defaultdict(bool)
    observations = _ParameterSchemaObservations(
        numeric_positions_by_template=numeric_positions_by_template,
        non_numeric_positions_by_template=non_numeric_positions_by_template,
        saw_elapsed_by_template=saw_elapsed_by_template,
    )
    prepare_task = _prepare_parameter_schema_progress(
        normal_sequences=normal_sequences,
        progress=progress,
    )
    _collect_parameter_schema_observations(
        normal_sequences=normal_sequences,
        include_elapsed_time=include_elapsed_time,
        observations=observations,
        progress=progress,
        prepare_task=prepare_task,
    )
    if progress is not None and prepare_task is not None:
        progress.remove_task(prepare_task)

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


@dataclass(frozen=True, slots=True)
class _ParameterSchemaObservations:
    """Mutable counters gathered while scanning DeepLog parameter schemas.

    Attributes:
        numeric_positions_by_template (dict[str, set[int]]): Numeric parameter
            positions observed for each template.
        non_numeric_positions_by_template (dict[str, set[int]]): Parameter
            positions rejected because at least one value was non-numeric.
        saw_elapsed_by_template (dict[str, bool]): Whether each template saw
            at least one non-null elapsed-time value.
    """

    numeric_positions_by_template: dict[str, set[int]]
    non_numeric_positions_by_template: dict[str, set[int]]
    saw_elapsed_by_template: dict[str, bool]


def _prepare_parameter_schema_progress(
    *,
    normal_sequences: Iterable[TemplateSequence],
    progress: Progress | None,
) -> TaskID | None:
    """Create a progress task for parameter-schema preparation.

    Args:
        normal_sequences (Iterable[TemplateSequence]): Normal training sequences.
        progress (Progress | None): Optional progress reporter.

    Returns:
        TaskID | None: Task identifier when progress tracking is enabled.
    """
    if progress is None:
        return None
    total = len(normal_sequences) if isinstance(normal_sequences, Sized) else None
    return progress.add_task(
        "Preparing DeepLog parameter schemas",
        total=total,
    )


def _collect_parameter_schema_observations(
    *,
    normal_sequences: Iterable[TemplateSequence],
    include_elapsed_time: bool,
    observations: _ParameterSchemaObservations,
    progress: Progress | None,
    prepare_task: TaskID | None,
) -> None:
    """Scan normal sequences and accumulate schema observations.

    Args:
        normal_sequences (Iterable[TemplateSequence]): Normal training sequences.
        include_elapsed_time (bool): Whether elapsed time may become a feature.
        observations (_ParameterSchemaObservations): Mutable schema counters.
        progress (Progress | None): Optional progress reporter.
        prepare_task (TaskID | None): Progress task used for preparation.
    """
    for sequence in normal_sequences:
        for template, parameters, dt_prev_ms in sequence.events:
            if include_elapsed_time and dt_prev_ms is not None:
                observations.saw_elapsed_by_template[template] = True
            for position, value in enumerate(parameters):
                if try_parse_numeric(value) is None:
                    observations.non_numeric_positions_by_template[template].add(
                        position,
                    )
                    continue
                observations.numeric_positions_by_template[template].add(position)
        if progress is not None and prepare_task is not None:
            progress.advance(prepare_task)


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
        eligible_target_indexes = set(training_event_index_mask(sequence))
        for event_index, (template, parameters, dt_prev_ms) in enumerate(
            sequence.events,
        ):
            if event_index not in eligible_target_indexes:
                continue
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
