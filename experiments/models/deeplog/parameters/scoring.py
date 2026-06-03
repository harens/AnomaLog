"""Inference-time scoring for DeepLog parameter models.

This module answers the runtime question:
"given a fitted per-template model, how do we score a new sequence?"

The key point is that parameter scoring is not run over the raw sequence as one
global stream. Just like training, each template is scored on its own ordered
subsequence, and only for events that survived the key-model stage.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import torch

from experiments.models.deeplog.parameters.dataset import masked_mse
from experiments.models.deeplog.parameters.schema import (
    RawParameterVector,
    denormalize_vector,
    masked_optional_values,
    most_anomalous_feature,
    normalize_vector,
    raw_parameter_vector_for_event,
)
from experiments.models.deeplog.shared import (
    EPSILON,
    DeepLogParameterFinding,
    ParameterModelState,
)

if TYPE_CHECKING:
    from anomalog.sequences import TemplateSequence


ParameterEventsByTemplate = dict[str, list[tuple[list[str], int | None]]]


def score_parameter_sequence(
    *,
    sequence: TemplateSequence,
    parameter_models: dict[str, ParameterModelState],
    history_size: int,
    eligible_event_indexes: set[int] | None = None,
    prefix_events_by_template: ParameterEventsByTemplate | None = None,
) -> dict[int, DeepLogParameterFinding]:
    """Score one sequence with the template-specific parameter models.

    The flow is intentionally parallel to training:

    1. group sequence events by template
    2. build the ordered raw and normalized vectors for that template
    3. form inference histories of length `history_size`
    4. run the template-specific LSTM on those histories
    5. compare predicted vs observed vectors with masked residual MSE

    Args:
        sequence (TemplateSequence): Sequence to score.
        parameter_models (dict[str, ParameterModelState]): Fitted models by template.
        history_size (int): Number of prior template events per inference history.
        eligible_event_indexes (set[int] | None): Optional event indexes
            allowed for parameter scoring.
        prefix_events_by_template (ParameterEventsByTemplate | None): Optional
            carried-over template histories from the immediately preceding
            chronological batch boundary.

    Returns:
        dict[int, DeepLogParameterFinding]: Event index to parameter finding.
    """
    findings: dict[int, DeepLogParameterFinding] = {}
    history_by_template, raw_history_by_template = _prime_parameter_histories(
        prefix_events_by_template=prefix_events_by_template or {},
        parameter_models=parameter_models,
        history_size=history_size,
    )

    for event_index, (template, parameters, dt_prev_ms) in enumerate(sequence.events):
        state = parameter_models.get(template)
        if state is None:
            continue
        raw_vector = raw_parameter_vector_for_event(
            parameters=parameters,
            dt_prev_ms=dt_prev_ms,
            schema=state.schema,
        )
        normalized_vector = normalize_vector(
            vector=raw_vector,
            normalisation=state.normalisation,
        )
        history = history_by_template[template]
        if len(history) < history_size:
            history.append(normalized_vector)
            raw_history_by_template[template].append(raw_vector)
            _trim_parameter_history(
                history_by_template=history_by_template,
                raw_history_by_template=raw_history_by_template,
                template=template,
                history_size=history_size,
            )
            continue
        if (
            eligible_event_indexes is not None
            and event_index not in eligible_event_indexes
        ):
            history.append(normalized_vector)
            raw_history_by_template[template].append(raw_vector)
            _trim_parameter_history(
                history_by_template=history_by_template,
                raw_history_by_template=raw_history_by_template,
                template=template,
                history_size=history_size,
            )
            continue
        if not any(raw_vector.mask):
            history.append(normalized_vector)
            raw_history_by_template[template].append(raw_vector)
            _trim_parameter_history(
                history_by_template=history_by_template,
                raw_history_by_template=raw_history_by_template,
                template=template,
                history_size=history_size,
            )
            continue
        model_device = next(state.model.parameters()).device
        inputs = torch.tensor(
            [history[-history_size:]],
            dtype=torch.float32,
            device=model_device,
        )
        with torch.inference_mode():
            predicted = state.model(inputs).cpu().squeeze(0).tolist()
        findings[event_index] = build_parameter_finding(
            event_index=event_index,
            state=state,
            raw_target=raw_vector,
            normalized_target=normalized_vector,
            predicted=predicted,
        )
        history.append(normalized_vector)
        raw_history_by_template[template].append(raw_vector)
        _trim_parameter_history(
            history_by_template=history_by_template,
            raw_history_by_template=raw_history_by_template,
            template=template,
            history_size=history_size,
        )
    return findings


def _prime_parameter_histories(
    *,
    prefix_events_by_template: dict[str, list[tuple[list[str], int | None]]],
    parameter_models: dict[str, ParameterModelState],
    history_size: int,
) -> tuple[dict[str, list[list[float]]], dict[str, list[RawParameterVector]]]:
    history_by_template: dict[str, list[list[float]]] = defaultdict(list)
    raw_history_by_template: dict[str, list[RawParameterVector]] = defaultdict(list)

    for template, prefix_events in prefix_events_by_template.items():
        state = parameter_models.get(template)
        if state is None:
            continue
        for parameters, dt_prev_ms in prefix_events:
            raw_vector = raw_parameter_vector_for_event(
                parameters=parameters,
                dt_prev_ms=dt_prev_ms,
                schema=state.schema,
            )
            history_by_template[template].append(
                normalize_vector(vector=raw_vector, normalisation=state.normalisation),
            )
            raw_history_by_template[template].append(raw_vector)
            _trim_parameter_history(
                history_by_template=history_by_template,
                raw_history_by_template=raw_history_by_template,
                template=template,
                history_size=history_size,
            )

    return history_by_template, raw_history_by_template


def _trim_parameter_history(
    *,
    history_by_template: dict[str, list[list[float]]],
    raw_history_by_template: dict[str, list[RawParameterVector]],
    template: str,
    history_size: int,
) -> None:
    history = history_by_template[template]
    if len(history) > history_size:
        history_by_template[template] = history[-history_size:]
        raw_history_by_template[template] = raw_history_by_template[template][
            -history_size:
        ]


def build_parameter_finding(
    *,
    event_index: int,
    state: ParameterModelState,
    raw_target: RawParameterVector,
    normalized_target: list[float],
    predicted: list[float],
) -> DeepLogParameterFinding:
    """Build one event-level parameter-model finding.

    Args:
        event_index (int): Sequence-local event index.
        state (ParameterModelState): Fitted model state for the event template.
        raw_target (RawParameterVector): Observed raw target vector.
        normalized_target (list[float]): Normalized target vector.
        predicted (list[float]): Normalized model prediction.

    Returns:
        DeepLogParameterFinding: Serialised parameter-model decision payload.
    """
    residual_mse = masked_mse(
        observed=normalized_target,
        predicted=predicted,
        mask=raw_target.mask,
    )
    denormalized_prediction = denormalize_vector(
        normalized_values=predicted,
        normalisation=state.normalisation,
    )
    return DeepLogParameterFinding(
        event_index=event_index,
        template=state.template,
        feature_names=state.schema.feature_names,
        observed_vector=masked_optional_values(
            raw_target.values,
            raw_target.mask,
        ),
        predicted_vector=masked_optional_values(
            denormalized_prediction,
            raw_target.mask,
        ),
        residual_mse=residual_mse,
        gaussian_mean=state.gaussian.mean,
        gaussian_stddev=state.gaussian.stddev,
        gaussian_lower_bound=state.gaussian.lower_bound,
        gaussian_upper_bound=state.gaussian.upper_bound,
        most_anomalous_feature=most_anomalous_feature(
            feature_names=state.schema.feature_names,
            observed=raw_target.values,
            predicted=denormalized_prediction,
            mask=raw_target.mask,
        ),
        is_anomalous=residual_mse > state.gaussian.upper_bound,
    )


def normalized_parameter_score(finding: DeepLogParameterFinding) -> float:
    """Normalize a parameter residual relative to its anomaly threshold.

    Args:
        finding (DeepLogParameterFinding): Parameter finding to normalize.

    Returns:
        float: Residual divided by the fitted anomaly threshold.
    """
    threshold = max(finding.gaussian_upper_bound, EPSILON)
    return finding.residual_mse / threshold


def parameter_anomaly_score(finding: DeepLogParameterFinding) -> float:
    """Return anomaly-only parameter score as margin beyond Gaussian bounds.

    Args:
        finding (DeepLogParameterFinding): Parameter finding to score.

    Returns:
        float: Margin beyond the fitted Gaussian upper bound, or `0.0` when
            the finding is not anomalous.
    """
    if not finding.is_anomalous:
        return 0.0
    upper_bound = max(finding.gaussian_upper_bound, EPSILON)
    return (finding.residual_mse - finding.gaussian_upper_bound) / upper_bound
