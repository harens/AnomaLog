"""DeepLog parameter-value CI reporting helpers.

This module keeps the Figure 9 approximation report separate from detector
runtime state. The published report is aggregate-first, while the detailed
per-event trace stays available for explicit debugging artefacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import TYPE_CHECKING, Protocol, TypeVar

import msgspec

if TYPE_CHECKING:
    from anomalog.sequences import TemplateSequence
    from experiments.models.deeplog.shared import (
        DeepLogParameterFinding,
        ParameterModelState,
    )


class _HasTemplate(Protocol):
    """Minimal template-bearing series contract used by report selection."""

    template: str


TSeries = TypeVar("TSeries", bound=_HasTemplate)

_MIN_CALIBRATION_SAMPLE_COUNT = 30


class DeepLogParameterCiThresholds(msgspec.Struct, frozen=True):
    """Gaussian upper thresholds used for the Figure 9 style CI report."""

    confidence_98: float
    confidence_99: float
    confidence_999: float


class DeepLogParameterCiThresholdSummary(msgspec.Struct, frozen=True):
    """Aggregate threshold outcome for one template series."""

    confidence: float
    threshold: float
    point_count: int
    anomalous_point_count: int
    detected_point_count: int
    detected_anomalous_point_count: int
    detected_normal_point_count: int
    detection_rate: float
    anomalous_detection_rate: float


class DeepLogParameterCiSeries(msgspec.Struct, frozen=True):
    """Aggregate per-template parameter residual series for the report."""

    template: str
    feature_names: list[str]
    train_pair_count: int
    validation_pair_count: int
    validation_sample_warning: str | None
    point_count: int
    anomalous_point_count: int
    residual_mse_mean: float
    residual_mse_min: float
    residual_mse_max: float
    thresholds: DeepLogParameterCiThresholds
    threshold_summaries: list[DeepLogParameterCiThresholdSummary]


class DeepLogParameterCiTracePoint(msgspec.Struct, frozen=True):
    """One scored parameter-model point in the debug trace."""

    window_id: int
    split_label: str
    label: int
    event_index: int
    template: str
    residual_mse: float
    detected_at_98: bool
    detected_at_99: bool
    detected_at_999: bool
    most_anomalous_feature: str | None
    feature_squared_errors: list[float | None]


class DeepLogParameterCiTraceSeries(msgspec.Struct, frozen=True):
    """Per-template parameter residual trace for debugging."""

    template: str
    feature_names: list[str]
    train_pair_count: int
    validation_pair_count: int
    validation_sample_warning: str | None
    point_count: int
    anomalous_point_count: int
    thresholds: DeepLogParameterCiThresholds
    points: list[DeepLogParameterCiTracePoint]


class DeepLogParameterCiReport(msgspec.Struct, frozen=True):
    """Compact DeepLog parameter-value report for the OpenStack approximation."""

    task: str
    paper_approximation: bool
    paper_exact_reproduction: bool
    result_note: str
    train_sequence_count: int
    test_sequence_count: int
    series_count: int
    highlighted_templates: list[str]
    series: list[DeepLogParameterCiSeries]
    total_point_count: int
    total_anomalous_point_count: int


class DeepLogParameterCiTraceReport(msgspec.Struct, frozen=True):
    """Verbose parameter-value trace for explicit debugging."""

    task: str
    paper_approximation: bool
    paper_exact_reproduction: bool
    result_note: str
    train_sequence_count: int
    test_sequence_count: int
    series_count: int
    highlighted_templates: list[str]
    series: list[DeepLogParameterCiTraceSeries]
    anomalous_points: list[DeepLogParameterCiTracePoint]
    total_point_count: int
    total_anomalous_point_count: int


@dataclass(slots=True)
class _ParameterCiSeriesState:
    """Mutable per-template state for the parameter CI approximation report."""

    feature_names: list[str]
    train_pair_count: int
    validation_pair_count: int
    gaussian_mean: float
    gaussian_stddev: float
    points: list[DeepLogParameterCiTracePoint] = field(default_factory=list)


@dataclass(slots=True)
class ParameterCiState:
    """Run-local accumulator for the parameter-value CI approximation."""

    series_by_template: dict[str, _ParameterCiSeriesState] = field(
        default_factory=dict,
    )

    def record_sequence(
        self,
        *,
        sequence: TemplateSequence,
        parameter_findings: dict[int, DeepLogParameterFinding],
        parameter_models: dict[str, ParameterModelState],
    ) -> None:
        """Record one scored sequence into the parameter CI trace."""
        for event_index, finding in sorted(parameter_findings.items()):
            state = parameter_models.get(finding.template)
            if state is None:
                continue
            series_state = self.series_by_template.get(finding.template)
            if series_state is None:
                series_state = _ParameterCiSeriesState(
                    feature_names=list(finding.feature_names),
                    train_pair_count=state.train_pair_count,
                    validation_pair_count=state.validation_pair_count,
                    gaussian_mean=state.gaussian.mean,
                    gaussian_stddev=state.gaussian.stddev,
                )
                self.series_by_template[finding.template] = series_state
            thresholds = parameter_ci_thresholds(
                mean=series_state.gaussian_mean,
                stddev=series_state.gaussian_stddev,
            )
            series_state.points.append(
                DeepLogParameterCiTracePoint(
                    window_id=sequence.window_id,
                    split_label=sequence.split_label.value,
                    label=_event_label_for_sequence_point(
                        sequence=sequence,
                        event_index=event_index,
                    ),
                    event_index=event_index,
                    template=finding.template,
                    residual_mse=finding.residual_mse,
                    detected_at_98=finding.residual_mse > thresholds.confidence_98,
                    detected_at_99=finding.residual_mse > thresholds.confidence_99,
                    detected_at_999=finding.residual_mse > thresholds.confidence_999,
                    most_anomalous_feature=finding.most_anomalous_feature,
                    feature_squared_errors=_feature_squared_errors(
                        observed=finding.observed_vector,
                        predicted=finding.predicted_vector,
                    ),
                ),
            )

    def snapshot_summary(
        self,
        *,
        train_sequence_count: int,
        test_sequence_count: int,
        highlighted_templates: tuple[str, ...] | None = None,
        include_empty: bool = False,
    ) -> DeepLogParameterCiReport | None:
        """Return the aggregate publication-facing CI report."""
        if not self.series_by_template:
            if not include_empty:
                return None
            return DeepLogParameterCiReport(
                task="parameter_ci_approximation",
                paper_approximation=True,
                paper_exact_reproduction=False,
                result_note=_parameter_ci_result_note(),
                train_sequence_count=train_sequence_count,
                test_sequence_count=test_sequence_count,
                series_count=0,
                highlighted_templates=[],
                series=[],
                total_point_count=0,
                total_anomalous_point_count=0,
            )
        series: list[DeepLogParameterCiSeries] = []
        for template, state in sorted(
            self.series_by_template.items(),
            key=lambda item: (-len(item[1].points), item[0]),
        ):
            thresholds = parameter_ci_thresholds(
                mean=state.gaussian_mean,
                stddev=state.gaussian_stddev,
            )
            point_count = len(state.points)
            anomalous_point_count = sum(1 for point in state.points if point.label != 0)
            residual_mses = [point.residual_mse for point in state.points]
            threshold_summary_items = _threshold_summaries(
                points=state.points,
                thresholds=thresholds,
            )
            series.append(
                DeepLogParameterCiSeries(
                    template=template,
                    feature_names=list(state.feature_names),
                    train_pair_count=state.train_pair_count,
                    validation_pair_count=state.validation_pair_count,
                    validation_sample_warning=_validation_sample_warning(
                        state.validation_pair_count,
                    ),
                    point_count=point_count,
                    anomalous_point_count=anomalous_point_count,
                    residual_mse_mean=_mean(residual_mses),
                    residual_mse_min=min(residual_mses),
                    residual_mse_max=max(residual_mses),
                    thresholds=thresholds,
                    threshold_summaries=threshold_summary_items,
                ),
            )
        highlighted_series = _select_highlighted_series(
            series,
            requested_templates=highlighted_templates,
        )
        highlighted_template_names = [item.template for item in highlighted_series]
        return DeepLogParameterCiReport(
            task="parameter_ci_approximation",
            paper_approximation=True,
            paper_exact_reproduction=False,
            result_note=_parameter_ci_result_note(),
            train_sequence_count=train_sequence_count,
            test_sequence_count=test_sequence_count,
            series_count=len(series),
            highlighted_templates=highlighted_template_names,
            series=highlighted_series,
            total_point_count=sum(series_item.point_count for series_item in series),
            total_anomalous_point_count=sum(
                series_item.anomalous_point_count for series_item in series
            ),
        )

    def snapshot_trace(
        self,
        *,
        train_sequence_count: int,
        test_sequence_count: int,
        highlighted_templates: tuple[str, ...] | None = None,
        include_empty: bool = False,
    ) -> DeepLogParameterCiTraceReport | None:
        """Return the verbose per-point trace for debugging."""
        if not self.series_by_template:
            if not include_empty:
                return None
            return DeepLogParameterCiTraceReport(
                task="parameter_ci_approximation",
                paper_approximation=True,
                paper_exact_reproduction=False,
                result_note=_parameter_ci_result_note(),
                train_sequence_count=train_sequence_count,
                test_sequence_count=test_sequence_count,
                series_count=0,
                highlighted_templates=[],
                series=[],
                anomalous_points=[],
                total_point_count=0,
                total_anomalous_point_count=0,
            )
        series: list[DeepLogParameterCiTraceSeries] = []
        anomalous_points: list[DeepLogParameterCiTracePoint] = []
        for template, state in sorted(
            self.series_by_template.items(),
            key=lambda item: (-len(item[1].points), item[0]),
        ):
            thresholds = parameter_ci_thresholds(
                mean=state.gaussian_mean,
                stddev=state.gaussian_stddev,
            )
            points = list(state.points)
            anomalous_points.extend(point for point in points if point.label != 0)
            series.append(
                DeepLogParameterCiTraceSeries(
                    template=template,
                    feature_names=list(state.feature_names),
                    train_pair_count=state.train_pair_count,
                    validation_pair_count=state.validation_pair_count,
                    validation_sample_warning=_validation_sample_warning(
                        state.validation_pair_count,
                    ),
                    point_count=len(points),
                    anomalous_point_count=sum(
                        1 for point in points if point.label != 0
                    ),
                    thresholds=thresholds,
                    points=points,
                ),
            )
        highlighted_series = _select_highlighted_series(
            series,
            requested_templates=highlighted_templates,
        )
        highlighted_template_names = [item.template for item in highlighted_series]
        return DeepLogParameterCiTraceReport(
            task="parameter_ci_approximation",
            paper_approximation=True,
            paper_exact_reproduction=False,
            result_note=_parameter_ci_result_note(),
            train_sequence_count=train_sequence_count,
            test_sequence_count=test_sequence_count,
            series_count=len(series),
            highlighted_templates=highlighted_template_names,
            series=highlighted_series,
            anomalous_points=anomalous_points,
            total_point_count=sum(len(series_item.points) for series_item in series),
            total_anomalous_point_count=sum(
                series_item.anomalous_point_count for series_item in series
            ),
        )


def parameter_ci_thresholds(
    *,
    mean: float,
    stddev: float,
) -> DeepLogParameterCiThresholds:
    """Return Gaussian upper thresholds for the configured CI levels."""
    if stddev <= 0.0:
        threshold = round(mean, 8)
        return DeepLogParameterCiThresholds(
            confidence_98=threshold,
            confidence_99=threshold,
            confidence_999=threshold,
        )
    distribution = NormalDist(mu=mean, sigma=stddev)
    return DeepLogParameterCiThresholds(
        confidence_98=round(distribution.inv_cdf(0.98), 8),
        confidence_99=round(distribution.inv_cdf(0.99), 8),
        confidence_999=round(distribution.inv_cdf(0.999), 8),
    )


def _threshold_summaries(
    *,
    points: list[DeepLogParameterCiTracePoint],
    thresholds: DeepLogParameterCiThresholds,
) -> list[DeepLogParameterCiThresholdSummary]:
    """Build threshold summaries for the configured Figure 9 cut-offs.

    Returns:
        list[DeepLogParameterCiThresholdSummary]: Threshold-wise detection
            aggregates for the configured confidence levels.
    """
    return [
        _threshold_summary(
            points=points,
            confidence=0.98,
            threshold=thresholds.confidence_98,
        ),
        _threshold_summary(
            points=points,
            confidence=0.99,
            threshold=thresholds.confidence_99,
        ),
        _threshold_summary(
            points=points,
            confidence=0.999,
            threshold=thresholds.confidence_999,
        ),
    ]


def _threshold_summary(
    *,
    points: list[DeepLogParameterCiTracePoint],
    confidence: float,
    threshold: float,
) -> DeepLogParameterCiThresholdSummary:
    detected_points = [point for point in points if point.residual_mse > threshold]
    anomalous_points = [point for point in points if point.label != 0]
    detected_anomalous_point_count = sum(
        1 for point in detected_points if point.label != 0
    )
    detected_normal_point_count = len(detected_points) - detected_anomalous_point_count
    anomalous_point_count = len(anomalous_points)
    point_count = len(points)
    return DeepLogParameterCiThresholdSummary(
        confidence=confidence,
        threshold=threshold,
        point_count=point_count,
        anomalous_point_count=anomalous_point_count,
        detected_point_count=len(detected_points),
        detected_anomalous_point_count=detected_anomalous_point_count,
        detected_normal_point_count=detected_normal_point_count,
        detection_rate=(
            0.0 if point_count == 0 else len(detected_points) / point_count
        ),
        anomalous_detection_rate=(
            0.0
            if anomalous_point_count == 0
            else detected_anomalous_point_count / anomalous_point_count
        ),
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 8)


def _validation_sample_warning(validation_pair_count: int) -> str | None:
    """Return a note when Gaussian calibration uses a small validation tail."""
    if validation_pair_count >= _MIN_CALIBRATION_SAMPLE_COUNT:
        return None
    return (
        "validation_pair_count="
        f"{validation_pair_count}; Gaussian CI calibration is based on a "
        "small normal validation tail and should be treated as fragile."
    )


def _parameter_ci_result_note() -> str:
    """Return the stable note attached to the Figure 9 approximation."""
    return (
        "Approximation of DeepLog Figure 9 on the available OpenStack corpus; "
        "highlighted anomaly points are injected overlays, and Gaussian "
        "thresholds are calibrated from normal validation residuals only."
    )


def _event_label_for_sequence_point(
    *,
    sequence: TemplateSequence,
    event_index: int,
) -> int:
    """Return the most specific available label for one scored event."""
    event_labels = sequence.event_labels
    if event_labels is not None and event_index < len(event_labels):
        event_label = event_labels[event_index]
        if event_label is not None:
            return event_label
    return sequence.label


def _feature_squared_errors(
    *,
    observed: list[float | None],
    predicted: list[float | None],
) -> list[float | None]:
    """Return per-feature squared errors for a scored parameter point."""
    return [
        None if obs is None or pred is None else (obs - pred) ** 2
        for obs, pred in zip(observed, predicted, strict=True)
    ]


def _select_highlighted_series(
    series: list[TSeries],
    *,
    requested_templates: tuple[str, ...] | None,
) -> list[TSeries]:
    """Return the compact report subset, honouring any explicit ordering."""
    if requested_templates:
        series_by_template = {item.template: item for item in series}
        selected: list[TSeries] = []
        seen_templates: set[str] = set()
        for template in requested_templates:
            if template in seen_templates:
                continue
            item = series_by_template.get(template)
            if item is None:
                continue
            selected.append(item)
            seen_templates.add(template)
        if selected:
            return selected
    return series[:4]
