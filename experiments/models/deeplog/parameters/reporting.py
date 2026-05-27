"""DeepLog parameter-value CI reporting helpers.

This module owns the Figure 9 style approximation report for the parameter
branch. Keeping the data model here avoids making the detector a dumping
ground for report-specific state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist
from typing import TYPE_CHECKING

import msgspec

if TYPE_CHECKING:
    from anomalog.sequences import TemplateSequence
    from experiments.models.deeplog.shared import (
        DeepLogParameterFinding,
        ParameterModelState,
    )


class DeepLogParameterCiThresholds(msgspec.Struct, frozen=True):
    """Gaussian upper thresholds used for the Figure 9 style CI report."""

    confidence_98: float
    confidence_99: float
    confidence_999: float


class DeepLogParameterCiPoint(msgspec.Struct, frozen=True):
    """One scored parameter-model point in the CI approximation report."""

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


class DeepLogParameterCiSeries(msgspec.Struct, frozen=True):
    """Per-template parameter residual series in the CI approximation report."""

    template: str
    feature_names: list[str]
    train_pair_count: int
    validation_pair_count: int
    point_count: int
    anomalous_point_count: int
    thresholds: DeepLogParameterCiThresholds
    points: list[DeepLogParameterCiPoint]


class DeepLogParameterCiReport(msgspec.Struct, frozen=True):
    """Compact DeepLog parameter-value report for the OpenStack approximation."""

    task: str
    paper_approximation: bool
    paper_exact_reproduction: bool
    train_sequence_count: int
    test_sequence_count: int
    series_count: int
    highlighted_templates: list[str]
    series: list[DeepLogParameterCiSeries]
    anomalous_points: list[DeepLogParameterCiPoint]
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
    points: list[DeepLogParameterCiPoint] = field(default_factory=list)


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
        """Record one scored sequence into the parameter CI report."""
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
                DeepLogParameterCiPoint(
                    window_id=sequence.window_id,
                    split_label=sequence.split_label.value,
                    label=sequence.label,
                    event_index=event_index,
                    template=finding.template,
                    residual_mse=finding.residual_mse,
                    detected_at_98=finding.residual_mse > thresholds.confidence_98,
                    detected_at_99=finding.residual_mse > thresholds.confidence_99,
                    detected_at_999=finding.residual_mse > thresholds.confidence_999,
                    most_anomalous_feature=finding.most_anomalous_feature,
                ),
            )

    def snapshot(
        self,
        *,
        train_sequence_count: int,
        test_sequence_count: int,
    ) -> DeepLogParameterCiReport | None:
        """Return a serialisable view of the recorded CI series."""
        if not self.series_by_template:
            return None
        series: list[DeepLogParameterCiSeries] = []
        for template, state in sorted(
            self.series_by_template.items(),
            key=lambda item: (-len(item[1].points), item[0]),
        ):
            thresholds = parameter_ci_thresholds(
                mean=state.gaussian_mean,
                stddev=state.gaussian_stddev,
            )
            series.append(
                DeepLogParameterCiSeries(
                    template=template,
                    feature_names=list(state.feature_names),
                    train_pair_count=state.train_pair_count,
                    validation_pair_count=state.validation_pair_count,
                    point_count=len(state.points),
                    anomalous_point_count=sum(
                        1 for point in state.points if point.label != 0
                    ),
                    thresholds=thresholds,
                    points=list(state.points),
                ),
            )
        highlighted_series = series[:4]
        highlighted_templates = [item.template for item in highlighted_series]
        anomalous_points = [
            point
            for series_item in series
            for point in series_item.points
            if point.label != 0
        ]
        return DeepLogParameterCiReport(
            task="parameter_ci_approximation",
            paper_approximation=True,
            paper_exact_reproduction=False,
            train_sequence_count=train_sequence_count,
            test_sequence_count=test_sequence_count,
            series_count=len(series),
            highlighted_templates=highlighted_templates,
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
