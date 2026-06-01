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
    """Minimal template-bearing series contract used by report selection.

    Attributes:
        template (str): Template identifier used to select highlighted series.
    """

    template: str


TSeries = TypeVar("TSeries", bound=_HasTemplate)

_MIN_CALIBRATION_SAMPLE_COUNT = 30


class DeepLogParameterCiThresholds(msgspec.Struct, frozen=True):
    """Gaussian upper thresholds used for the Figure 9 style CI report.

    Attributes:
        confidence_98 (float): Threshold at the 98th percentile.
        confidence_99 (float): Threshold at the 99th percentile.
        confidence_999 (float): Threshold at the 99.9th percentile.
    """

    confidence_98: float
    confidence_99: float
    confidence_999: float


class DeepLogParameterCiThresholdSummary(msgspec.Struct, frozen=True):
    """Aggregate threshold outcome for one template series.

    Attributes:
        confidence (float): Quantile used for the threshold.
        threshold (float): Numeric threshold value.
        point_count (int): Total scored points in the series.
        anomalous_point_count (int): Anomalous points in the series.
        detected_point_count (int): Points above the threshold.
        detected_anomalous_point_count (int): Anomalous points above the
            threshold.
        detected_normal_point_count (int): Normal points above the threshold.
        detection_rate (float): Fraction of all points detected.
        anomalous_detection_rate (float): Fraction of anomalous points detected.
    """

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
    """Aggregate per-template parameter residual series for the report.

    Attributes:
        template (str): Template identifier for the series.
        feature_names (list[str]): Feature names observed for the template.
        train_pair_count (int): Number of training pairs used to fit the model.
        validation_pair_count (int): Number of validation pairs used to
            calibrate the thresholds.
        validation_sample_warning (str | None): Warning emitted when validation
            samples are too sparse.
        point_count (int): Total scored points in the series.
        anomalous_point_count (int): Anomalous points in the series.
        residual_mse_mean (float): Mean residual error across the series.
        residual_mse_min (float): Minimum residual error across the series.
        residual_mse_max (float): Maximum residual error across the series.
        thresholds (DeepLogParameterCiThresholds): Gaussian thresholds.
        threshold_summaries (list[DeepLogParameterCiThresholdSummary]):
            Threshold outcome summaries for the configured confidence levels.
    """

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
    """One scored parameter-model point in the debug trace.

    Attributes:
        window_id (int): Sequence/window identifier.
        split_label (str): Train/test split label for the point.
        label (int): Ground-truth anomaly label for the point.
        event_index (int): Index of the scored event within the sequence.
        template (str): Template identifier used for the score.
        residual_mse (float): Residual mean-squared error for the point.
        detected_at_98 (bool): Whether the point exceeded the 98th percentile.
        detected_at_99 (bool): Whether the point exceeded the 99th percentile.
        detected_at_999 (bool): Whether the point exceeded the 99.9th percentile.
        most_anomalous_feature (str | None): Feature contributing most to the
            residual.
        feature_squared_errors (list[float | None]): Per-feature squared errors.
    """

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
    """Per-template parameter residual trace for debugging.

    Attributes:
        template (str): Template identifier for the series.
        feature_names (list[str]): Feature names observed for the template.
        train_pair_count (int): Number of training pairs used to fit the model.
        validation_pair_count (int): Number of validation pairs used to
            calibrate the thresholds.
        validation_sample_warning (str | None): Warning emitted when validation
            samples are too sparse.
        point_count (int): Total scored points in the series.
        anomalous_point_count (int): Anomalous points in the series.
        thresholds (DeepLogParameterCiThresholds): Gaussian thresholds.
        points (list[DeepLogParameterCiTracePoint]): Per-point debug trace.
    """

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
    """Compact DeepLog parameter-value report for the OpenStack approximation.

    Attributes:
        task (str): Stable task identifier used in manifest output.
        paper_approximation (bool): Whether the summary follows the paper
            approximation.
        paper_exact_reproduction (bool): Whether the summary exactly reproduces
            the paper procedure.
        result_note (str): Human-readable note describing the report flavour.
        series_scope (str): Scope string describing the selected series subset.
        train_sequence_count (int): Number of training sequences scored.
        test_sequence_count (int): Number of test sequences scored.
        series_count (int): Number of available template series.
        highlighted_series_count (int): Number of series included in the
            published summary.
        highlighted_templates (list[str]): Templates retained in the compact
            summary.
        series (list[DeepLogParameterCiSeries]): Highlighted series payloads.
        total_point_count (int): Total scored points across all series.
        total_anomalous_point_count (int): Total anomalous points across all
            series.
    """

    task: str
    paper_approximation: bool
    paper_exact_reproduction: bool
    result_note: str
    series_scope: str
    train_sequence_count: int
    test_sequence_count: int
    series_count: int
    highlighted_series_count: int
    highlighted_templates: list[str]
    series: list[DeepLogParameterCiSeries]
    total_point_count: int
    total_anomalous_point_count: int


class DeepLogParameterCiTraceReport(msgspec.Struct, frozen=True):
    """Verbose parameter-value trace for explicit debugging.

    Attributes:
        task (str): Stable task identifier used in manifest output.
        paper_approximation (bool): Whether the summary follows the paper
            approximation.
        paper_exact_reproduction (bool): Whether the summary exactly reproduces
            the paper procedure.
        result_note (str): Human-readable note describing the report flavour.
        series_scope (str): Scope string describing the selected series subset.
        train_sequence_count (int): Number of training sequences scored.
        test_sequence_count (int): Number of test sequences scored.
        series_count (int): Number of available template series.
        highlighted_series_count (int): Number of series included in the
            published summary.
        highlighted_templates (list[str]): Templates retained in the compact
            summary.
        series (list[DeepLogParameterCiTraceSeries]): Highlighted series
            payloads.
        anomalous_points (list[DeepLogParameterCiTracePoint]): All anomalous
            debug trace points.
        total_point_count (int): Total scored points across all series.
        total_anomalous_point_count (int): Total anomalous points across all
            series.
    """

    task: str
    paper_approximation: bool
    paper_exact_reproduction: bool
    result_note: str
    series_scope: str
    train_sequence_count: int
    test_sequence_count: int
    series_count: int
    highlighted_series_count: int
    highlighted_templates: list[str]
    series: list[DeepLogParameterCiTraceSeries]
    anomalous_points: list[DeepLogParameterCiTracePoint]
    total_point_count: int
    total_anomalous_point_count: int


@dataclass(slots=True)
class _ParameterCiSeriesState:
    """Mutable per-template state for the parameter CI approximation report.

    Attributes:
        feature_names (list[str]): Feature names observed for the template.
        train_pair_count (int): Number of training pairs used to fit the model.
        validation_pair_count (int): Number of validation pairs used to
            calibrate the thresholds.
        gaussian_mean (float): Mean residual value learned from training.
        gaussian_stddev (float): Standard deviation of the learned residuals.
        points (list[DeepLogParameterCiTracePoint]): Recorded trace points for
            the template.
    """

    feature_names: list[str]
    train_pair_count: int
    validation_pair_count: int
    gaussian_mean: float
    gaussian_stddev: float
    points: list[DeepLogParameterCiTracePoint] = field(default_factory=list)


@dataclass(slots=True)
class ParameterCiState:
    """Run-local accumulator for the parameter-value CI approximation.

    Attributes:
        series_by_template (dict[str, _ParameterCiSeriesState]): Per-template
            mutable state used to assemble summary and trace reports.
    """

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
        """Record one scored sequence into the parameter CI trace.

        Args:
            sequence (TemplateSequence): Sequence being scored.
            parameter_findings (dict[int, DeepLogParameterFinding]): Per-event
                residual findings for the sequence.
            parameter_models (dict[str, ParameterModelState]): Learned per-template
                parameter models.
        """
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
        """Return the aggregate publication-facing CI report.

        Args:
            train_sequence_count (int): Number of training sequences scored.
            test_sequence_count (int): Number of test sequences scored.
            highlighted_templates (tuple[str, ...] | None): Optional template
                subset to retain in the compact report.
            include_empty (bool): Whether to return an empty report instead of
                `None` when no series were recorded.

        Returns:
            DeepLogParameterCiReport | None: Compact report, or `None` when no
                series were recorded and `include_empty` is false.
        """
        if not self.series_by_template:
            if not include_empty:
                return None
            return DeepLogParameterCiReport(
                task="parameter_ci_approximation",
                paper_approximation=True,
                paper_exact_reproduction=False,
                result_note=_parameter_ci_result_note(),
                series_scope="highlighted_subset",
                train_sequence_count=train_sequence_count,
                test_sequence_count=test_sequence_count,
                series_count=0,
                highlighted_series_count=0,
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
            series_scope="highlighted_subset",
            train_sequence_count=train_sequence_count,
            test_sequence_count=test_sequence_count,
            series_count=len(series),
            highlighted_series_count=len(highlighted_series),
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
        """Return the verbose per-point trace for debugging.

        Args:
            train_sequence_count (int): Number of training sequences scored.
            test_sequence_count (int): Number of test sequences scored.
            highlighted_templates (tuple[str, ...] | None): Optional template
                subset to retain in the trace report.
            include_empty (bool): Whether to return an empty report instead of
                `None` when no series were recorded.

        Returns:
            DeepLogParameterCiTraceReport | None: Verbose trace report, or
                `None` when no series were recorded and `include_empty` is
                false.
        """
        if not self.series_by_template:
            if not include_empty:
                return None
            return DeepLogParameterCiTraceReport(
                task="parameter_ci_approximation",
                paper_approximation=True,
                paper_exact_reproduction=False,
                result_note=_parameter_ci_result_note(),
                series_scope="highlighted_subset",
                train_sequence_count=train_sequence_count,
                test_sequence_count=test_sequence_count,
                series_count=0,
                highlighted_series_count=0,
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
            series_scope="highlighted_subset",
            train_sequence_count=train_sequence_count,
            test_sequence_count=test_sequence_count,
            series_count=len(series),
            highlighted_series_count=len(highlighted_series),
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
    """Return Gaussian upper thresholds for the configured CI levels.

    Args:
        mean (float): Gaussian mean fitted from the normal residuals.
        stddev (float): Gaussian standard deviation fitted from the normal
            residuals.

    Returns:
        DeepLogParameterCiThresholds: Upper thresholds for the configured
            confidence levels.
    """
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

    Args:
        points (list[DeepLogParameterCiTracePoint]): Scored trace points.
        thresholds (DeepLogParameterCiThresholds): Gaussian upper thresholds.

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
    """Build a threshold summary for one confidence level.

    Args:
        points (list[DeepLogParameterCiTracePoint]): Scored trace points.
        confidence (float): Confidence level represented by the summary.
        threshold (float): Threshold value being evaluated.

    Returns:
        DeepLogParameterCiThresholdSummary: Aggregate detection counts for the
            threshold.
    """
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
    """Return a note when Gaussian calibration uses a small validation tail.

    Args:
        validation_pair_count (int): Number of validation pairs used for the
            Gaussian calibration.

    Returns:
        str | None: Warning text when the calibration tail is small, otherwise
            `None`.
    """
    if validation_pair_count >= _MIN_CALIBRATION_SAMPLE_COUNT:
        return None
    return (
        "validation_pair_count="
        f"{validation_pair_count}; Gaussian CI calibration is based on a "
        "small normal validation tail and should be treated as fragile."
    )


def _parameter_ci_result_note() -> str:
    """Return the stable note attached to the Figure 9 approximation.

    Returns:
        str: Stable explanatory note written into the compact report.
    """
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
    """Return the most specific available label for one scored event.

    Args:
        sequence (TemplateSequence): Sequence containing the scored event.
        event_index (int): Position of the event within the sequence.

    Returns:
        int: Best available anomaly label for the scored event.
    """
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
    """Return per-feature squared errors for a scored parameter point.

    Args:
        observed (list[float | None]): Observed feature values.
        predicted (list[float | None]): Predicted feature values.

    Returns:
        list[float | None]: Per-feature squared errors, or `None` when a
            feature value is missing.
    """
    return [
        None if obs is None or pred is None else (obs - pred) ** 2
        for obs, pred in zip(observed, predicted, strict=True)
    ]


def _select_highlighted_series(
    series: list[TSeries],
    *,
    requested_templates: tuple[str, ...] | None,
) -> list[TSeries]:
    """Return the compact report subset, honouring any explicit ordering.

    Args:
        series (list[TSeries]): Candidate series sorted by publication order.
        requested_templates (tuple[str, ...] | None): Optional template subset
            to retain in the compact report.

    Returns:
        list[TSeries]: Highlighted series subset for the compact report.
    """
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
