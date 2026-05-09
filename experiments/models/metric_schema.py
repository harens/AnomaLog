"""Stable metric taxonomy used by experiment result reporting."""

from __future__ import annotations

from enum import Enum


class MetricScope(str, Enum):  # noqa: DOC601 DOC603
    """Stable metric blocks reported by experiment runs.

    Attributes:
        EVENT_LEVEL_DETECTION: Event-granularity binary detection metrics.
        SEQUENCE_LEVEL_DETECTION: Sequence-granularity binary detection metrics.
        WINDOW_LEVEL_DETECTION: Sliding-window binary detection metrics.
        STREAM_LEVEL_DETECTION: Continuous-stream binary detection metrics.
        NEXT_EVENT_PREDICTION: Next-event modelling and hit-rate metrics.
        CLUSTER_LEVEL_TRIAGE: Cluster or review-group triage metrics.
        MANUAL_WORKLOAD_REDUCTION: Manual review workload reduction metrics.
        SEMI_AUTOMATIC_WORKLOAD_REDUCTION: Semi-automatic workload reduction metrics.
    """

    EVENT_LEVEL_DETECTION = "event_level_detection"
    SEQUENCE_LEVEL_DETECTION = "sequence_level_detection"
    WINDOW_LEVEL_DETECTION = "window_level_detection"
    STREAM_LEVEL_DETECTION = "stream_level_detection"
    NEXT_EVENT_PREDICTION = "next_event_prediction"
    CLUSTER_LEVEL_TRIAGE = "cluster_level_triage"
    MANUAL_WORKLOAD_REDUCTION = "manual_workload_reduction"
    SEMI_AUTOMATIC_WORKLOAD_REDUCTION = "semi_automatic_workload_reduction"


class MetricStatus(str, Enum):  # noqa: DOC601 DOC603
    """Validity status for one metric block.

    Attributes:
        VALID: The block passed validation and can be treated as headline data.
        INVALID: The block failed validation and should not be promoted.
        NOT_APPLICABLE: The scope does not apply to this run.
        DIAGNOSTIC_ONLY: The block is informative but not a headline metric.
    """

    VALID = "valid"
    INVALID = "invalid"
    NOT_APPLICABLE = "not_applicable"
    DIAGNOSTIC_ONLY = "diagnostic_only"


class EvaluationUnit(str, Enum):  # noqa: DOC601 DOC603
    """Stable evaluation and prediction units used in metric reports.

    Attributes:
        EVENT: Individual log events.
        SEQUENCE: Whole sequence or case abstractions.
        WINDOW: Fixed-size sliding windows.
        STREAM: A generic stream segment.
        NEXT_EVENT: Next-event prediction samples.
        CLUSTER: Human triage or clustering units.
        CHRONOLOGICAL_EVENT_STREAM: Chronologically ordered event stream slices.
        CONTINUOUS_EVENT_STREAM: Continuous event-stream slices.
    """

    EVENT = "event"
    SEQUENCE = "sequence"
    WINDOW = "window"
    STREAM = "stream"
    NEXT_EVENT = "next_event"
    CLUSTER = "cluster"
    CHRONOLOGICAL_EVENT_STREAM = "chronological_event_stream"
    CONTINUOUS_EVENT_STREAM = "continuous_event_stream"


class AggregationPolicy(str, Enum):  # noqa: DOC601 DOC603
    """Stable aggregation policies for derived metric blocks.

    Attributes:
        AUTO_DECISION: Metrics over automatic decisions only.
        TOP_K: Metrics over top-k next-event predictions.
        TOP_1: Metrics over top-1 next-event predictions.
        ANY_EVENT: A block is positive if any event in the unit is positive.
        MAX_EVENT_SCORE: Aggregate using the maximum event score.
        MEAN_EVENT_SCORE: Aggregate using the mean event score.
        MANUAL_SAMPLE: Metrics computed over manually reviewed samples.
        SEMI_AUTOMATIC: Metrics computed over semi-automatic decisions.
        NONE: No aggregation beyond the base unit.
    """

    AUTO_DECISION = "auto_decision"
    TOP_K = "top_k"
    TOP_1 = "top_1"
    ANY_EVENT = "any_event"
    MAX_EVENT_SCORE = "max_event_score"
    MEAN_EVENT_SCORE = "mean_event_score"
    MANUAL_SAMPLE = "manual_sample"
    SEMI_AUTOMATIC = "semi_automatic"
    NONE = "none"
