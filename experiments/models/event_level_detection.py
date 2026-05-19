"""Shared event-level detection diagnostics for experiment detectors."""

from __future__ import annotations

from dataclasses import dataclass

import msgspec

from anomalog.parsers.structured.contracts import is_anomalous_label


class EventLevelDetectionDiagnostics(msgspec.Struct, frozen=True):  # noqa: DOC601 DOC603
    """Binary event-level detection summary for a detector run.

    Attributes:
        task (str): Metric scope name used in report output.
        events_seen (int): Number of labelled events evaluated.
        events_eligible (int): Number of eligible events evaluated.
        tp (int): True positives.
        tn (int): True negatives.
        fp (int): False positives.
        fn (int): False negatives.
        normal_event_count (int): Number of normal labelled events seen.
        anomalous_event_count (int): Number of anomalous labelled events seen.
        precision (float): Precision rounded for report output.
        recall (float): Recall rounded for report output.
        f1 (float): F1 score rounded for report output.
    """

    task: str
    events_seen: int
    events_eligible: int
    tp: int
    tn: int
    fp: int
    fn: int
    normal_event_count: int
    anomalous_event_count: int
    precision: float
    recall: float
    f1: float


@dataclass(slots=True)
class EventLevelDetectionState:
    """Mutable counters for event-level detection metrics.

    Attributes:
        events_seen (int): Number of labelled events evaluated.
        events_eligible (int): Number of eligible events evaluated.
        tp (int): True positives.
        tn (int): True negatives.
        fp (int): False positives.
        fn (int): False negatives.
        normal_event_count (int): Number of normal labelled events seen.
        anomalous_event_count (int): Number of anomalous labelled events seen.
    """

    events_seen: int = 0
    events_eligible: int = 0
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0
    normal_event_count: int = 0
    anomalous_event_count: int = 0

    def record(self, *, actual_label: int, predicted_label: int) -> None:
        """Accumulate one scored labelled event.

        Args:
            actual_label (int): Ground-truth event label.
            predicted_label (int): Detector-predicted event label.
        """
        self.events_seen += 1
        self.events_eligible += 1
        if is_anomalous_label(actual_label):
            self.anomalous_event_count += 1
            if predicted_label == 1:
                self.tp += 1
            else:
                self.fn += 1
            return
        self.normal_event_count += 1
        if predicted_label == 0:
            self.tn += 1
        else:
            self.fp += 1

    def snapshot(self, *, task: str) -> EventLevelDetectionDiagnostics | None:
        """Return a serialisable metrics snapshot if any events were seen.

        Args:
            task (str): Metric scope name used in report output.

        Returns:
            EventLevelDetectionDiagnostics | None: Latest event-level metrics,
            or `None` when no labelled events were observed.
        """
        if self.events_seen <= 0:
            return None
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return EventLevelDetectionDiagnostics(
            task=task,
            events_seen=self.events_seen,
            events_eligible=self.events_eligible,
            tp=self.tp,
            tn=self.tn,
            fp=self.fp,
            fn=self.fn,
            normal_event_count=self.normal_event_count,
            anomalous_event_count=self.anomalous_event_count,
            precision=round(precision, 8),
            recall=round(recall, 8),
            f1=round(f1, 8),
        )

    def reset(self) -> None:
        """Clear the accumulated counters."""
        self.events_seen = 0
        self.events_eligible = 0
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
        self.normal_event_count = 0
        self.anomalous_event_count = 0
