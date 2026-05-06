"""Tests for the DeepCase experiment integration."""

from __future__ import annotations

import msgspec
import numpy as np
import pytest
import torch
from deepcase import DeepCASE
from deepcase.context_builder.context_builder import ContextBuilder
from deepcase.interpreter.interpreter import Interpreter
from rich.progress import Progress

from anomalog.sequences import SplitLabel, TemplateSequence
from experiments import ConfigError
from experiments.models import resolve_model_config_type
from experiments.models.base import SequenceSummary, decode_experiment_model_config
from experiments.models.deepcase import DeepCaseModelConfig
from experiments.models.deepcase import shared as deepcase_shared
from experiments.models.deepcase.detector import DeepCaseDetector, DeepCaseRunMetrics
from experiments.models.deepcase.shared import (
    DeepCaseEventIdMap,
    DeepCaseSampleBatch,
    DeepCaseSequenceDecision,
    DeepCaseWorkloadAlertSampling,
    DeepCaseWorkloadMode,
    ScoreReason,
    aggregate_sequence_score,
    build_sample_batch,
    build_training_batch,
    build_workload_reduction_metrics,
    decision_label_for_score,
    finding_reason_for_score,
)
from experiments.models.next_event_metrics import (
    NextEventPredictionDiagnostics,
    VocabularyPolicy,
)

MALICIOUS_SCORE = 3.0
UNKNOWN_EVENT_SCORE = -2.0
EXPECTED_EVENT_METRICS = {
    "event_count": 5,
    "event_auto_decision_count": 4,
    "event_abstained_decision_count": 1,
    "event_auto_coverage": pytest.approx(0.8),
    "event_abstain_rate": pytest.approx(0.2),
    "event_tp": 1,
    "event_fp": 1,
    "event_tn": 1,
    "event_fn": 1,
    "event_precision": pytest.approx(0.5),
    "event_recall": pytest.approx(0.5),
    "event_f1": pytest.approx(0.5),
    "event_accuracy": pytest.approx(0.5),
    "event_predicted_normal_count": 2,
    "event_predicted_anomalous_count": 2,
    "event_true_normal_count": 3,
    "event_true_anomalous_count": 2,
}
EXPECTED_CONFIDENT_EVENT_COUNT = 4
EXPECTED_ABSTAINED_EVENT_COUNT = 1
EXPECTED_CONFIDENT_ANOMALY_EVENT_COUNT = 2
EXPECTED_ABSTAINED_ANOMALOUS_LABEL_COUNT = 0
EXPECTED_ABSTAINED_NORMAL_LABEL_COUNT = 1
EXPECTED_SEQUENCE_CONFIDENT_ANOMALY_COUNT = 2
EXPECTED_SEQUENCE_CONFIDENT_NORMAL_COUNT = 2
EXPECTED_SEQUENCE_ABSTAINED_COUNT = 1
EXPECTED_WORKLOAD_CLUSTER_COUNT = 5
EXPECTED_WORKLOAD_ALERTS_PER_CLUSTER = 10
EXPECTED_WORKLOAD_ALERT_COUNT = 50
EXPECTED_WORKLOAD_COVERAGE = pytest.approx(0.8)
EXPECTED_WORKLOAD_REDUCTION = pytest.approx(0.375)
EXPECTED_WORKLOAD_OVERALL = pytest.approx(0.3)
EXPECTED_WORKLOAD_SEMI_AUTOMATIC_REDUCTION = pytest.approx(1.0)
EXPECTED_WORKLOAD_SEMI_AUTOMATIC_OVERALL = pytest.approx(0.8)
ConfigValue = str | int | float | bool | None


def _deep_case_config(**values: ConfigValue) -> DeepCaseModelConfig:
    raw_config = {"name": "deepcase", **values}
    try:
        return decode_experiment_model_config(
            raw_config,
            config_type=DeepCaseModelConfig,
        )
    except ConfigError:
        raise
    except (msgspec.ValidationError, msgspec.DecodeError, TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc


def _sequence(
    *,
    templates: list[str],
    label: int = 0,
    entity_ids: list[str] | None = None,
    split_label: SplitLabel = SplitLabel.TRAIN,
    event_metadata: dict[str, list[int | None] | None] | None = None,
) -> TemplateSequence:
    """Build a compact `TemplateSequence` fixture for DeepCase tests.

    Args:
        templates (list[str]): Ordered event templates for the sequence.
        label (int): Parent sequence label.
        entity_ids (list[str] | None): Optional entity ids for the sequence.
        split_label (SplitLabel): Dataset split assigned to the sequence.
        event_metadata (dict[str, list[int | None] | None] | None): Optional
            per-event metadata for tests that need timestamps or event labels.

    Returns:
        TemplateSequence: Sequence fixture with aligned event rows.
    """
    resolved_dts = (
        event_metadata.get("dts_by_event") if event_metadata is not None else None
    )
    resolved_event_labels = (
        event_metadata.get("event_labels") if event_metadata is not None else None
    )
    if resolved_dts is None:
        resolved_dts = [None for _ in templates]
    if resolved_event_labels is None:
        resolved_event_labels = None
    return TemplateSequence(
        events=[
            (template, [], dt_prev_ms)
            for template, dt_prev_ms in zip(templates, resolved_dts, strict=True)
        ],
        label=label,
        entity_ids=["entity-1"] if entity_ids is None else entity_ids,
        window_id=0,
        split_label=split_label,
        event_labels=(
            tuple(resolved_event_labels) if resolved_event_labels is not None else None
        ),
    )


def _finding(
    *,
    event_index: int,
    raw_score: float,
) -> deepcase_shared.DeepCaseEventFinding:
    reason = finding_reason_for_score(raw_score)
    return deepcase_shared.DeepCaseEventFinding(
        event_index=event_index,
        template=f"event-{event_index}",
        event_id=event_index,
        raw_score=raw_score,
        reason=reason,
        predicted_label=decision_label_for_score(raw_score),
        is_abstained=reason.is_abstained,
    )


def _deepcase_detector_for_event_decision_tests() -> DeepCaseDetector:
    """Return a minimal fitted-shape DeepCASE detector for event tests.

    Returns:
        DeepCaseDetector: Detector with train vocabulary and model handles
        populated enough for synthetic prediction tests.
    """
    train_sequence = _sequence(templates=["A", "B", "C"])
    detector = DeepCaseDetector(
        config=_deep_case_config(name="deepcase", epochs=1, iterations=100),
    )
    detector.event_id_map = DeepCaseEventIdMap.from_sequences((train_sequence,))
    detector.model = DeepCASE(features=len(detector.event_id_map.event_id_to_template))
    return detector


def _assert_event_decision_diagnostics(
    diagnostics: deepcase_shared.DeepCasePredictionDiagnostics,
) -> None:
    """Assert the expected DeepCASE event decision diagnostics for the fixture run.

    Args:
        diagnostics (deepcase_shared.DeepCasePredictionDiagnostics): Run
            diagnostics to verify.
    """
    assert diagnostics.event_count == EXPECTED_EVENT_METRICS["event_count"]
    assert (
        diagnostics.confident_event_count + diagnostics.abstained_event_count
        == diagnostics.event_count
    )
    assert diagnostics.confident_event_count == EXPECTED_CONFIDENT_EVENT_COUNT
    assert diagnostics.abstained_event_count == EXPECTED_ABSTAINED_EVENT_COUNT
    assert (
        diagnostics.confident_anomaly_event_count
        == EXPECTED_CONFIDENT_ANOMALY_EVENT_COUNT
    )
    assert (
        diagnostics.abstained_anomalous_label_count
        == EXPECTED_ABSTAINED_ANOMALOUS_LABEL_COUNT
    )
    assert (
        diagnostics.abstained_normal_label_count
        == EXPECTED_ABSTAINED_NORMAL_LABEL_COUNT
    )
    assert (
        diagnostics.sequence_confident_anomaly_count
        == EXPECTED_SEQUENCE_CONFIDENT_ANOMALY_COUNT
    )
    assert (
        diagnostics.sequence_confident_normal_count
        == EXPECTED_SEQUENCE_CONFIDENT_NORMAL_COUNT
    )
    assert diagnostics.sequence_abstained_count == EXPECTED_SEQUENCE_ABSTAINED_COUNT
    assert diagnostics.reason_counts == {
        "known_benign_cluster": 2,
        "known_malicious_cluster": 2,
        "not_confident_enough": 1,
    }
    event_decision_metrics = diagnostics.event_decision_metrics
    assert isinstance(
        event_decision_metrics,
        deepcase_shared.DeepCaseEventDecisionMetrics,
    )
    for field_name, expected_value in EXPECTED_EVENT_METRICS.items():
        assert getattr(event_decision_metrics, field_name) == expected_value
    assert (
        event_decision_metrics.event_auto_decision_count
        == event_decision_metrics.event_tp
        + event_decision_metrics.event_fp
        + event_decision_metrics.event_tn
        + event_decision_metrics.event_fn
    )
    assert (
        event_decision_metrics.event_predicted_normal_count
        + event_decision_metrics.event_predicted_anomalous_count
        == event_decision_metrics.event_auto_decision_count
    )
    assert (
        event_decision_metrics.event_count
        == event_decision_metrics.event_auto_decision_count
        + event_decision_metrics.event_abstained_decision_count
    )


def _build_event_decision_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> deepcase_shared.DeepCasePredictionDiagnostics:
    """Run a synthetic DeepCASE bulk prediction pass and return diagnostics.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the batch prediction hook
            with deterministic fixture scores.

    Returns:
        deepcase_shared.DeepCasePredictionDiagnostics: Diagnostics captured
        from the synthetic run.
    """
    detector = _deepcase_detector_for_event_decision_tests()
    sequences = [
        _sequence(
            templates=["A"],
            label=0,
            split_label=SplitLabel.TEST,
            event_metadata={"event_labels": [0]},
        ),
        _sequence(
            templates=["A"],
            label=1,
            split_label=SplitLabel.TEST,
            event_metadata={"event_labels": [1]},
        ),
        _sequence(
            templates=["A"],
            label=1,
            split_label=SplitLabel.TEST,
            event_metadata={"event_labels": [1]},
        ),
        _sequence(
            templates=["A"],
            label=0,
            split_label=SplitLabel.TEST,
            event_metadata={"event_labels": [0]},
        ),
        _sequence(
            templates=["A"],
            label=0,
            split_label=SplitLabel.TEST,
            event_metadata={"event_labels": [0]},
        ),
    ]
    raw_scores = [
        0.0,
        0.0,
        MALICIOUS_SCORE,
        -1.0,
        MALICIOUS_SCORE,
    ]
    monkeypatch.setattr(detector, "_predict_batch", lambda _batch: raw_scores)

    outcomes = list(detector.predict_all(sequences))
    assert len(outcomes) == len(sequences)

    metrics = detector.run_metrics(
        run_metrics={
            "test_sequence_count": len(sequences),
            "counted_predictions": len(sequences) - 1,
            "abstained_prediction_count": 1,
            "ignored_sequence_count": 0,
        },
    )
    diagnostics = metrics.prediction_diagnostics
    assert diagnostics is not None
    return diagnostics


def _build_single_event_decision_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raw_score: float,
    true_label: int,
) -> deepcase_shared.DeepCasePredictionDiagnostics:
    """Run a one-event DeepCASE prediction and return diagnostics.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the batch prediction hook.
        raw_score (float): Synthetic DeepCASE score for the single event.
        true_label (int): Ground-truth label for the single event.

    Returns:
        deepcase_shared.DeepCasePredictionDiagnostics: Diagnostics captured
        from the synthetic run.
    """
    detector = _deepcase_detector_for_event_decision_tests()
    sequence = _sequence(
        templates=["A"],
        label=true_label,
        split_label=SplitLabel.TEST,
        event_metadata={"event_labels": [true_label]},
    )
    monkeypatch.setattr(detector, "_predict_batch", lambda _batch: [raw_score])

    outcomes = list(detector.predict_all((sequence,)))
    assert len(outcomes) == 1

    metrics = detector.run_metrics(
        run_metrics={
            "test_sequence_count": 1,
            "counted_predictions": 1,
            "abstained_prediction_count": 0,
            "ignored_sequence_count": 0,
        },
    )
    diagnostics = metrics.prediction_diagnostics
    assert diagnostics is not None
    return diagnostics


@pytest.mark.allow_no_new_coverage
def test_deepcase_model_config_validates_hyperparameters() -> None:
    """DeepCase configs should reject invalid detector settings."""
    # This protects msgspec-backed config constraints; those failures do not
    # map to uncovered Python branches in the experiment code.
    with pytest.raises(ConfigError, match="context_length"):
        _deep_case_config(name="bad", context_length=0)
    with pytest.raises(ConfigError, match="cluster_score_strategy"):
        _deep_case_config(name="bad", cluster_score_strategy="median")
    with pytest.raises(ConfigError, match="device"):
        _deep_case_config(name="bad", device="tpu")


@pytest.mark.allow_no_new_coverage
def test_deepcase_model_config_accepts_mps_device() -> None:
    """DeepCase configs should allow MPS for Apple Silicon runs."""
    # This protects the msgspec-backed device literal contract; exercising an
    # uncovered runtime branch is not the right expression of this config case.
    config = _deep_case_config(name="deepcase", device="mps")

    assert config.device == "mps"


def test_deepcase_model_config_defaults_next_event_policy() -> None:
    """DeepCase should default next-event diagnostics to the full dataset."""
    config = _deep_case_config(name="deepcase")

    assert config.vocabulary_policy is VocabularyPolicy.FULL_DATASET


def test_deepcase_model_config_accepts_train_only_next_event_policy() -> None:
    """DeepCase should accept train-only next-event diagnostics when requested."""
    config = _deep_case_config(name="deepcase", vocabulary_policy="train_only")

    assert config.vocabulary_policy is VocabularyPolicy.TRAIN_ONLY


@pytest.mark.allow_no_new_coverage
def test_deepcase_fit_passes_label_smoothing_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepCase training should pass configured smoothing to the library.

    The original library does not allow configuring delta, even though in the paper
    it is a tunable hyperparameter. This test ensures we the forked version is
    correctly wired to forward the value from the config.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the library fit stages so the
            test can capture forwarded hyperparameters without real training.
    """
    # This protects the parameter-forwarding contract without relying on
    # incidental new line coverage in the upstream library wrappers.
    captured_delta: list[float] = []

    def _fit_context_builder(
        self: ContextBuilder,
        *,
        delta: float,
        **kwargs: object,
    ) -> ContextBuilder:
        del kwargs
        captured_delta.append(float(delta))
        return self

    def _fit_interpreter(self: Interpreter, **kwargs: object) -> Interpreter:
        del kwargs
        self.clusters = np.array([0, -1])
        return self

    monkeypatch.setattr(ContextBuilder, "fit", _fit_context_builder)
    monkeypatch.setattr(Interpreter, "fit", _fit_interpreter)
    detector = _deep_case_config(
        name="deepcase",
        label_smoothing_delta=0.25,
        epochs=1,
    ).build_detector()
    sequence = _sequence(templates=["A", "B"])

    with Progress(disable=True) as progress:
        detector.fit((sequence,), progress=progress)

    assert captured_delta == [0.25]


def test_deepcase_fit_runs_context_builder_one_epoch_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepCase should surface per-epoch progress while training context builder.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the upstream DeepCase fit
            methods so the test can observe how many one-epoch calls are made.
    """
    epoch_calls: list[int] = []

    def _fit_context_builder(
        self: ContextBuilder,
        *,
        epochs: int,
        **kwargs: object,
    ) -> ContextBuilder:
        del kwargs
        epoch_calls.append(epochs)
        return self

    def _fit_interpreter(self: Interpreter, **kwargs: object) -> Interpreter:
        del kwargs
        self.clusters = np.array([0, -1])
        return self

    monkeypatch.setattr(ContextBuilder, "fit", _fit_context_builder)
    monkeypatch.setattr(Interpreter, "fit", _fit_interpreter)
    detector = _deep_case_config(
        name="deepcase",
        epochs=3,
        iterations=100,
    ).build_detector()
    sequence = _sequence(templates=["A", "B"])

    with Progress(disable=True) as progress:
        detector.fit((sequence,), progress=progress)

    assert epoch_calls == [1, 1, 1]


def test_deepcase_detector_rejects_repeated_fit() -> None:
    """DeepCase detector instances should only be trained once."""
    detector = _deep_case_config(
        name="deepcase",
        epochs=1,
        iterations=100,
    ).build_detector()
    sequence = _sequence(templates=["A", "B"])

    with Progress(disable=True) as progress:
        detector.fit((sequence,), progress=progress)

    with (
        Progress(disable=True) as progress,
        pytest.raises(
            RuntimeError,
            match="can only be fit once",
        ),
    ):
        detector.fit((sequence,), progress=progress)


def test_model_registry_resolves_deepcase() -> None:
    """DeepCase should be registered as a built-in experiment detector."""
    assert resolve_model_config_type("deepcase") is DeepCaseModelConfig


def test_build_sample_batch_left_pads_missing_context() -> None:
    """DeepCase contexts should be left-padded with the NO_EVENT id."""
    sequence = _sequence(templates=["A", "B", "C"])
    event_id_map = DeepCaseEventIdMap.from_sequences((sequence,))

    batch = build_sample_batch(
        (sequence,),
        event_id_map=event_id_map,
        context_length=3,
        timeout_seconds=86_400,
    )

    no_event = event_id_map.no_event_id
    assert batch.contexts.tolist() == [
        [no_event, no_event, no_event],
        [no_event, no_event, event_id_map.template_to_event_id["A"]],
        [
            no_event,
            event_id_map.template_to_event_id["A"],
            event_id_map.template_to_event_id["B"],
        ],
    ]
    assert batch.context_original_event_ids == [
        [None, None, None],
        [None, None, event_id_map.template_to_event_id["A"]],
        [
            None,
            event_id_map.template_to_event_id["A"],
            event_id_map.template_to_event_id["B"],
        ],
    ]
    assert batch.events.tolist() == [
        event_id_map.template_to_event_id["A"],
        event_id_map.template_to_event_id["B"],
        event_id_map.template_to_event_id["C"],
    ]


def test_build_sample_batch_uses_target_event_labels_when_available() -> None:
    """DeepCase should prefer target-event labels over smeared parent labels."""
    sequence = _sequence(
        templates=["A", "B", "C", "D", "E"],
        event_metadata={"event_labels": [0, 0, 1, 0, 0]},
        label=1,
    )
    event_id_map = DeepCaseEventIdMap.from_sequences((sequence,))

    batch = build_sample_batch(
        (sequence,),
        event_id_map=event_id_map,
        context_length=2,
        timeout_seconds=86_400,
    )

    assert batch.scores.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert batch.parent_sequence_fallback_count == 0


def test_build_sample_batch_respects_explicit_evaluation_event_mask() -> None:
    """DeepCase should only score events flagged for evaluation."""
    sequence = TemplateSequence(
        events=[
            ("A", [], None),
            ("B", [], None),
            ("C", [], None),
        ],
        label=1,
        entity_ids=["entity-1"],
        window_id=0,
        split_label=SplitLabel.TEST,
        event_labels=(0, 1, 1),
        evaluation_event_mask=(False, True, True),
    )
    event_id_map = DeepCaseEventIdMap.from_sequences((sequence,))

    batch = build_sample_batch(
        (sequence,),
        event_id_map=event_id_map,
        context_length=2,
        timeout_seconds=86_400,
    )

    assert batch.event_indexes == [1, 2]
    assert batch.templates == ["B", "C"]
    assert batch.scores.tolist() == [1.0, 1.0]
    assert batch.parent_sequence_fallback_count == 0


def test_build_sample_batch_falls_back_to_parent_label_when_event_labels_missing() -> (
    None
):
    """DeepCase should use the parent label when no event label is available."""
    sequence = _sequence(
        templates=["A", "B", "C", "D", "E"],
        label=1,
    )
    sample_count = len(sequence.events)
    event_id_map = DeepCaseEventIdMap.from_sequences((sequence,))

    batch = build_sample_batch(
        (sequence,),
        event_id_map=event_id_map,
        context_length=2,
        timeout_seconds=86_400,
    )

    assert batch.scores.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert batch.parent_sequence_fallback_count == sample_count


@pytest.mark.allow_no_new_coverage
def test_build_sample_batch_replaces_stale_context_events() -> None:
    """A target event should only keep context events within the timeout."""
    # This protects the timeout replacement rule for stale context rows; the
    # surrounding batch-building paths are already exercised by neighbouring
    # DeepCase tests, so a separate uncovered branch is not the right fit.
    sequence = _sequence(
        templates=["A", "B", "C"],
        event_metadata={"dts_by_event": [None, 1_000, 2_000]},
    )
    event_id_map = DeepCaseEventIdMap.from_sequences((sequence,))

    batch = build_sample_batch(
        (sequence,),
        event_id_map=event_id_map,
        context_length=2,
        timeout_seconds=2.5,
    )

    no_event = event_id_map.no_event_id
    a_id = event_id_map.template_to_event_id["A"]
    b_id = event_id_map.template_to_event_id["B"]
    assert batch.contexts.tolist() == [
        [no_event, no_event],
        [no_event, a_id],
        [no_event, b_id],
    ]


def test_build_sample_batch_uses_train_vocabulary_for_unknown_events() -> None:
    """Unseen prediction templates should keep their original train id unset."""
    train_sequence = _sequence(templates=["A"])
    test_sequence = _sequence(templates=["B"], split_label=SplitLabel.TEST)
    event_id_map = DeepCaseEventIdMap.from_sequences((train_sequence,))

    batch = build_sample_batch(
        (test_sequence,),
        event_id_map=event_id_map,
        context_length=2,
        timeout_seconds=86_400,
        unknown_event_id=event_id_map.no_event_id,
    )

    assert batch.events.tolist() == [event_id_map.no_event_id]
    assert batch.templates == ["B"]
    assert batch.original_event_ids == [None]


def test_build_training_batch_matches_direct_sequence_batch() -> None:
    """Training should reuse one train vocabulary while keeping sample rows aligned."""
    sequences = (
        _sequence(
            templates=["A", "B", "C"],
            event_metadata={"dts_by_event": [None, 1_000, 2_000]},
        ),
        _sequence(templates=["B", "A"], label=1),
    )

    event_id_map, batch = build_training_batch(
        sequences,
        context_length=2,
        timeout_seconds=2.5,
    )

    a_id = event_id_map.template_to_event_id["A"]
    b_id = event_id_map.template_to_event_id["B"]
    c_id = event_id_map.template_to_event_id["C"]
    assert event_id_map.template_to_event_id == {"A": a_id, "B": b_id, "C": c_id}
    assert batch.contexts.tolist() == [
        [event_id_map.no_event_id, event_id_map.no_event_id],
        [event_id_map.no_event_id, a_id],
        [event_id_map.no_event_id, b_id],
        [event_id_map.no_event_id, event_id_map.no_event_id],
        [event_id_map.no_event_id, b_id],
    ]
    assert batch.events.tolist() == [a_id, b_id, c_id, b_id, a_id]
    assert batch.scores.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert batch.original_event_ids == [a_id, b_id, c_id, b_id, a_id]


def test_build_training_batch_materialises_sequence_templates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepCase training should read each sequence's templates a single time.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces DeepCase helpers with
            counters so the test can observe how often the hot path resolves
            sequence-level inputs.
    """
    event_labels = (0, 1, 0, 1)
    sequence = TemplateSequence(
        events=[("A", [], 1), ("B", [], 1), ("C", [], 1), ("D", [], 1)],
        label=1,
        entity_ids=["entity-1"],
        window_id=0,
        event_labels=event_labels,
    )
    templates_access_count = 0
    original_templates = TemplateSequence.templates.fget
    assert original_templates is not None
    label_resolution_count = 0
    original_is_anomalous_label = deepcase_shared.is_anomalous_label

    def _counting_templates(
        self: TemplateSequence,
    ) -> list[str]:
        nonlocal templates_access_count
        templates_access_count += 1
        return original_templates(self)

    def _counting_is_anomalous_label(label: int) -> bool:
        nonlocal label_resolution_count
        label_resolution_count += 1
        return original_is_anomalous_label(label)

    monkeypatch.setattr(TemplateSequence, "templates", property(_counting_templates))
    monkeypatch.setattr(
        deepcase_shared,
        "is_anomalous_label",
        _counting_is_anomalous_label,
    )

    _, batch = build_training_batch(
        (sequence,),
        context_length=2,
        timeout_seconds=2.5,
    )

    assert templates_access_count == 1
    assert label_resolution_count == len(sequence.events) + 1
    assert batch.sample_count == len(sequence.events)
    assert batch.parent_sequence_fallback_count == 0


def test_build_training_batch_uses_target_event_labels_when_available() -> None:
    """DeepCase training should also avoid smearing parent labels."""
    sequences = (
        _sequence(
            templates=["A", "B", "C", "D", "E"],
            event_metadata={"event_labels": [0, 0, 1, 0, 0]},
            label=1,
        ),
    )

    _, batch = build_training_batch(
        sequences,
        context_length=2,
        timeout_seconds=2.5,
    )

    assert batch.scores.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert batch.parent_sequence_fallback_count == 0


def test_build_training_batch_respects_explicit_training_event_mask() -> None:
    """DeepCase should only train on events flagged for training."""
    sequence = TemplateSequence(
        events=[
            ("A", [], None),
            ("B", [], None),
            ("C", [], None),
        ],
        label=1,
        entity_ids=["entity-1"],
        window_id=0,
        training_event_mask=(True, False, True),
        event_labels=(0, 1, 1),
    )

    _, batch = build_training_batch(
        (sequence,),
        context_length=2,
        timeout_seconds=2.5,
    )

    assert batch.event_indexes == [0, 2]
    assert batch.templates == ["A", "C"]
    assert batch.scores.tolist() == [0.0, 1.0]
    assert batch.parent_sequence_fallback_count == 0


def test_build_training_batch_falls_back_to_parent_label_when_labels_missing() -> None:
    """Training should keep the parent label only when event labels are absent."""
    sequence = _sequence(
        templates=["A", "B", "C", "D", "E"],
        label=1,
    )

    _, batch = build_training_batch(
        (sequence,),
        context_length=2,
        timeout_seconds=2.5,
    )

    assert batch.scores.tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]
    assert batch.parent_sequence_fallback_count == len(sequence.events)


def test_deepcase_rejects_multi_entity_sequences() -> None:
    """DeepCase should reject contexts that already mix entity ids."""
    detector = _deep_case_config(
        name="deepcase",
        epochs=1,
        iterations=100,
    ).build_detector()
    sequence = _sequence(templates=["A"], entity_ids=["one", "two"])

    with (
        Progress(disable=True) as progress,
        pytest.raises(
            ValueError,
            match="entity-local",
        ),
    ):
        detector.fit((sequence,), progress=progress)


def test_deepcase_score_mapping_is_conservative() -> None:
    """DeepCase special codes and positive scores should map to alerts."""
    assert finding_reason_for_score(0.0) is ScoreReason.KNOWN_BENIGN_CLUSTER
    assert finding_reason_for_score(2.0) is ScoreReason.KNOWN_MALICIOUS_CLUSTER
    assert finding_reason_for_score(-1.0) is ScoreReason.NOT_CONFIDENT_ENOUGH
    assert (
        finding_reason_for_score(-2.0) is ScoreReason.EVENT_NOT_IN_TRAINING_VOCABULARY
    )
    assert finding_reason_for_score(-3.0) is ScoreReason.CLOSEST_CLUSTER_OUTSIDE_EPSILON
    assert decision_label_for_score(0.0) == 0
    assert decision_label_for_score(1.0) == 1
    assert decision_label_for_score(-1.0) == 0
    assert finding_reason_for_score(-1.0).is_abstained
    assert aggregate_sequence_score([0.0, MALICIOUS_SCORE, -1.0]) == MALICIOUS_SCORE
    assert aggregate_sequence_score([0.0, -1.0]) == pytest.approx(0.0)
    assert not aggregate_sequence_score([0.0, 0.0])


def test_deepcase_workload_reduction_metrics_apply_paper_formulas() -> None:
    """DeepCase workload metrics should use the paper's alert formulas."""
    manual = build_workload_reduction_metrics(
        mode=DeepCaseWorkloadMode.MANUAL,
        total_contextual_sequence_count=100,
        covered_contextual_sequence_count=80,
        uncovered_contextual_sequence_count=20,
        alert_sampling=DeepCaseWorkloadAlertSampling(
            cluster_count=EXPECTED_WORKLOAD_CLUSTER_COUNT,
            alerts_per_cluster=EXPECTED_WORKLOAD_ALERTS_PER_CLUSTER,
        ),
    )
    semi_automatic = build_workload_reduction_metrics(
        mode=DeepCaseWorkloadMode.SEMI_AUTOMATIC,
        total_contextual_sequence_count=100,
        covered_contextual_sequence_count=80,
        uncovered_contextual_sequence_count=20,
        alert_sampling=DeepCaseWorkloadAlertSampling(
            cluster_count=EXPECTED_WORKLOAD_CLUSTER_COUNT,
            alerts_per_cluster=EXPECTED_WORKLOAD_ALERTS_PER_CLUSTER,
        ),
    )

    assert manual.mode is DeepCaseWorkloadMode.MANUAL
    assert manual.cluster_count == EXPECTED_WORKLOAD_CLUSTER_COUNT
    assert manual.alerts_per_cluster == EXPECTED_WORKLOAD_ALERTS_PER_CLUSTER
    assert manual.alert_count == EXPECTED_WORKLOAD_ALERT_COUNT
    assert manual.coverage == EXPECTED_WORKLOAD_COVERAGE
    assert manual.reduction == EXPECTED_WORKLOAD_REDUCTION
    assert manual.overall == EXPECTED_WORKLOAD_OVERALL
    assert semi_automatic.mode is DeepCaseWorkloadMode.SEMI_AUTOMATIC
    assert semi_automatic.cluster_count == EXPECTED_WORKLOAD_CLUSTER_COUNT
    assert semi_automatic.alert_count is None
    assert semi_automatic.coverage == EXPECTED_WORKLOAD_COVERAGE
    assert semi_automatic.reduction == EXPECTED_WORKLOAD_SEMI_AUTOMATIC_REDUCTION
    assert semi_automatic.overall == EXPECTED_WORKLOAD_SEMI_AUTOMATIC_OVERALL


@pytest.mark.parametrize(
    ("raw_score", "true_label", "expected_field"),
    [
        (0.0, 0, "event_tn"),
        (0.0, 1, "event_fn"),
        (MALICIOUS_SCORE, 1, "event_tp"),
        (MALICIOUS_SCORE, 0, "event_fp"),
    ],
)
def test_deepcase_event_decision_metrics_cover_auto_confusion_cells(
    monkeypatch: pytest.MonkeyPatch,
    raw_score: float,
    true_label: int,
    expected_field: str,
) -> None:
    """DeepCase should map confident event decisions onto the confusion matrix.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the batch prediction hook.
        raw_score (float): Synthetic DeepCASE score for the single event.
        true_label (int): Ground-truth label for the single event.
        expected_field (str): The confusion-matrix field expected to hit one.
    """
    diagnostics = _build_single_event_decision_diagnostics(
        monkeypatch,
        raw_score=raw_score,
        true_label=true_label,
    )

    event_metrics = diagnostics.event_decision_metrics
    assert getattr(event_metrics, expected_field) == 1
    assert event_metrics.event_auto_decision_count == 1
    assert event_metrics.event_abstained_decision_count == 0
    assert event_metrics.event_count == 1
    assert event_metrics.event_auto_coverage == pytest.approx(1.0)
    assert event_metrics.event_abstain_rate == pytest.approx(0.0)


def test_deepcase_event_decision_metrics_count_abstentions_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepCase should count abstained events outside the confusion matrix.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the batch prediction hook.
    """
    diagnostics = _build_single_event_decision_diagnostics(
        monkeypatch,
        raw_score=-1.0,
        true_label=1,
    )

    event_metrics = diagnostics.event_decision_metrics
    assert event_metrics.event_count == 1
    assert event_metrics.event_auto_decision_count == 0
    assert event_metrics.event_abstained_decision_count == 1
    assert event_metrics.event_tp == 0
    assert event_metrics.event_fp == 0
    assert event_metrics.event_tn == 0
    assert event_metrics.event_fn == 0
    assert event_metrics.event_auto_coverage == pytest.approx(0.0)
    assert event_metrics.event_abstain_rate == pytest.approx(1.0)
    assert event_metrics.event_precision == pytest.approx(0.0)
    assert event_metrics.event_recall == pytest.approx(0.0)
    assert event_metrics.event_f1 == pytest.approx(0.0)
    assert event_metrics.event_accuracy == pytest.approx(0.0)


def test_deepcase_prediction_diagnostics_track_event_decision_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepCase diagnostics should expose the event-level decision metrics block.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces batch prediction with
            deterministic fixture scores.
    """
    diagnostics = _build_event_decision_diagnostics(monkeypatch)
    _assert_event_decision_diagnostics(diagnostics)


def test_deepcase_predict_aggregates_event_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepCase sequence predictions should preserve event-level raw scores.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces batch prediction so the test
            can isolate AnomaLog's sequence-level aggregation logic.
    """
    train_sequence = _sequence(templates=["A", "B"])
    test_sequence = _sequence(
        templates=["A", "UNSEEN"],
        split_label=SplitLabel.TEST,
    )
    detector = DeepCaseDetector(
        config=_deep_case_config(name="deepcase", epochs=1, iterations=100),
    )
    detector.event_id_map = DeepCaseEventIdMap.from_sequences((train_sequence,))
    detector.model = DeepCASE(features=len(detector.event_id_map.event_id_to_template))

    def _fake_predict_batch(_batch: DeepCaseSampleBatch) -> list[float]:
        del _batch
        return [0.0, UNKNOWN_EVENT_SCORE]

    monkeypatch.setattr(detector, "_predict_batch", _fake_predict_batch)

    outcome = detector.predict(test_sequence)

    assert outcome.predicted_label == 0
    assert outcome.score == pytest.approx(0.0)
    assert outcome.sequence_decision is DeepCaseSequenceDecision.ABSTAINED
    assert outcome.confident_event_count == 1
    assert outcome.abstained_event_count == 1
    assert outcome.confident_anomaly_event_count == 0
    assert [finding.raw_score for finding in outcome.findings] == [
        0.0,
        UNKNOWN_EVENT_SCORE,
    ]
    assert outcome.findings[1].event_id is None
    assert outcome.findings[1].reason is ScoreReason.EVENT_NOT_IN_TRAINING_VOCABULARY
    assert not outcome.findings[1].predicted_label
    assert outcome.findings[1].is_abstained


def test_deepcase_predict_marks_all_abstained_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepCase should mark sequences with only abstained findings as abstained.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces batch prediction so the test
            can isolate the sequence-level abstain decision.
    """
    train_sequence = _sequence(templates=["A", "B"])
    test_sequence = _sequence(
        templates=["A", "UNSEEN"],
        split_label=SplitLabel.TEST,
    )
    detector = DeepCaseDetector(
        config=_deep_case_config(name="deepcase", epochs=1, iterations=100),
    )
    detector.event_id_map = DeepCaseEventIdMap.from_sequences((train_sequence,))
    detector.model = DeepCASE(features=len(detector.event_id_map.event_id_to_template))

    def _fake_predict_batch(_batch: DeepCaseSampleBatch) -> list[float]:
        del _batch
        return [-1.0, -2.0]

    monkeypatch.setattr(detector, "_predict_batch", _fake_predict_batch)

    outcome = detector.predict(test_sequence)

    expected_abstained_event_count = len(test_sequence.events)
    assert outcome.is_abstained
    assert outcome.sequence_decision is DeepCaseSequenceDecision.ABSTAINED
    assert outcome.predicted_label == 0
    assert outcome.confident_event_count == 0
    assert outcome.abstained_event_count == expected_abstained_event_count
    assert outcome.confident_anomaly_event_count == 0


def test_deepcase_run_metrics_reports_parent_sequence_fallback_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepCase run metrics should report parent-label fallback usage.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the upstream DeepCASE fit
            hooks so the test can isolate fallback-count reporting.
    """
    train_sequence = _sequence(templates=["A", "B"])
    test_sequence = _sequence(
        templates=["A", "B", "C", "D", "E"],
        label=1,
        split_label=SplitLabel.TEST,
    )
    detector = DeepCaseDetector(
        config=_deep_case_config(name="deepcase", epochs=1, iterations=100),
    )
    detector.event_id_map = DeepCaseEventIdMap.from_sequences((train_sequence,))
    detector.model = DeepCASE(features=len(detector.event_id_map.event_id_to_template))

    def _fake_context_predict(
        self: ContextBuilder,
        *,
        steps: int = 1,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del self, kwargs
        assert steps == 1
        confidence = torch.full((5, 1, 2), 0.5, dtype=torch.float32)
        attention = torch.zeros((5, 1, 2), dtype=torch.float32)
        return confidence, attention

    monkeypatch.setattr(ContextBuilder, "predict", _fake_context_predict)
    monkeypatch.setattr(
        detector,
        "_predict_batch",
        lambda batch: [0.0] * len(batch.events),
    )

    detector.predict(test_sequence)
    metrics = detector.run_metrics(
        run_metrics={
            "test_sequence_count": 1,
            "counted_predictions": 1,
            "abstained_prediction_count": 0,
            "ignored_sequence_count": 0,
        },
    )
    assert metrics.parent_sequence_fallback_count == len(test_sequence.events)
    assert metrics.prediction_diagnostics is not None


def _assert_next_event_prediction_metrics(
    *,
    next_event_prediction: NextEventPredictionDiagnostics | None,
    metrics: DeepCaseRunMetrics,
    expected_events_seen: int,
    expected_eligible_events: int,
) -> None:
    """Assert the next-event comparison and workload blocks for one run.

    Args:
        next_event_prediction (NextEventPredictionDiagnostics | None):
            Next-event diagnostics captured from the run.
        metrics (DeepCaseRunMetrics): DeepCASE run metrics for the same run.
        expected_events_seen (int): Expected sample count in the run.
        expected_eligible_events (int): Expected number of eligible next-event
            samples.
    """
    assert next_event_prediction is not None
    assert next_event_prediction.task == "next_event_prediction"
    totals = next_event_prediction.totals
    top_k = next_event_prediction.top_k
    exclusions = next_event_prediction.exclusions
    macro = next_event_prediction.classification_top1_macro
    weighted = next_event_prediction.classification_top1_weighted
    table_iv = next_event_prediction.table_iv_prediction_metrics
    assert totals.events_seen == expected_events_seen
    assert totals.events_eligible == expected_eligible_events
    assert totals.coverage == pytest.approx(1.0)
    assert top_k.k_values == [1, 2, 3, 5]
    assert top_k.hit_count == {
        "1": 2,
        "2": 2,
        "3": 2,
        "5": 2,
    }
    assert top_k.accuracy == {
        "1": pytest.approx(2 / 3),
        "2": pytest.approx(2 / 3),
        "3": pytest.approx(2 / 3),
        "5": pytest.approx(2 / 3),
    }
    assert macro.precision == pytest.approx(2 / 3)
    assert macro.recall == pytest.approx(2 / 3)
    assert macro.f1 == pytest.approx(2 / 3)
    assert macro.accuracy == pytest.approx(2 / 3)
    assert weighted.precision == pytest.approx(2 / 3)
    assert weighted.recall == pytest.approx(2 / 3)
    assert weighted.f1 == pytest.approx(2 / 3)
    assert weighted.accuracy == pytest.approx(2 / 3)
    assert table_iv.precision == pytest.approx(weighted.precision)
    assert table_iv.recall == pytest.approx(weighted.recall)
    assert table_iv.f1 == pytest.approx(weighted.f1)
    assert table_iv.accuracy == pytest.approx(weighted.accuracy)
    assert exclusions.insufficient_history == 0
    assert exclusions.unknown_history == 0
    assert exclusions.unknown_target == 0
    assert next_event_prediction.vocabulary_policy is VocabularyPolicy.FULL_DATASET
    assert metrics.auto_decision_count == 1
    assert metrics.abstained_prediction_count == 0
    assert metrics.auto_coverage == pytest.approx(1.0)
    assert metrics.abstain_rate == pytest.approx(0.0)
    assert metrics.random_seed == 0
    assert metrics.manual_workload_reduction is not None
    assert metrics.manual_workload_reduction.mode is DeepCaseWorkloadMode.MANUAL
    assert metrics.manual_workload_reduction.alert_count == 0
    assert metrics.semi_automatic_workload_reduction is not None
    assert (
        metrics.semi_automatic_workload_reduction.mode
        is DeepCaseWorkloadMode.SEMI_AUTOMATIC
    )


def test_deepcase_run_metrics_reports_next_event_prediction_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepCase run metrics should expose Context Builder next-event diagnostics.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the upstream prediction
            hooks so the test can observe next-event diagnostics directly.
    """
    train_sequence = _sequence(templates=["A", "B", "C"])
    test_sequence = _sequence(
        templates=["A", "UNSEEN", "C"],
        split_label=SplitLabel.TEST,
    )
    detector = DeepCaseDetector(
        config=_deep_case_config(
            name="deepcase",
            context_length=2,
            epochs=1,
            iterations=100,
        ),
    )
    detector.event_id_map = DeepCaseEventIdMap.from_sequences((train_sequence,))
    detector.model = DeepCASE(features=len(detector.event_id_map.event_id_to_template))

    def _fake_context_predict(
        self: ContextBuilder,
        *,
        steps: int = 1,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = kwargs["X"]
        del self, x
        assert steps == 1
        confidence = torch.tensor(
            [
                [[0.7, 0.2, 0.1, 0.0]],
                [[0.1, 0.7, 0.1, 0.1]],
                [[0.1, 0.1, 0.7, 0.1]],
            ],
            dtype=torch.float32,
        )
        attention = torch.zeros((3, 1, 2), dtype=torch.float32)
        return confidence, attention

    def _fake_predict_batch(_batch: DeepCaseSampleBatch) -> list[float]:
        return [0.0, 0.0, 0.0]

    monkeypatch.setattr(ContextBuilder, "predict", _fake_context_predict)
    monkeypatch.setattr(detector, "_predict_batch", _fake_predict_batch)

    detector.predict(test_sequence)
    metrics = detector.run_metrics(
        run_metrics={
            "test_sequence_count": 1,
            "counted_predictions": 1,
            "abstained_prediction_count": 0,
            "ignored_sequence_count": 0,
        },
    )
    next_event_prediction = metrics.next_event_prediction
    expected_events_seen = len(test_sequence.events)
    expected_eligible_events = 3
    _assert_next_event_prediction_metrics(
        next_event_prediction=next_event_prediction,
        metrics=metrics,
        expected_events_seen=expected_events_seen,
        expected_eligible_events=expected_eligible_events,
    )


def test_deepcase_manifest_reports_cluster_score_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepCase manifests should expose fitted cluster score diagnostics.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the upstream fit stage so
            the test can control the fitted cluster score assignments.
    """
    train_sequences = (
        _sequence(templates=["A"], label=0),
        _sequence(templates=["B"], label=1),
        _sequence(templates=["C"], label=0),
        _sequence(templates=["D"], label=1),
    )
    detector = DeepCaseDetector(
        config=_deep_case_config(
            name="deepcase",
            epochs=1,
            iterations=100,
        ),
    )

    def _fit_context_builder(self: ContextBuilder, **kwargs: object) -> ContextBuilder:
        del kwargs
        return self

    def _fit_interpreter(self: Interpreter, **kwargs: object) -> Interpreter:
        del kwargs
        self.clusters = np.array([0, -1, 1, 2])
        self.labels = {
            0: {0: 0.0},
            1: {0: 1.0},
            2: {0: 0.0},
        }
        return self

    monkeypatch.setattr(ContextBuilder, "fit", _fit_context_builder)
    monkeypatch.setattr(Interpreter, "fit", _fit_interpreter)

    with Progress(disable=True) as progress:
        detector.fit(train_sequences, progress=progress)

    manifest = detector.model_manifest(
        sequence_summary=SequenceSummary(
            sequence_count=4,
            train_sequence_count=4,
            test_sequence_count=0,
            train_label_counts={0: 2, 1: 2},
            test_label_counts={},
        ),
    )

    expected_benign_cluster_count = 2
    expected_malicious_cluster_count = 1
    expected_unknown_cluster_score_count = 1
    assert manifest.known_benign_cluster_count == expected_benign_cluster_count
    assert manifest.known_malicious_cluster_count == expected_malicious_cluster_count
    assert manifest.unknown_cluster_score_count == expected_unknown_cluster_score_count


def test_deepcase_predict_all_batches_multiple_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepCase bulk scoring should collapse multiple sequences into one batch.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces batch prediction so the test
            can verify sequence regrouping around one shared model call.
    """
    train_sequence = _sequence(templates=["A", "B", "C"])
    test_sequences = [
        _sequence(templates=["A", "B"], split_label=SplitLabel.TEST),
        _sequence(templates=["C"], split_label=SplitLabel.TEST),
    ]
    detector = DeepCaseDetector(
        config=_deep_case_config(name="deepcase", epochs=1, iterations=100),
    )
    detector.event_id_map = DeepCaseEventIdMap.from_sequences((train_sequence,))
    detector.model = DeepCASE(features=len(detector.event_id_map.event_id_to_template))
    predict_batch_sizes: list[int] = []

    def _fake_predict_batch(batch: DeepCaseSampleBatch) -> list[float]:
        predict_batch_sizes.append(batch.sample_count)
        return [0.0, MALICIOUS_SCORE, 0.0]

    monkeypatch.setattr(detector, "_predict_batch", _fake_predict_batch)

    outcomes = list(detector.predict_all(test_sequences))

    assert predict_batch_sizes == [3]
    assert [len(sequence.events) for sequence, _ in outcomes] == [2, 1]
    assert [outcome.predicted_label for _, outcome in outcomes] == [1, 0]
    assert [outcome.sequence_decision for _, outcome in outcomes] == [
        DeepCaseSequenceDecision.CONFIDENT_ANOMALY,
        DeepCaseSequenceDecision.CONFIDENT_NORMAL,
    ]
    assert [outcome.score for _, outcome in outcomes] == [MALICIOUS_SCORE, 0.0]
    assert [
        [finding.raw_score for finding in outcome.findings] for _, outcome in outcomes
    ] == [
        [0.0, MALICIOUS_SCORE],
        [0.0],
    ]
    assert [outcome.abstained_event_count for _, outcome in outcomes] == [0, 0]


def test_deepcase_next_event_predictions_reset_between_bulk_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepCase bulk next-event diagnostics should reflect only the latest run.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the upstream prediction
            hooks so the test can isolate the diagnostic reset behaviour.
    """
    train_sequence = _sequence(templates=["A", "B", "C"])
    detector = DeepCaseDetector(
        config=_deep_case_config(
            name="deepcase",
            context_length=2,
            epochs=1,
            iterations=100,
        ),
    )
    detector.event_id_map = DeepCaseEventIdMap.from_sequences((train_sequence,))
    detector.model = DeepCASE(features=len(detector.event_id_map.event_id_to_template))

    def _fake_context_predict(
        self: ContextBuilder,
        *,
        steps: int = 1,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = kwargs["X"]
        del self, x
        assert steps == 1
        confidence = torch.tensor(
            [
                [[0.8, 0.1, 0.1, 0.0]],
                [[0.1, 0.8, 0.1, 0.0]],
                [[0.1, 0.1, 0.8, 0.0]],
            ],
            dtype=torch.float32,
        )
        attention = torch.zeros((3, 1, 2), dtype=torch.float32)
        return confidence, attention

    def _fake_predict_batch(_batch: DeepCaseSampleBatch) -> list[float]:
        return [0.0, 0.0, 0.0]

    monkeypatch.setattr(ContextBuilder, "predict", _fake_context_predict)
    monkeypatch.setattr(detector, "_predict_batch", _fake_predict_batch)

    first_sequence = _sequence(templates=["A", "B"], split_label=SplitLabel.TEST)
    second_sequence = _sequence(
        templates=["A", "B", "C"],
        split_label=SplitLabel.TEST,
    )

    list(detector.predict_all((first_sequence,)))
    first_metrics = detector.run_metrics(
        run_metrics={
            "test_sequence_count": 1,
            "counted_predictions": 1,
            "abstained_prediction_count": 0,
            "ignored_sequence_count": 0,
        },
    )
    list(detector.predict_all((second_sequence,)))
    second_metrics = detector.run_metrics(
        run_metrics={
            "test_sequence_count": 1,
            "counted_predictions": 1,
            "abstained_prediction_count": 0,
            "ignored_sequence_count": 0,
        },
    )

    assert first_metrics.next_event_prediction is not None
    assert first_metrics.next_event_prediction.totals.events_seen == len(
        first_sequence.events,
    )
    assert second_metrics.next_event_prediction is not None
    assert second_metrics.next_event_prediction.totals.events_seen == len(
        second_sequence.events,
    )


def test_deepcase_predict_batch_uses_config_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepCase scoring should forward the configured query iteration budget.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the upstream predict method so
            the test can verify the forwarded iteration budget.
    """
    train_sequence = _sequence(templates=["A", "B"])
    detector = DeepCaseDetector(
        config=_deep_case_config(name="deepcase", epochs=1, iterations=7),
    )
    detector.event_id_map = DeepCaseEventIdMap.from_sequences((train_sequence,))
    detector.model = DeepCASE(features=len(detector.event_id_map.event_id_to_template))
    captured_iterations: list[int] = []

    def _fake_predict(self: DeepCASE, **kwargs: object) -> np.ndarray:
        del self
        iterations = kwargs["iterations"]
        assert isinstance(iterations, int)
        captured_iterations.append(iterations)
        return np.array([0.0, 0.0], dtype=float)

    monkeypatch.setattr(DeepCASE, "predict", _fake_predict)

    outcome = detector.predict(train_sequence)

    assert captured_iterations == [7]
    assert outcome.score == pytest.approx(0.0)
