"""Tests for the DeepCase experiment integration."""

from __future__ import annotations

import msgspec
import numpy as np
import pytest
from deepcase import DeepCASE
from deepcase.context_builder.context_builder import ContextBuilder
from deepcase.interpreter.interpreter import Interpreter
from rich.progress import Progress

from anomalog.sequences import SplitLabel, TemplateSequence
from experiments import ConfigError
from experiments.models import resolve_model_config_type
from experiments.models.base import decode_experiment_model_config
from experiments.models.deepcase import DeepCaseModelConfig
from experiments.models.deepcase.detector import DeepCaseDetector
from experiments.models.deepcase.shared import (
    DeepCaseEventIdMap,
    DeepCaseSampleBatch,
    ScoreReason,
    aggregate_sequence_score,
    build_sample_batch,
    build_training_batch,
    finding_reason_for_score,
    label_for_score,
)

MALICIOUS_SCORE = 3.0
MANUAL_REVIEW_SCORE = 1.0
UNKNOWN_EVENT_SCORE = -2.0
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
    dts_by_event: list[int | None] | None = None,
    label: int = 0,
    entity_ids: list[str] | None = None,
    split_label: SplitLabel = SplitLabel.TRAIN,
) -> TemplateSequence:
    resolved_dts = dts_by_event or [None for _ in templates]
    return TemplateSequence(
        events=[
            (template, [], dt_prev_ms)
            for template, dt_prev_ms in zip(templates, resolved_dts, strict=True)
        ],
        label=label,
        entity_ids=["entity-1"] if entity_ids is None else entity_ids,
        window_id=0,
        split_label=split_label,
    )


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
    ).build_detector()
    sequence = _sequence(templates=["A", "B"])

    with Progress(disable=True) as progress:
        detector.fit((sequence,), progress=progress)

    assert captured_delta == [0.25]


def test_deepcase_detector_rejects_repeated_fit() -> None:
    """DeepCase detector instances should only be trained once."""
    detector = _deep_case_config(
        name="deepcase",
        epochs=1,
        iterations=0,
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
    assert batch.events.tolist() == [
        event_id_map.template_to_event_id["A"],
        event_id_map.template_to_event_id["B"],
        event_id_map.template_to_event_id["C"],
    ]


@pytest.mark.allow_no_new_coverage
def test_build_sample_batch_replaces_stale_context_events() -> None:
    """A target event should only keep context events within the timeout."""
    # This protects the timeout replacement rule for stale context rows; the
    # surrounding batch-building paths are already exercised by neighbouring
    # DeepCase tests, so a separate uncovered branch is not the right fit.
    sequence = _sequence(
        templates=["A", "B", "C"],
        dts_by_event=[None, 1_000, 2_000],
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
        _sequence(templates=["A", "B", "C"], dts_by_event=[None, 1_000, 2_000]),
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


def test_deepcase_rejects_multi_entity_sequences() -> None:
    """DeepCase should reject contexts that already mix entity ids."""
    detector = _deep_case_config(
        name="deepcase",
        epochs=1,
        iterations=0,
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
    assert finding_reason_for_score(0.0) == "known_benign_cluster"
    assert finding_reason_for_score(2.0) == "known_malicious_cluster"
    assert finding_reason_for_score(-1.0) == "not_confident_enough"
    assert finding_reason_for_score(-2.0) == "event_not_in_training_vocabulary"
    assert finding_reason_for_score(-3.0) == "closest_cluster_outside_epsilon"
    assert label_for_score(0.0) == 0
    assert label_for_score(1.0) == 1
    assert label_for_score(-1.0) == 1
    assert aggregate_sequence_score([0.0, MALICIOUS_SCORE, -1.0]) == MALICIOUS_SCORE
    assert aggregate_sequence_score([0.0, -1.0]) == MANUAL_REVIEW_SCORE
    assert not aggregate_sequence_score([0.0, 0.0])


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
        config=_deep_case_config(name="deepcase", epochs=1, iterations=0),
    )
    detector.event_id_map = DeepCaseEventIdMap.from_sequences((train_sequence,))
    detector.model = DeepCASE(features=len(detector.event_id_map.event_id_to_template))

    def _fake_predict_batch(_batch: DeepCaseSampleBatch) -> list[float]:
        del _batch
        return [0.0, UNKNOWN_EVENT_SCORE]

    monkeypatch.setattr(detector, "_predict_batch", _fake_predict_batch)

    outcome = detector.predict(test_sequence)

    assert outcome.predicted_label == 1
    assert outcome.score == MANUAL_REVIEW_SCORE
    assert [finding.raw_score for finding in outcome.findings] == [
        0.0,
        UNKNOWN_EVENT_SCORE,
    ]
    assert outcome.findings[1].event_id is None
    assert outcome.findings[1].reason is ScoreReason.EVENT_NOT_IN_TRAINING_VOCABULARY


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
        config=_deep_case_config(name="deepcase", epochs=1, iterations=0),
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
    assert [outcome.score for _, outcome in outcomes] == [MALICIOUS_SCORE, 0.0]
    assert [
        [finding.raw_score for finding in outcome.findings] for _, outcome in outcomes
    ] == [
        [0.0, MALICIOUS_SCORE],
        [0.0],
    ]
