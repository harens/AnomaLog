"""Contract tests for experiment detectors."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.progress import Progress

from anomalog.representations import (
    SequenceRepresentationView,
    SequentialRepresentation,
)
from anomalog.sequences import (
    EntitySequenceBuilder,
    SplitLabel,
    TemplateSequence,
)
from experiments.models import resolve_model_config_type
from experiments.models.base import SequenceSummary
from experiments.models.deeplog import DeepLogModelConfig
from experiments.models.naive_bayes import NaiveBayesModelConfig
from experiments.models.river import RiverDetector, RiverModelConfig
from experiments.models.template_frequency import (
    TemplateFrequencyModelConfig,
)
from tests.unit.helpers import (
    InMemoryStructuredSink,
    NullStructuredParser,
    structured_line,
)

REPRESENTATION_VIEW_WINDOW_ID = 0


def _sequence(
    window_id: int,
    *,
    templates: list[str],
    label: int,
    split_label: SplitLabel = SplitLabel.TRAIN,
) -> TemplateSequence:
    return TemplateSequence(
        events=[(template, [], None) for template in templates],
        label=label,
        entity_ids=[f"entity-{window_id}"],
        window_id=window_id,
        split_label=split_label,
    )


def _supervised_train_sequences() -> list[TemplateSequence]:
    return [
        _sequence(1, templates=["normal login", "normal read"], label=0),
        _sequence(2, templates=["normal read", "normal read"], label=0),
        _sequence(3, templates=["panic failure", "disk failure"], label=1),
        _sequence(4, templates=["panic failure", "panic failure"], label=1),
    ]


def test_sequence_representation_view_yields_samples_and_labeled_examples() -> None:
    """Represented sequence views expose samples and `(x, y)` pairs."""
    sink = InMemoryStructuredSink(
        dataset_name="demo",
        raw_dataset_path=Path("raw.log"),
        parser=NullStructuredParser(),
        rows=[
            structured_line(
                line_order=0,
                timestamp_unix_ms=1_000,
                entity_id="node-a",
                untemplated_message_text="login ok",
                anomalous=None,
            ),
            structured_line(
                line_order=1,
                timestamp_unix_ms=1_100,
                entity_id="node-a",
                untemplated_message_text="read block",
                anomalous=None,
            ),
        ],
    )
    builder = EntitySequenceBuilder(
        sink=sink,
        infer_template=lambda text: (text, []),
        label_for_group=lambda _entity_id: 0,
        train_frac=0.0,
    )

    view = SequenceRepresentationView(
        sequences=builder,
        representation=SequentialRepresentation(),
    )

    sample = next(iter(view))
    assert sample.window_id == REPRESENTATION_VIEW_WINDOW_ID
    assert sample.entity_ids == ["node-a"]
    assert sample.split_label is SplitLabel.TEST
    assert sample.as_labeled_example() == (["login ok", "read block"], 0)
    assert list(view.iter_labeled_examples()) == [(["login ok", "read block"], 0)]


@pytest.mark.allow_no_new_coverage
def test_model_registries_resolve_builtins() -> None:
    """Built-in model configs register themselves by detector name."""
    # This protects experiment-layer registry wiring outside the configured
    # `anomalog` coverage target.
    assert resolve_model_config_type("deeplog") is DeepLogModelConfig
    assert resolve_model_config_type("naive_bayes") is NaiveBayesModelConfig
    assert resolve_model_config_type("river") is RiverModelConfig
    assert (
        resolve_model_config_type("template_frequency") is TemplateFrequencyModelConfig
    )


@pytest.mark.allow_no_new_coverage
def test_template_frequency_detector_predictions_are_repeatable() -> None:
    """Repeated predictions from the fitted baseline should be identical."""
    # This protects experiment-layer detector determinism, which sits outside
    # the configured `anomalog` coverage target.
    detector = TemplateFrequencyModelConfig(name="template_frequency").build_detector()
    with Progress(disable=True) as progress:
        detector.fit(_supervised_train_sequences(), progress=progress)
    sequence = _sequence(
        20,
        templates=["panic failure", "disk failure"],
        label=1,
        split_label=SplitLabel.TEST,
    )

    first = detector.predict(sequence)
    second = detector.predict(sequence)

    assert first == second


@pytest.mark.allow_no_new_coverage
def test_template_frequency_detector_learns_threshold_from_normal_train_scores() -> (
    None
):
    """Template-frequency defaults should calibrate against train normal scores."""
    # This protects experiment-layer threshold calibration outside the
    # configured `anomalog` coverage target.
    detector = TemplateFrequencyModelConfig(
        name="template_frequency",
    ).build_detector()
    train_sequences = _supervised_train_sequences()
    with Progress(disable=True) as progress:
        detector.fit(train_sequences, progress=progress)

    normal_scores = [
        detector.score(sequence) for sequence in train_sequences if sequence.label == 0
    ]
    anomalous_sequence = _sequence(
        23,
        templates=["novel panic", "novel panic"],
        label=1,
        split_label=SplitLabel.TEST,
    )

    assert detector.threshold_source == "train_score_quantile"
    assert detector.score_threshold <= max(normal_scores)
    assert detector.predict(train_sequences[0]).predicted_label == 0
    assert detector.predict(anomalous_sequence).predicted_label == 1


@pytest.mark.allow_no_new_coverage
def test_naive_bayes_detector_predictions_are_repeatable() -> None:
    """Repeated predictions from the handwritten detector should be identical."""
    # This protects experiment-layer detector determinism, which sits outside
    # the configured `anomalog` coverage target.
    detector = NaiveBayesModelConfig(name="naive_bayes").build_detector()
    with Progress(disable=True) as progress:
        detector.fit(_supervised_train_sequences(), progress=progress)
    sequence = _sequence(
        21,
        templates=["panic failure", "disk failure"],
        label=1,
        split_label=SplitLabel.TEST,
    )

    first = detector.predict(sequence)
    second = detector.predict(sequence)

    assert first == second


@pytest.mark.allow_no_new_coverage
def test_river_detector_predictions_are_stable_across_equal_fits() -> None:
    """Two fits on the same training data should produce the same prediction."""
    # This protects experiment-layer detector determinism, which sits outside
    # the configured `anomalog` coverage target.
    config = RiverModelConfig(name="river")
    left = config.build_detector()
    right = config.build_detector()
    train_sequences = _supervised_train_sequences()
    sequence = _sequence(
        22,
        templates=["panic failure", "disk failure"],
        label=1,
        split_label=SplitLabel.TEST,
    )

    with Progress(disable=True) as left_progress:
        left.fit(train_sequences, progress=left_progress)
    with Progress(disable=True) as right_progress:
        right.fit(train_sequences, progress=right_progress)

    assert left.predict(sequence) == right.predict(sequence)


@pytest.mark.allow_no_new_coverage
def test_river_model_config_supports_additional_count_based_estimators() -> None:
    """River configs should accept the supported alternate Naive Bayes estimators."""
    # This protects experiment-layer estimator registration outside the
    # configured `anomalog` coverage target.
    assert isinstance(
        RiverModelConfig(
            name="river-bernoulli",
            estimator="naive_bayes.BernoulliNB",
        ).build_detector(),
        RiverDetector,
    )


@pytest.mark.allow_no_new_coverage
def test_naive_bayes_manifest_includes_shared_sequence_summary_fields() -> None:
    """Detector manifests should be typed and include shared run summary counts."""
    # This protects experiment-layer manifest shaping outside the configured
    # `anomalog` coverage target.
    detector = NaiveBayesModelConfig(name="naive_bayes").build_detector()
    with Progress(disable=True) as progress:
        detector.fit(_supervised_train_sequences(), progress=progress)
    sequence_summary = SequenceSummary(
        sequence_count=4,
        train_sequence_count=3,
        test_sequence_count=1,
        train_label_counts={0: 2, 1: 1},
        test_label_counts={0: 1},
    )

    manifest = detector.model_manifest(
        sequence_summary=sequence_summary,
    )

    assert manifest.detector == "naive_bayes"
    assert manifest.train_sequence_count == sequence_summary.train_sequence_count
    assert manifest.test_sequence_count == sequence_summary.test_sequence_count
    assert manifest.train_label_counts == sequence_summary.train_label_counts
    assert manifest.test_label_counts == sequence_summary.test_label_counts
    assert isinstance(
        RiverModelConfig(
            name="river-complement",
            estimator="naive_bayes.ComplementNB",
        ).build_detector(),
        RiverDetector,
    )
