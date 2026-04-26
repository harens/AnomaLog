"""Contract tests for experiment detectors."""

from __future__ import annotations

import json
import logging
import types
from collections.abc import Callable, Iterable, Sized
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from rich.progress import Progress, TextColumn
from typing_extensions import override

from anomalog.representations import (
    SequenceRepresentationView,
    SequentialRepresentation,
)
from anomalog.sequences import (
    EntitySequenceBuilder,
    SplitLabel,
    TemplateSequence,
)
from experiments import ConfigError
from experiments.models import resolve_model_config_type
from experiments.models.base import (
    ExperimentDetector,
    ModelManifest,
    PredictionOutcome,
    SequencePrediction,
    SequenceSummary,
    decode_experiment_model_config,
)
from experiments.models.deeplog import DeepLogModelConfig
from experiments.models.evaluate import (
    TrainProgressHint,
    fit_detector,
    stream_predictions,
)
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
ConfigValue = str | int | float | bool | None


def _template_frequency_config(**values: ConfigValue) -> TemplateFrequencyModelConfig:
    return decode_experiment_model_config(
        {"name": "template_frequency", **values},
        config_type=TemplateFrequencyModelConfig,
    )


def _naive_bayes_config(**values: ConfigValue) -> NaiveBayesModelConfig:
    return decode_experiment_model_config(
        {"name": "naive_bayes", **values},
        config_type=NaiveBayesModelConfig,
    )


def _river_config(**values: ConfigValue) -> RiverModelConfig:
    return decode_experiment_model_config(
        {"name": "river", **values},
        config_type=RiverModelConfig,
    )


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


def test_fit_detector_wraps_lazy_train_stream_with_known_total() -> None:
    """Fitting should preserve laziness while exposing a train total when known."""

    @dataclass(slots=True)
    class _RecordingDetector(ExperimentDetector):
        detector_name: str = "recording"
        seen_total: int | None = None
        seen_unit: str | None = None
        seen_sequences: list[TemplateSequence] = field(default_factory=list)

        def fit(
            self,
            train_sequences: Iterable[TemplateSequence],
            *,
            progress: Progress,
            logger: logging.Logger | None = None,
        ) -> None:
            del logger
            if isinstance(train_sequences, Sized):
                self.seen_total = len(train_sequences)
            last_column = progress.columns[6]
            if isinstance(last_column, TextColumn):
                self.seen_unit = last_column.text_format
            self.seen_sequences = list(
                progress.track(train_sequences, description="Recording fit"),
            )

        @override
        def predict(self, sequence: TemplateSequence) -> PredictionOutcome:
            del sequence
            return PredictionOutcome(predicted_label=0, score=0.0)

        @override
        def model_manifest(self, *, sequence_summary: SequenceSummary) -> ModelManifest:
            del sequence_summary
            return ModelManifest(
                detector=self.detector_name,
                train_sequence_count=0,
                test_sequence_count=0,
                train_label_counts={},
                test_label_counts={},
            )

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
                anomalous=0,
            ),
            structured_line(
                line_order=1,
                timestamp_unix_ms=1_100,
                entity_id="node-b",
                untemplated_message_text="read block",
                anomalous=0,
            ),
        ],
    )
    sequences = EntitySequenceBuilder(
        sink=sink,
        infer_template=lambda text: (text, []),
        label_for_group=lambda _entity_id: 0,
        train_frac=0.5,
    )
    detector = _RecordingDetector()
    logger = logging.getLogger("tests.fit_detector")

    fit_detector(
        detector=detector,
        train_sequences=(
            sequence
            for sequence in sequences
            if sequence.split_label is SplitLabel.TRAIN
        ),
        logger=logger,
        train_progress_hint=TrainProgressHint(
            total=sequences.train_sequence_count_hint(),
            unit=sequences.train_sequence_count_unit_hint(),
        ),
    )

    assert detector.seen_total == 1
    assert detector.seen_unit == "entities"
    assert [sequence.entity_ids for sequence in detector.seen_sequences] == [["node-a"]]


def test_stream_predictions_only_scores_test_sequences(
    tmp_path: Path,
) -> None:
    """Streaming evaluation should write predictions for the test split only.

    Args:
        tmp_path (Path): Temporary filesystem root for the prediction stream.
    """

    @dataclass(slots=True)
    class _RecordingDetector(ExperimentDetector):
        detector_name: str = "recording"
        predicted_window_ids: list[int] = field(default_factory=list)

        @override
        def fit(
            self,
            train_sequences: Iterable[TemplateSequence],
            *,
            progress: Progress,
            logger: logging.Logger | None = None,
        ) -> None:
            del train_sequences, progress, logger

        @override
        def predict(self, sequence: TemplateSequence) -> PredictionOutcome:
            self.predicted_window_ids.append(sequence.window_id)
            return PredictionOutcome(predicted_label=sequence.label, score=0.0)

        @override
        def model_manifest(self, *, sequence_summary: SequenceSummary) -> ModelManifest:
            del sequence_summary
            return ModelManifest(
                detector=self.detector_name,
                train_sequence_count=0,
                test_sequence_count=0,
                train_label_counts={},
                test_label_counts={},
            )

    sequences = [
        _sequence(1, templates=["train-a"], label=0, split_label=SplitLabel.TRAIN),
        _sequence(2, templates=["train-b"], label=1, split_label=SplitLabel.TRAIN),
        _sequence(3, templates=["test-a"], label=0, split_label=SplitLabel.TEST),
        _sequence(4, templates=["test-b"], label=1, split_label=SplitLabel.TEST),
    ]
    expected_test_window_ids = [sequence.window_id for sequence in sequences[2:]]
    expected_train_sequence_count = sum(
        1 for sequence in sequences if sequence.split_label is SplitLabel.TRAIN
    )
    expected_test_sequence_count = sum(
        1 for sequence in sequences if sequence.split_label is SplitLabel.TEST
    )
    detector = _RecordingDetector()
    predictions_path = tmp_path / "predictions.jsonl"
    logger = logging.getLogger("tests.stream_predictions")

    summary = stream_predictions(
        detector=detector,
        sequence_factory=lambda: iter(sequences),
        predictions_path=predictions_path,
        logger=logger,
    )

    assert detector.predicted_window_ids == expected_test_window_ids
    assert summary.sequence_count == len(sequences)
    assert summary.train_sequence_count == expected_train_sequence_count
    assert summary.test_sequence_count == expected_test_sequence_count
    predictions = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [
        prediction["window_id"] for prediction in predictions
    ] == expected_test_window_ids
    assert [prediction["split_label"] for prediction in predictions] == [
        "test",
        "test",
    ]


@pytest.mark.allow_no_new_coverage
def test_sequence_prediction_to_dict_keeps_detector_fields_nested_once() -> None:
    """Sequence predictions should flatten detector fields without duplication."""
    # This protects the duplicated-field regression; the shared-field branch is
    # already exercised by neighbouring tests in this module.
    sequence = _sequence(
        1,
        templates=["test-a"],
        label=0,
        split_label=SplitLabel.TEST,
    )
    prediction = SequencePrediction.from_sequence(
        sequence,
        outcome=PredictionOutcome(predicted_label=1, score=0.5),
        detector_fields={"key_phrases": ["foo", "bar"]},
    )

    assert prediction.to_shared_dict() == {
        "window_id": 1,
        "split_label": "test",
        "label": 0,
        "predicted_label": 1,
        "score": 0.5,
        "entity_ids": ["entity-1"],
        "event_count": 1,
    }
    assert prediction.to_dict() == {
        "window_id": 1,
        "split_label": "test",
        "label": 0,
        "predicted_label": 1,
        "score": 0.5,
        "entity_ids": ["entity-1"],
        "event_count": 1,
        "key_phrases": ["foo", "bar"],
    }
    assert "detector_fields" not in prediction.to_dict()


def test_stream_predictions_logs_scored_test_sequence_counts(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Score-progress logs should count only scored test sequences.

    Args:
        tmp_path (Path): Temporary filesystem root for the prediction stream.
        caplog (pytest.LogCaptureFixture): Captured log records for the scoring run.
    """

    @dataclass(slots=True)
    class _RecordingDetector(ExperimentDetector):
        detector_name: str = "recording"

        @override
        def fit(
            self,
            train_sequences: Iterable[TemplateSequence],
            *,
            progress: Progress,
            logger: logging.Logger | None = None,
        ) -> None:
            del train_sequences, progress, logger

        @override
        def predict(self, sequence: TemplateSequence) -> PredictionOutcome:
            del sequence
            return PredictionOutcome(predicted_label=0, score=0.0)

        @override
        def model_manifest(self, *, sequence_summary: SequenceSummary) -> ModelManifest:
            del sequence_summary
            return ModelManifest(
                detector=self.detector_name,
                train_sequence_count=0,
                test_sequence_count=0,
                train_label_counts={},
                test_label_counts={},
            )

    sequences = [
        _sequence(
            window_id=index,
            templates=[f"train-{index}"],
            label=0,
            split_label=SplitLabel.TRAIN,
        )
        for index in range(10_000)
    ] + [
        _sequence(
            window_id=10_000 + index,
            templates=[f"test-{index}"],
            label=0,
            split_label=SplitLabel.TEST,
        )
        for index in range(10_000)
    ]
    detector = _RecordingDetector()
    predictions_path = tmp_path / "predictions.jsonl"
    logger = logging.getLogger("tests.stream_predictions_progress")

    with caplog.at_level(logging.INFO, logger=logger.name):
        stream_predictions(
            detector=detector,
            sequence_factory=lambda: iter(sequences),
            predictions_path=predictions_path,
            logger=logger,
        )

    assert "Processed 10000 test sequences for recording detector" in caplog.messages


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


def test_model_registry_imports_requested_model_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Model resolution should import only the requested detector module.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the module importer so the
            test can observe which model module is requested.
    """
    imported_modules: list[str] = []
    fake_name = "template_frequency_lazy_test"
    registration = types.SimpleNamespace(
        module_path="experiments.models.template_frequency",
        config_type_name="TemplateFrequencyModelConfig",
    )

    def _import_module(module_path: str) -> types.SimpleNamespace:
        imported_modules.append(module_path)
        return types.SimpleNamespace(
            TemplateFrequencyModelConfig=TemplateFrequencyModelConfig,
        )

    monkeypatch.setattr(
        "experiments.models.registry._MODEL_REGISTRATIONS",
        {fake_name: registration},
    )
    monkeypatch.setattr("experiments.models.registry.import_module", _import_module)

    resolve_model_config_type.cache_clear()
    try:
        config_type = resolve_model_config_type(fake_name)
    finally:
        resolve_model_config_type.cache_clear()

    assert config_type is TemplateFrequencyModelConfig
    assert imported_modules == ["experiments.models.template_frequency"]


def test_model_registry_reports_missing_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing optional detector dependencies should fail with an install hint.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the module importer so the
            test can simulate a missing backend dependency.
    """
    missing_dependency = ModuleNotFoundError("No module named 'river'")
    missing_dependency.name = "river"
    registration = types.SimpleNamespace(
        module_path="experiments.models.river",
        config_type_name="RiverModelConfig",
    )

    def _import_module(module_path: str) -> types.SimpleNamespace:
        del module_path
        raise missing_dependency

    monkeypatch.setattr(
        "experiments.models.registry._MODEL_REGISTRATIONS",
        {"river": registration},
    )
    monkeypatch.setattr("experiments.models.registry.import_module", _import_module)

    resolve_model_config_type.cache_clear()
    try:
        with pytest.raises(
            ConfigError,
            match=r"Detector 'river' requires optional backend dependencies",
        ):
            resolve_model_config_type("river")
    finally:
        resolve_model_config_type.cache_clear()


@pytest.mark.allow_no_new_coverage
def test_model_configs_reject_direct_construction() -> None:
    """Model configs should be decoded so Annotated constraints are enforced."""
    # This protects the experiment config construction contract outside the
    # configured `anomalog` coverage target.
    with pytest.raises(TypeError, match="must be decoded"):
        TemplateFrequencyModelConfig(name="template_frequency")


@pytest.mark.parametrize(
    "build_detector",
    [
        lambda: _template_frequency_config(name="template_frequency").build_detector(),
        lambda: _naive_bayes_config(name="naive_bayes").build_detector(),
        lambda: _river_config(name="river").build_detector(),
    ],
)
def test_detectors_only_accept_one_successful_fit(
    build_detector: Callable[[], ExperimentDetector],
) -> None:
    """Detector instances should reject repeated fitting.

    Args:
        build_detector (Callable[[], ExperimentDetector]): Factory for the
            detector under test.
    """
    detector = build_detector()

    with Progress(disable=True) as progress:
        detector.fit(_supervised_train_sequences(), progress=progress)

    with (
        Progress(disable=True) as progress,
        pytest.raises(
            RuntimeError,
            match="can only be fit once",
        ),
    ):
        detector.fit(_supervised_train_sequences(), progress=progress)


@pytest.mark.allow_no_new_coverage
def test_template_frequency_detector_predictions_are_repeatable() -> None:
    """Repeated predictions from the fitted baseline should be identical."""
    # This protects experiment-layer detector determinism, which sits outside
    # the configured `anomalog` coverage target.
    detector = _template_frequency_config(name="template_frequency").build_detector()
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
    detector = _template_frequency_config(
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
    detector = _naive_bayes_config(name="naive_bayes").build_detector()
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
    config = _river_config(name="river")
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
        _river_config(
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
    detector = _naive_bayes_config(name="naive_bayes").build_detector()
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
        _river_config(
            name="river-complement",
            estimator="naive_bayes.ComplementNB",
        ).build_detector(),
        RiverDetector,
    )
