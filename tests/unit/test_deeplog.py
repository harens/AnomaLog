# ruff: noqa: PLR0913, PLR2004
"""Tests for the scoped DeepLog experiment implementation."""

from __future__ import annotations

import msgspec
import pytest
import torch
from rich.progress import Progress, TaskID

from anomalog.sequences import SplitLabel, TemplateSequence
from experiments import ConfigError
from experiments.models.base import SequenceSummary, decode_experiment_model_config
from experiments.models.deeplog.detector import DeepLogDetector, DeepLogModelConfig
from experiments.models.deeplog.key import (
    KeyScoringContext,
    fit_key_model,
    iter_key_examples,
    score_key_sequence,
)
from experiments.models.deeplog.parameters import (
    build_parameter_datasets,
    build_parameter_schemas,
    fit_gaussian_threshold,
    fit_parameter_models,
    masked_mse,
    masked_regression_loss,
    raw_parameter_vector_for_event,
)
from experiments.models.deeplog.shared import (
    DeepLogManifest,
    GaussianThreshold,
    KeyLSTM,
    NormalisationStats,
    NormalTrainingCorpus,
    ParameterFeatureSchema,
    ParameterLSTM,
    ParameterModelState,
    build_normal_training_corpus,
)
from experiments.models.next_event_metrics import (
    NextEventPredictionState,
    VocabularyPolicy,
)

# DeepLog lives in `experiments/`, outside the configured `--cov=anomalog`
# target. These tests still protect the experiment-layer detector contract.
pytestmark = pytest.mark.allow_no_new_coverage
ConfigValue = str | int | float | bool | None


def _deep_log_config(**values: ConfigValue) -> DeepLogModelConfig:
    raw_config = {"name": "deeplog", **values}
    try:
        return decode_experiment_model_config(
            raw_config,
            config_type=DeepLogModelConfig,
        )
    except ConfigError:
        raise
    except (msgspec.ValidationError, msgspec.DecodeError, TypeError, ValueError) as exc:
        raise ConfigError(str(exc)) from exc


def test_deeplog_model_config_defaults_next_event_policy() -> None:
    """DeepLog should default next-event diagnostics to full-dataset scope."""
    config = _deep_log_config(name="deeplog")

    assert config.vocabulary_policy is VocabularyPolicy.FULL_DATASET


def test_deeplog_model_config_accepts_full_dataset_next_event_policy() -> None:
    """DeepLog should decode the full-dataset policy for diagnostics."""
    config = _deep_log_config(name="deeplog", vocabulary_policy="full_dataset")

    assert config.vocabulary_policy is VocabularyPolicy.FULL_DATASET


def test_deeplog_model_config_defaults_parameter_detection_enabled() -> None:
    """DeepLog should enable parameter scoring by default."""
    config = _deep_log_config(name="deeplog")

    assert config.parameter_detection_enabled is True


def test_deeplog_model_config_accepts_parameter_detection_disabled() -> None:
    """DeepLog should decode explicit key-only HDFS reproduction configs."""
    config = _deep_log_config(
        name="deeplog",
        parameter_detection_enabled=False,
    )

    assert config.parameter_detection_enabled is False


def _sequence(
    *,
    templates: list[str],
    params_by_event: list[list[str]] | None = None,
    dts_by_event: list[int | None] | None = None,
    label: int = 0,
    split_label: SplitLabel = SplitLabel.TEST,
    event_labels: tuple[int | None, ...] | None = None,
    training_event_mask: tuple[bool, ...] | None = None,
    evaluation_event_mask: tuple[bool, ...] | None = None,
) -> TemplateSequence:
    resolved_params = params_by_event or [[] for _ in templates]
    resolved_dts = dts_by_event or [None for _ in templates]
    return TemplateSequence(
        events=[
            (template, params, dt_prev_ms)
            for template, params, dt_prev_ms in zip(
                templates,
                resolved_params,
                resolved_dts,
                strict=True,
            )
        ],
        label=label,
        entity_ids=["entity-1"],
        window_id=0,
        split_label=split_label,
        event_labels=event_labels,
        training_event_mask=training_event_mask,
        evaluation_event_mask=evaluation_event_mask,
    )


class _StaticKeyModel(KeyLSTM):
    """Deterministic key-model stub for focused scoring tests.

    Args:
        logits (list[float]): One logit per template id in the fake vocabulary.
    """

    def __init__(self, logits: list[float]) -> None:
        super().__init__(vocab_size=len(logits), hidden_size=1, num_layers=1)
        self._logits = torch.tensor([logits], dtype=torch.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Repeat the configured logits for each batch item.

        Args:
            inputs (torch.Tensor): Batched inputs.

        Returns:
            torch.Tensor: Repeated logits.
        """
        return self._logits.repeat(inputs.shape[0], 1)


class _StaticParameterModel(ParameterLSTM):
    """Deterministic parameter-model stub for focused scoring tests.

    Args:
        output_vector (list[float]): Per-feature prediction returned regardless
            of the provided history.
    """

    def __init__(self, output_vector: list[float]) -> None:
        super().__init__(
            input_size=len(output_vector),
            hidden_size=1,
            num_layers=1,
            output_size=len(output_vector),
        )
        self._output_vector = torch.tensor([output_vector], dtype=torch.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Repeat the configured prediction for each batch item.

        Args:
            inputs (torch.Tensor): Batched inputs.

        Returns:
            torch.Tensor: Repeated parameter-vector predictions.
        """
        return self._output_vector.repeat(inputs.shape[0], 1)


def _key_context(*, model: KeyLSTM, top_g: int) -> KeyScoringContext:
    template_to_index = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
    }
    return KeyScoringContext(
        model=model,
        template_to_index=template_to_index,
        index_to_template={
            index: template for template, index in template_to_index.items()
        },
        history_size=2,
        top_g=top_g,
    )


def test_score_key_sequence_uses_ranked_top_g_candidates() -> None:
    """Observed keys inside the top-`g` candidate set should stay normal."""
    sequence = _sequence(templates=["A", "B", "D"])
    model = _StaticKeyModel(logits=[-5.0, -5.0, 2.0, 1.0])

    top_two = score_key_sequence(
        sequence=sequence,
        context=_key_context(model=model, top_g=2),
    )[2]
    top_one = score_key_sequence(
        sequence=sequence,
        context=_key_context(model=model, top_g=1),
    )[2]

    assert top_two.is_anomalous is False
    assert top_one.is_anomalous is True
    assert [prediction.template for prediction in top_two.top_predictions] == ["C", "D"]


def test_score_key_sequence_marks_oov_targets_as_anomalous() -> None:
    """Unseen inference-time templates should be flagged as key anomalies."""
    sequence = _sequence(templates=["A", "B", "UNSEEN"])
    finding = score_key_sequence(
        sequence=sequence,
        context=_key_context(
            model=_StaticKeyModel(logits=[-5.0, -5.0, 3.0, -5.0]),
            top_g=2,
        ),
    )[2]

    assert finding.is_oov is True
    assert finding.is_anomalous is True
    assert finding.actual_probability is None
    assert finding.unknown_history_templates == []


def test_score_key_sequence_records_oov_history_templates() -> None:
    """Unknown history templates should be surfaced and treated as anomalies."""
    sequence = _sequence(templates=["UNSEEN", "B", "C"])
    finding = score_key_sequence(
        sequence=sequence,
        context=_key_context(
            model=_StaticKeyModel(logits=[-5.0, -5.0, 3.0, -5.0]),
            top_g=1,
        ),
    )[2]

    assert finding.is_oov is False
    assert finding.is_anomalous is True
    assert finding.actual_template == "C"
    assert finding.unknown_history_templates == ["UNSEEN"]
    assert finding.actual_probability is None
    assert finding.top_predictions == []


def test_iter_key_examples_yields_sliding_windows_per_sequence() -> None:
    """Key examples should slide within each sequence without crossing boundaries."""
    examples = list(
        iter_key_examples(
            sequences=[
                _sequence(templates=["A", "B", "C", "D"], split_label=SplitLabel.TRAIN),
                _sequence(templates=["B", "C", "D"], split_label=SplitLabel.TRAIN),
            ],
            template_to_index={"A": 0, "B": 1, "C": 2, "D": 3},
            history_size=2,
        ),
    )

    assert examples == [
        ([0, 1], 2),
        ([1, 2], 3),
        ([1, 2], 3),
    ]


def test_iter_key_examples_starts_target_at_history_boundary() -> None:
    """The first DeepLog target should be the event immediately after history."""
    sequence = _sequence(
        templates=[f"T{i}" for i in range(11)],
        split_label=SplitLabel.TRAIN,
    )
    template_to_index = {f"T{i}": i for i in range(11)}

    examples = list(
        iter_key_examples(
            sequences=[sequence],
            template_to_index=template_to_index,
            history_size=10,
        ),
    )

    assert examples == [
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10),
    ]


def test_iter_key_examples_skips_sequences_shorter_than_history_window() -> None:
    """Sequences without a full history-target pair should yield no examples."""
    examples = list(
        iter_key_examples(
            sequences=[
                _sequence(templates=["A", "B"], split_label=SplitLabel.TRAIN),
                _sequence(templates=["C"], split_label=SplitLabel.TRAIN),
            ],
            template_to_index={"A": 0, "B": 1, "C": 2},
            history_size=2,
        ),
    )

    assert examples == []


def test_iter_key_examples_respects_eligible_target_indexes() -> None:
    """Training eligibility should filter key-model targets without breaking context."""
    examples = list(
        iter_key_examples(
            sequences=[
                TemplateSequence(
                    events=[
                        ("A", [], None),
                        ("B", [], None),
                        ("C", [], None),
                        ("D", [], None),
                    ],
                    label=1,
                    entity_ids=["entity-1"],
                    window_id=0,
                    split_label=SplitLabel.TRAIN,
                    training_event_mask=(False, True, True, False),
                ),
            ],
            template_to_index={"A": 0, "B": 1, "C": 2, "D": 3},
            history_size=2,
            eligible_target_indexes={1, 2},
        ),
    )

    assert examples == [
        ([0, 1], 2),
    ]


def test_fit_key_model_reports_example_preparation_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepLog key training should surface example preparation before epochs.

    Args:
        monkeypatch (pytest.MonkeyPatch): Records progress callbacks during
            key-model fitting.
    """
    added_tasks: list[str] = []
    advanced_tasks: list[int] = []
    removed_tasks: list[int] = []
    original_add_task = Progress.add_task
    original_advance = Progress.advance
    original_remove_task = Progress.remove_task

    def _add_task(
        self: Progress,
        description: str,
        *,
        total: float | None = None,
    ) -> TaskID:
        added_tasks.append(description)
        return original_add_task(self, description, total=total)

    def _advance(self: Progress, task_id: TaskID) -> None:
        advanced_tasks.append(task_id)
        return original_advance(self, task_id)

    def _remove_task(self: Progress, task_id: TaskID) -> None:
        removed_tasks.append(task_id)
        return original_remove_task(self, task_id)

    monkeypatch.setattr(Progress, "add_task", _add_task)
    monkeypatch.setattr(Progress, "advance", _advance)
    monkeypatch.setattr(Progress, "remove_task", _remove_task)

    corpus = NormalTrainingCorpus(
        sequences=(
            _sequence(
                templates=["A", "B", "C"],
                split_label=SplitLabel.TRAIN,
            ),
            _sequence(
                templates=["B", "C", "A"],
                split_label=SplitLabel.TRAIN,
            ),
        ),
        templates=("A", "B", "C"),
        event_count=6,
    )

    config = _deep_log_config(
        name="deeplog",
        history_size=1,
        epochs=1,
        batch_size=1,
        hidden_size=4,
        num_layers=1,
    )
    with Progress(disable=True) as progress:
        fit_key_model(
            training_corpus=corpus,
            config=config,
            device=torch.device("cpu"),
            progress=progress,
        )

    assert added_tasks[0] == "Preparing DeepLog key examples"
    assert "Training DeepLog key model" in added_tasks
    expected_preparation_advances = 2
    assert advanced_tasks.count(0) >= expected_preparation_advances
    assert removed_tasks[0] == 0


def test_fit_parameter_models_reports_schema_preparation_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepLog parameter training should surface schema preparation progress.

    Args:
        monkeypatch (pytest.MonkeyPatch): Records progress callbacks during
            parameter-model fitting.
    """
    added_tasks: list[str] = []
    advanced_tasks: list[int] = []
    removed_tasks: list[int] = []
    original_add_task = Progress.add_task
    original_advance = Progress.advance
    original_remove_task = Progress.remove_task

    def _add_task(
        self: Progress,
        description: str,
        *,
        total: float | None = None,
    ) -> TaskID:
        added_tasks.append(description)
        return original_add_task(self, description, total=total)

    def _advance(self: Progress, task_id: TaskID) -> None:
        advanced_tasks.append(task_id)
        return original_advance(self, task_id)

    def _remove_task(self: Progress, task_id: TaskID) -> None:
        removed_tasks.append(task_id)
        return original_remove_task(self, task_id)

    monkeypatch.setattr(Progress, "add_task", _add_task)
    monkeypatch.setattr(Progress, "advance", _advance)
    monkeypatch.setattr(Progress, "remove_task", _remove_task)
    monkeypatch.setattr(
        "experiments.models.deeplog.parameters.training.fit_parameter_model",
        lambda **_: (None, "skipped"),
    )

    corpus = NormalTrainingCorpus(
        sequences=(
            _sequence(
                templates=["A", "B", "A"],
                params_by_event=[[], [], []],
                split_label=SplitLabel.TRAIN,
            ),
            _sequence(
                templates=["B", "A", "B"],
                params_by_event=[[], [], []],
                split_label=SplitLabel.TRAIN,
            ),
        ),
        templates=("A", "B"),
        event_count=6,
    )

    config = _deep_log_config(
        name="deeplog",
        history_size=1,
        epochs=1,
        batch_size=1,
        hidden_size=4,
        num_layers=1,
    )
    with Progress(disable=True) as progress:
        fit_parameter_models(
            training_corpus=corpus,
            config=config,
            device=torch.device("cpu"),
            progress=progress,
        )

    assert added_tasks[0] == "Preparing DeepLog parameter schemas"
    assert "Training DeepLog parameter models" in added_tasks
    expected_preparation_advances = 2
    assert advanced_tasks.count(0) >= expected_preparation_advances
    assert removed_tasks[0] == 0


def test_build_parameter_schemas_uses_strict_numeric_policy() -> None:
    """Mixed-type parameter positions should be dropped under the strict policy."""
    schemas = build_parameter_schemas(
        normal_sequences=[
            _sequence(
                templates=["T", "T", "NO_DT"],
                params_by_event=[
                    ["3.5", "not-a-number", "7"],
                    ["4.0", "5.0", "still-not-a-number"],
                    ["x"],
                ],
                dts_by_event=[25, 30, None],
            ),
        ],
        include_elapsed_time=True,
    )

    assert schemas["T"].feature_names == ["dt_prev_ms", "param_0"]
    assert schemas["T"].numeric_parameter_positions == [0]
    assert schemas["T"].dropped_parameter_positions == [1, 2]
    assert schemas["NO_DT"].feature_names == []


def test_raw_parameter_vector_for_event_masks_missing_positions() -> None:
    """Missing elapsed time or parameters should be padded and masked out."""
    schema = ParameterFeatureSchema(
        feature_names=["dt_prev_ms", "param_0", "param_2"],
        numeric_parameter_positions=[0, 2],
        include_elapsed_time=True,
        dropped_parameter_positions=[1],
    )

    vector = raw_parameter_vector_for_event(
        parameters=["10.0"],
        dt_prev_ms=None,
        schema=schema,
    )

    assert vector.values == [0.0, 10.0, 0.0]
    assert vector.mask == [False, True, False]


def test_masked_mse_ignores_unobserved_parameter_positions() -> None:
    """Residual MSE should only consider target features that are observed."""
    residual = masked_mse(
        observed=[10.0, 0.0, 5.0],
        predicted=[7.0, 1000.0, 2.0],
        mask=[True, False, True],
    )

    assert residual == pytest.approx(9.0)


def test_masked_regression_loss_ignores_unobserved_target_dimensions() -> None:
    """Parameter training loss should only optimize observed target dimensions."""
    loss = masked_regression_loss(
        outputs=torch.tensor([[1.0, 500.0, 3.0]], dtype=torch.float32),
        targets=torch.tensor([[3.0, 0.0, 3.0]], dtype=torch.float32),
        mask=torch.tensor([[True, False, True]]),
    )

    assert loss.item() == pytest.approx(2.0)


def test_fit_gaussian_threshold_clamps_lower_bound() -> None:
    """Gaussian calibration should produce a non-negative lower residual bound."""
    threshold = fit_gaussian_threshold(
        residuals=[0.0, 0.0, 0.1],
        confidence=0.95,
    )

    assert threshold.mean == pytest.approx(1 / 30)
    assert threshold.stddev > 0
    assert threshold.lower_bound >= 0.0
    assert threshold.upper_bound > threshold.mean


def test_build_parameter_datasets_uses_temporal_tail_validation_split() -> None:
    """Validation pairs should come from the temporal tail of each series."""
    schema = ParameterFeatureSchema(
        feature_names=["param_0"],
        numeric_parameter_positions=[0],
        include_elapsed_time=False,
        dropped_parameter_positions=[],
    )
    train_pairs, validation_pairs, normalisation = build_parameter_datasets(
        normal_sequences=[
            _sequence(
                templates=["T", "T", "T", "T", "T", "T"],
                params_by_event=[
                    ["1.0"],
                    ["2.0"],
                    ["3.0"],
                    ["4.0"],
                    ["5.0"],
                    ["6.0"],
                ],
                split_label=SplitLabel.TRAIN,
            ),
        ],
        template="T",
        schema=schema,
        history_size=2,
        validation_fraction=0.5,
    )

    expected_pair_count = 2
    assert len(train_pairs) == expected_pair_count
    assert len(validation_pairs) == expected_pair_count
    assert normalisation.means == [2.5]
    assert train_pairs[0].raw_target == [3.0]
    assert train_pairs[1].raw_target == [4.0]
    assert validation_pairs[0].raw_target == [5.0]
    assert validation_pairs[1].raw_target == [6.0]


def test_build_parameter_datasets_uses_plain_normalized_history_vectors() -> None:
    """Parameter histories should stay faithful to the paper's plain vectors."""
    expected_validation_pairs = 2
    schema = ParameterFeatureSchema(
        feature_names=["dt_prev_ms", "param_0"],
        numeric_parameter_positions=[0],
        include_elapsed_time=True,
        dropped_parameter_positions=[],
    )
    train_pairs, validation_pairs, _ = build_parameter_datasets(
        normal_sequences=[
            _sequence(
                templates=["T", "T", "T", "T"],
                params_by_event=[["1.0"], ["2.0"], ["3.0"], ["4.0"]],
                dts_by_event=[1, None, 3, 4],
                split_label=SplitLabel.TRAIN,
            ),
        ],
        template="T",
        schema=schema,
        history_size=1,
        validation_fraction=0.34,
    )

    assert len(train_pairs) == 1
    assert len(validation_pairs) == expected_validation_pairs
    assert train_pairs[0].history_inputs[0] == [0.0, -1.0]
    assert validation_pairs[0].history_inputs[0] == [0.0, 1.0]


def test_predict_flags_key_model_anomalies() -> None:
    """Sequence-level DeepLog output should fire when the key model fires."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
        ),
    )
    detector.key_model = _StaticKeyModel(logits=[-5.0, -5.0, 2.0, 1.0])
    assert detector.key_model is not None
    key_context = _key_context(model=detector.key_model, top_g=1)
    detector.template_to_index = key_context.template_to_index
    detector.index_to_template = key_context.index_to_template

    outcome = detector.predict(_sequence(templates=["A", "B", "D"]))

    assert outcome.predicted_label == 1
    assert outcome.triggered_by_key_model is True
    assert outcome.triggered_by_parameter_model is False
    assert outcome.score > 0.0
    assert outcome.findings[0].key_model_finding is not None

    metrics = detector.run_metrics(run_metrics={"test_sequence_count": 1})
    next_event_prediction = metrics.next_event_prediction
    expected_events_seen = len(outcome.findings) + 2
    expected_events_eligible = 1
    expected_insufficient_history = 2
    assert next_event_prediction is not None
    assert next_event_prediction.task == "next_event_prediction"
    totals = next_event_prediction.totals
    top_k = next_event_prediction.top_k
    exclusions = next_event_prediction.exclusions
    assert totals.events_seen == expected_events_seen
    assert totals.events_eligible == expected_events_eligible
    assert totals.coverage == pytest.approx(1 / 3)
    assert top_k.k_values == [1]
    assert top_k.hit_count == {"1": 0}
    assert top_k.accuracy == {"1": 0.0}
    assert exclusions.insufficient_history == expected_insufficient_history
    assert exclusions.unknown_history == 0
    assert exclusions.unknown_target == 0
    assert next_event_prediction.vocabulary_policy is VocabularyPolicy.FULL_DATASET
    assert metrics.next_event_prediction is not None


def test_predict_ignores_parameter_models_when_key_only_reproduction_is_disabled() -> (
    None
):
    """Key-only HDFS reproduction should not surface parameter-triggered anomalies."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
            parameter_detection_enabled=False,
        ),
    )
    detector.key_model = _StaticKeyModel(logits=[-5.0, -5.0, 2.0, 1.0])
    assert detector.key_model is not None
    key_context = _key_context(model=detector.key_model, top_g=1)
    detector.template_to_index = key_context.template_to_index
    detector.index_to_template = key_context.index_to_template
    detector.parameter_models["D"] = ParameterModelState(
        template="D",
        schema=ParameterFeatureSchema(
            feature_names=["param_0"],
            numeric_parameter_positions=[0],
            include_elapsed_time=False,
            dropped_parameter_positions=[],
        ),
        normalisation=NormalisationStats(means=[0.0], stddevs=[1.0]),
        gaussian=GaussianThreshold(
            mean=0.1,
            stddev=0.01,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        model=_StaticParameterModel(output_vector=[100.0]),
    )

    outcome = detector.predict(_sequence(templates=["A", "B", "D"]))
    metrics = detector.run_metrics(run_metrics={"test_sequence_count": 1})

    assert outcome.predicted_label == 1
    assert outcome.triggered_by_key_model is True
    assert outcome.triggered_by_parameter_model is False
    assert all(finding.parameter_model_finding is None for finding in outcome.findings)
    assert metrics.sequence_trigger_breakdown is not None
    assert metrics.sequence_trigger_breakdown.total_sequences == 1
    assert metrics.sequence_trigger_breakdown.normal_sequences == 1
    assert metrics.sequence_trigger_breakdown.anomalous_sequences == 0
    assert metrics.sequence_trigger_breakdown.key_only_normal_sequences == 1
    assert metrics.sequence_trigger_breakdown.parameter_only_normal_sequences == 0
    assert metrics.sequence_trigger_breakdown.both_normal_sequences == 0
    assert metrics.sequence_trigger_breakdown.neither_normal_sequences == 0


def test_predict_excludes_unknown_targets_under_train_only_policy() -> None:
    """Train-only next-event diagnostics should exclude unseen target templates."""
    expected_events_seen = 3
    expected_insufficient_history = 2
    expected_unknown_target = 1
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
            vocabulary_policy="train_only",
        ),
    )
    detector.key_model = _StaticKeyModel(logits=[-5.0, -5.0, 2.0])
    assert detector.key_model is not None
    detector.template_to_index = {
        "A": 0,
        "B": 1,
        "C": 2,
    }
    detector.index_to_template = {
        index: template for template, index in detector.template_to_index.items()
    }

    detector.predict(_sequence(templates=["A", "B", "UNSEEN"]))
    metrics = detector.run_metrics(run_metrics={"test_sequence_count": 1})
    next_event_prediction = metrics.next_event_prediction

    assert next_event_prediction is not None
    assert next_event_prediction.totals.events_seen == expected_events_seen
    assert next_event_prediction.totals.events_eligible == 0
    assert (
        next_event_prediction.exclusions.insufficient_history
        == expected_insufficient_history
    )
    assert next_event_prediction.exclusions.unknown_target == expected_unknown_target
    assert next_event_prediction.exclusions.unknown_history == 0
    assert next_event_prediction.vocabulary_policy is VocabularyPolicy.TRAIN_ONLY


def test_deeplog_sequence_trigger_breakdown_counts_source_combinations() -> None:
    """DeepLog should report whether anomalies came from key, parameter, or both."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
        ),
    )
    detector.parameter_models["T"] = ParameterModelState(
        template="T",
        schema=ParameterFeatureSchema(
            feature_names=["param_0"],
            numeric_parameter_positions=[0],
            include_elapsed_time=False,
            dropped_parameter_positions=[],
        ),
        normalisation=NormalisationStats(means=[0.0], stddevs=[1.0]),
        gaussian=GaussianThreshold(
            mean=0.1,
            stddev=0.01,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        model=_StaticParameterModel(output_vector=[0.0]),
    )

    detector.key_model = _StaticKeyModel(logits=[-5.0, -5.0, 2.0, 1.0])
    assert detector.key_model is not None
    detector.template_to_index = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
    }
    detector.index_to_template = {
        index: template for template, index in detector.template_to_index.items()
    }
    detector.predict(_sequence(templates=["A", "B", "D"], label=0))

    detector.key_model = _StaticKeyModel(logits=[5.0])
    detector.template_to_index = {"T": 0}
    detector.index_to_template = {0: "T"}
    detector.predict(
        _sequence(
            templates=["T", "T", "T", "T"],
            params_by_event=[["10.0"], ["10.0"], ["10.0"], ["10.0"]],
            label=1,
        ),
    )

    detector.key_model = _StaticKeyModel(logits=[1.0, 0.0, 5.0, 4.0])
    detector.template_to_index = {"A": 0, "B": 1, "T": 2, "D": 3}
    detector.index_to_template = {
        index: template for template, index in detector.template_to_index.items()
    }
    detector.predict(
        _sequence(
            templates=["A", "B", "T", "T", "T", "D"],
            params_by_event=[
                ["0.0"],
                ["0.0"],
                ["10.0"],
                ["10.0"],
                ["10.0"],
                ["0.0"],
            ],
            label=1,
        ),
    )

    metrics = detector.run_metrics(run_metrics={"test_sequence_count": 3})
    breakdown = metrics.sequence_trigger_breakdown

    assert breakdown is not None
    assert breakdown.total_sequences == 3
    assert breakdown.normal_sequences == 1
    assert breakdown.anomalous_sequences == 2
    assert breakdown.key_only_normal_sequences == 1
    assert breakdown.key_only_anomalous_sequences == 0
    assert breakdown.parameter_only_normal_sequences == 0
    assert breakdown.parameter_only_anomalous_sequences == 1
    assert breakdown.both_normal_sequences == 0
    assert breakdown.both_anomalous_sequences == 1
    assert breakdown.neither_normal_sequences == 0
    assert breakdown.neither_anomalous_sequences == 0


def test_deeplog_next_event_predictions_accumulate_across_test_sequences() -> None:
    """Run-level DeepLog diagnostics should aggregate all scored test sequences."""
    expected_events_seen = 8
    expected_events_eligible = 4
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=5,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
        ),
    )
    detector.key_model = _StaticKeyModel(logits=[0.1, 0.2, 9.0, 8.0, 7.0, -1.0])
    assert detector.key_model is not None
    detector.template_to_index = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4,
        "F": 5,
    }
    detector.index_to_template = {
        index: template for template, index in detector.template_to_index.items()
    }

    first_sequence = _sequence(templates=["A", "B", "C", "D"])
    second_sequence = _sequence(templates=["B", "C", "D", "F"])

    detector.predict(first_sequence)
    detector.predict(second_sequence)
    metrics = detector.run_metrics(run_metrics={"test_sequence_count": 2})
    next_event_prediction = metrics.next_event_prediction

    assert next_event_prediction is not None
    totals = next_event_prediction.totals
    assert totals.events_seen == expected_events_seen
    assert totals.events_eligible == expected_events_eligible
    assert totals.coverage == pytest.approx(
        expected_events_eligible / expected_events_seen,
    )
    assert next_event_prediction.top_k.hit_count == {
        "1": 1,
        "2": 3,
        "3": 3,
        "5": 3,
    }
    assert next_event_prediction.top_k.accuracy == {
        "1": pytest.approx(1 / 4),
        "2": pytest.approx(3 / 4),
        "3": pytest.approx(3 / 4),
        "5": pytest.approx(3 / 4),
    }


def test_deeplog_event_level_metrics_follow_event_labels() -> None:
    """DeepLog should report event-level precision and recall from line labels."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
        ),
    )
    detector.key_model = _StaticKeyModel(logits=[-5.0, -5.0, 2.0, 1.0])
    assert detector.key_model is not None
    detector.template_to_index = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
    }
    detector.index_to_template = {
        index: template for template, index in detector.template_to_index.items()
    }

    sequence = _sequence(
        templates=["A", "B", "C", "D"],
        event_labels=(0, 0, 0, 1),
    )

    detector.predict(sequence)
    metrics = detector.run_metrics(run_metrics={"test_sequence_count": 1})
    event_metrics = metrics.event_level_detection

    assert event_metrics is not None
    assert event_metrics.task == "event_level_detection"
    assert event_metrics.events_seen == 2
    assert event_metrics.events_eligible == 2
    assert event_metrics.tp == 1
    assert event_metrics.tn == 1
    assert event_metrics.fp == 0
    assert event_metrics.fn == 0
    assert event_metrics.precision == pytest.approx(1.0)
    assert event_metrics.recall == pytest.approx(1.0)
    assert event_metrics.f1 == pytest.approx(1.0)


def test_deeplog_event_level_metrics_use_event_masks_not_chunk_labels() -> None:
    """DeepLog should score mixed chunks by event mask, not by chunk label."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
        ),
    )
    detector.key_model = _StaticKeyModel(logits=[3.0, 2.0, 1.0, 0.0])
    assert detector.key_model is not None
    detector.template_to_index = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
    }
    detector.index_to_template = {
        index: template for template, index in detector.template_to_index.items()
    }

    sequence = _sequence(
        templates=["A", "B", "C", "D"],
        split_label=SplitLabel.TRAIN,
        event_labels=(0, 0, 0, 1),
        training_event_mask=(True, True, True, False),
        evaluation_event_mask=(False, False, False, True),
    )

    detector.predict(sequence)
    metrics = detector.run_metrics(run_metrics={"test_sequence_count": 1})
    event_metrics = metrics.event_level_detection

    assert event_metrics is not None
    assert detector.test_event_count == 1
    assert event_metrics.events_seen == 1
    assert event_metrics.events_eligible == 1
    assert event_metrics.tp == 1
    assert event_metrics.tn == 0
    assert event_metrics.fp == 0
    assert event_metrics.fn == 0


def test_next_event_prediction_state_computes_weighted_metrics_and_exclusions() -> None:
    """Next-event diagnostics should aggregate macro, weighted, and top-k metrics."""
    state = NextEventPredictionState(
        k_values=(1, 2, 5),
        vocabulary_policy=VocabularyPolicy.TRAIN_ONLY,
    )
    state.record_prediction(
        actual_label="A",
        predicted_labels=["A", "B"],
    )
    state.record_prediction(
        actual_label="A",
        predicted_labels=["B", "A"],
    )
    state.record_prediction(
        actual_label="B",
        predicted_labels=["B", "C"],
    )

    snapshot = state.snapshot()

    expected_events_seen = 3
    expected_eligible_events = 3
    assert snapshot is not None
    assert snapshot.totals.events_seen == expected_events_seen
    assert snapshot.totals.events_eligible == expected_eligible_events
    assert snapshot.totals.coverage == pytest.approx(1.0)
    assert snapshot.top_k.hit_count == {"1": 2, "2": 3, "5": 3}
    assert snapshot.top_k.accuracy == {"1": pytest.approx(2 / 3), "2": 1.0, "5": 1.0}
    assert snapshot.classification_top1_macro.precision == pytest.approx(0.75)
    assert snapshot.classification_top1_macro.recall == pytest.approx(0.75)
    assert snapshot.classification_top1_macro.f1 == pytest.approx(2 / 3)
    assert snapshot.classification_top1_macro.accuracy == pytest.approx(2 / 3)
    assert snapshot.classification_top1_weighted.precision == pytest.approx(5 / 6)
    assert snapshot.classification_top1_weighted.recall == pytest.approx(2 / 3)
    assert snapshot.classification_top1_weighted.f1 == pytest.approx(2 / 3)
    assert snapshot.classification_top1_weighted.accuracy == pytest.approx(2 / 3)
    assert snapshot.exclusions.insufficient_history == 0
    assert snapshot.exclusions.unknown_target == 0
    assert snapshot.exclusions.unknown_history == 0
    assert snapshot.vocabulary_policy is VocabularyPolicy.TRAIN_ONLY


def test_next_event_prediction_state_top_k_is_monotonic() -> None:
    """Top-k hit counts should never decrease as k grows."""
    state = NextEventPredictionState(
        k_values=(1, 2, 5),
        vocabulary_policy=VocabularyPolicy.TRAIN_ONLY,
    )
    state.record_prediction(
        actual_label="A",
        predicted_labels=["A", "B", "C", "D", "E"],
    )
    state.record_prediction(
        actual_label="B",
        predicted_labels=["C", "B", "D", "E", "F"],
    )
    state.record_prediction(
        actual_label="C",
        predicted_labels=["D", "E", "C", "F", "G"],
    )

    snapshot = state.snapshot()

    assert snapshot is not None
    assert snapshot.top_k.hit_count == {"1": 1, "2": 2, "5": 3}
    assert snapshot.top_k.accuracy == {
        "1": pytest.approx(1 / 3),
        "2": pytest.approx(2 / 3),
        "5": pytest.approx(1.0),
    }


def test_next_event_prediction_state_supports_k_beyond_candidate_count() -> None:
    """Top-k reporting should tolerate k values larger than the candidate set."""
    state = NextEventPredictionState(
        k_values=(1, 5),
        vocabulary_policy=VocabularyPolicy.TRAIN_ONLY,
    )
    state.record_prediction(
        actual_label="A",
        predicted_labels=["A", "B"],
    )
    state.record_prediction(
        actual_label="B",
        predicted_labels=["B", "A"],
    )

    snapshot = state.snapshot()

    assert snapshot is not None
    assert snapshot.top_k.hit_count == {"1": 2, "5": 2}
    assert snapshot.top_k.accuracy == {"1": 1.0, "5": 1.0}


def test_next_event_prediction_state_applies_vocabulary_policy() -> None:
    """Policy-aware observations should exclude or score samples consistently."""
    train_only = NextEventPredictionState.create(
        k_values=(1,),
        vocabulary_policy=VocabularyPolicy.TRAIN_ONLY,
    )
    train_only.record_observation(
        actual_label="A",
        predicted_labels=["A"],
        target_is_known=False,
    )
    train_only.record_observation(
        actual_label="B",
        predicted_labels=["B"],
        history_is_known=False,
    )

    full_dataset = NextEventPredictionState.create(
        k_values=(1,),
        vocabulary_policy=VocabularyPolicy.FULL_DATASET,
    )
    full_dataset.record_observation(
        actual_label="A",
        predicted_labels=["A"],
        target_is_known=False,
        history_is_known=False,
    )

    train_only_snapshot = train_only.snapshot()
    full_dataset_snapshot = full_dataset.snapshot()

    assert train_only_snapshot is not None
    expected_train_only_events_seen = 2
    expected_full_dataset_events_seen = 1
    assert train_only_snapshot.totals.events_seen == expected_train_only_events_seen
    assert train_only_snapshot.totals.events_eligible == 0
    assert train_only_snapshot.exclusions.unknown_target == 1
    assert train_only_snapshot.exclusions.unknown_history == 1
    assert full_dataset_snapshot is not None
    assert full_dataset_snapshot.totals.events_seen == expected_full_dataset_events_seen
    assert full_dataset_snapshot.totals.events_eligible == 1
    assert full_dataset_snapshot.exclusions.unknown_target == 0
    assert full_dataset_snapshot.exclusions.unknown_history == 0
    assert full_dataset_snapshot.top_k.hit_count == {"1": 1}


def test_deeplog_next_event_predictions_reset_after_run_metrics() -> None:
    """DeepLog next-event diagnostics should reflect the latest scoring run only."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
        ),
    )
    detector.key_model = _StaticKeyModel(logits=[-5.0, -5.0, 2.0, 1.0])
    assert detector.key_model is not None
    detector.template_to_index = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
    }
    detector.index_to_template = {
        index: template for template, index in detector.template_to_index.items()
    }

    first_sequence = _sequence(templates=["A", "B", "C"])
    second_sequence = _sequence(templates=["A", "B", "D", "C"])

    detector.predict(first_sequence)
    first_metrics = detector.run_metrics(run_metrics={"test_sequence_count": 1})
    detector.predict(second_sequence)
    second_metrics = detector.run_metrics(run_metrics={"test_sequence_count": 1})

    assert first_metrics.next_event_prediction is not None
    assert first_metrics.next_event_prediction.totals.events_seen == len(
        first_sequence.events,
    )
    assert second_metrics.next_event_prediction is not None
    assert second_metrics.next_event_prediction.totals.events_seen == len(
        second_sequence.events,
    )


def test_predict_flags_parameter_model_anomalies() -> None:
    """Sequence-level DeepLog output should fire when a parameter model fires."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
        ),
    )
    detector.key_model = _StaticKeyModel(logits=[5.0])
    detector.template_to_index = {"T": 0}
    detector.index_to_template = {0: "T"}
    detector.parameter_models["T"] = ParameterModelState(
        template="T",
        schema=ParameterFeatureSchema(
            feature_names=["param_0"],
            numeric_parameter_positions=[0],
            include_elapsed_time=False,
            dropped_parameter_positions=[],
        ),
        normalisation=NormalisationStats(means=[0.0], stddevs=[1.0]),
        gaussian=GaussianThreshold(
            mean=0.1,
            stddev=0.01,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        model=_StaticParameterModel(output_vector=[0.0]),
    )

    outcome = detector.predict(
        _sequence(
            templates=["T", "T", "T"],
            params_by_event=[["0.0"], ["0.0"], ["10.0"]],
        ),
    )

    assert outcome.predicted_label == 1
    assert outcome.triggered_by_key_model is False
    assert outcome.triggered_by_parameter_model is True
    assert outcome.score == pytest.approx(99.0)
    parameter_finding = outcome.findings[0].parameter_model_finding
    assert parameter_finding is not None
    assert parameter_finding.is_anomalous is True
    assert parameter_finding.most_anomalous_feature == "param_0"


def test_predict_does_not_score_normal_parameter_residuals_as_anomalies() -> None:
    """Normal parameter residuals should not inflate the sequence anomaly score."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
        ),
    )
    detector.key_model = _StaticKeyModel(logits=[5.0])
    detector.template_to_index = {"T": 0}
    detector.index_to_template = {0: "T"}
    detector.parameter_models["T"] = ParameterModelState(
        template="T",
        schema=ParameterFeatureSchema(
            feature_names=["param_0"],
            numeric_parameter_positions=[0],
            include_elapsed_time=False,
            dropped_parameter_positions=[],
        ),
        normalisation=NormalisationStats(means=[0.0], stddevs=[1.0]),
        gaussian=GaussianThreshold(
            mean=5.0,
            stddev=1.0,
            lower_bound=0.0,
            upper_bound=100.0,
        ),
        model=_StaticParameterModel(output_vector=[0.0]),
    )

    outcome = detector.predict(
        _sequence(
            templates=["T", "T", "T"],
            params_by_event=[["0.0"], ["0.0"], ["5.0"]],
        ),
    )

    assert outcome.predicted_label == 0
    assert outcome.score == pytest.approx(0.0)
    assert outcome.triggered_by_parameter_model is False


def test_predict_skips_parameter_scoring_when_key_model_fires() -> None:
    """Parameter scoring should be skipped once the key model fires."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
        ),
    )
    detector.key_model = _StaticKeyModel(logits=[-5.0, -5.0, 2.0, 1.0])
    assert detector.key_model is not None
    key_context = _key_context(model=detector.key_model, top_g=1)
    detector.template_to_index = key_context.template_to_index
    detector.index_to_template = key_context.index_to_template
    detector.parameter_models["D"] = ParameterModelState(
        template="D",
        schema=ParameterFeatureSchema(
            feature_names=["param_0"],
            numeric_parameter_positions=[0],
            include_elapsed_time=False,
            dropped_parameter_positions=[],
        ),
        normalisation=NormalisationStats(means=[0.0], stddevs=[1.0]),
        gaussian=GaussianThreshold(
            mean=0.1,
            stddev=0.01,
            lower_bound=0.0,
            upper_bound=0.2,
        ),
        model=_StaticParameterModel(output_vector=[0.0]),
    )

    outcome = detector.predict(
        _sequence(
            templates=["A", "B", "D"],
            params_by_event=[[], [], ["10.0"]],
        ),
    )

    finding = outcome.findings[0]
    assert outcome.predicted_label == 1
    assert outcome.triggered_by_key_model is True
    assert outcome.triggered_by_parameter_model is False
    assert finding.key_model_finding is not None
    assert finding.parameter_model_finding is None


def test_predict_masks_missing_parameter_values_in_event_findings() -> None:
    """Masked target features should stay `None` in serialised findings."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
        ),
    )
    detector.key_model = _StaticKeyModel(logits=[5.0])
    detector.template_to_index = {"T": 0}
    detector.index_to_template = {0: "T"}
    detector.parameter_models["T"] = ParameterModelState(
        template="T",
        schema=ParameterFeatureSchema(
            feature_names=["dt_prev_ms", "param_0"],
            numeric_parameter_positions=[0],
            include_elapsed_time=True,
            dropped_parameter_positions=[],
        ),
        normalisation=NormalisationStats(means=[0.0, 0.0], stddevs=[1.0, 1.0]),
        gaussian=GaussianThreshold(
            mean=0.1,
            stddev=0.01,
            lower_bound=0.0,
            upper_bound=100.0,
        ),
        model=_StaticParameterModel(output_vector=[9.0, 3.0]),
    )

    outcome = detector.predict(
        _sequence(
            templates=["T", "T", "T"],
            params_by_event=[["1.0"], ["2.0"], ["3.0"]],
            dts_by_event=[1, 2, None],
        ),
    )

    parameter_finding = outcome.findings[0].parameter_model_finding
    assert parameter_finding is not None
    assert parameter_finding.observed_vector == [None, 3.0]
    assert parameter_finding.predicted_vector == [None, 3.0]
    assert parameter_finding.most_anomalous_feature == "param_0"


def test_predict_requires_fit_before_scoring() -> None:
    """Predicting without a fitted key model should fail fast."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
        ),
    )

    with pytest.raises(ValueError, match="must be fit before prediction"):
        detector.predict(_sequence(templates=["A", "B", "C"]))


def test_fit_rejects_train_sets_without_normal_sequences() -> None:
    """DeepLog should refuse training data that lacks normal examples."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
        ),
    )

    progress = Progress(disable=True)
    with pytest.raises(ValueError, match="eligible training target"), progress:
        detector.fit(
            [_sequence(templates=["A", "B", "C"], label=1)],
            progress=progress,
        )


def test_build_normal_training_corpus_keeps_eligible_targets_from_mixed_chunks() -> (
    None
):
    """Mixed chronological chunks should still expose normal training targets."""
    progress = Progress(disable=True)
    sequence = TemplateSequence(
        events=[
            ("A", [], None),
            ("B", [], None),
            ("C", [], None),
            ("D", [], None),
        ],
        label=1,
        entity_ids=["entity-1"],
        window_id=0,
        split_label=SplitLabel.TRAIN,
        event_labels=(1, 0, 0, 1),
        training_event_mask=(False, True, True, False),
    )

    corpus = build_normal_training_corpus([sequence], progress=progress)
    expected_event_count = 2

    assert corpus.event_count == expected_event_count
    assert corpus.templates == ("A", "B", "C", "D")
    assert corpus.sequences == (sequence,)


def test_build_normal_training_corpus_rejects_multi_entity_sequences() -> None:
    """DeepLog should fail fast when a training sequence spans multiple entities."""
    progress = Progress(disable=True)

    with pytest.raises(ValueError, match="entity-local sequences"), progress:
        build_normal_training_corpus(
            [
                TemplateSequence(
                    events=[
                        ("A", [], None),
                        ("B", [], None),
                        ("C", [], None),
                    ],
                    label=0,
                    entity_ids=["entity-1", "entity-2"],
                    window_id=0,
                    split_label=SplitLabel.TRAIN,
                ),
            ],
            progress=progress,
        )


def test_fit_rejects_repeated_training() -> None:
    """DeepLog should only accept a single successful fit per instance."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=1,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
            validation_fraction=0.5,
            device="cpu",
        ),
    )

    with Progress(disable=True) as progress:
        detector.fit(
            [
                _sequence(
                    templates=["A", "B", "C"],
                    params_by_event=[[], [], []],
                    split_label=SplitLabel.TRAIN,
                ),
            ],
            progress=progress,
        )

    with (
        Progress(disable=True) as progress,
        pytest.raises(
            RuntimeError,
            match="can only be fit once",
        ),
    ):
        detector.fit(
            [
                _sequence(
                    templates=["A", "B", "C"],
                    params_by_event=[[], [], []],
                    split_label=SplitLabel.TRAIN,
                ),
            ],
            progress=progress,
        )

    assert detector.key_model is not None
    assert detector.template_to_index
    assert detector.parameter_models is not None


def test_deeplog_config_rejects_unknown_device() -> None:
    """DeepLog configs should reject unsupported device names."""
    with pytest.raises(ConfigError, match="device"):
        _deep_log_config(name="deeplog", device="tpu")


def test_fit_uses_configured_cpu_device_with_progress() -> None:
    """DeepLog fitting should honor an explicit CPU device and Rich progress."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=1,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=2,
            validation_fraction=0.5,
            device="cpu",
        ),
    )

    with Progress(disable=True) as progress:
        detector.fit(
            [
                _sequence(
                    templates=["A", "T", "A", "T", "A", "T"],
                    params_by_event=[[], ["1.0"], [], ["2.0"], [], ["3.0"]],
                    dts_by_event=[None, 10, None, 12, None, 14],
                    split_label=SplitLabel.TRAIN,
                ),
                _sequence(
                    templates=["A", "T", "A", "T", "A", "T"],
                    params_by_event=[[], ["4.0"], [], ["5.0"], [], ["6.0"]],
                    dts_by_event=[None, 15, None, 18, None, 21],
                    split_label=SplitLabel.TRAIN,
                ),
            ],
            progress=progress,
        )

    assert detector.device.type == "cpu"
    assert detector.key_model is not None
    assert "T" in detector.parameter_models


def test_fit_trains_models_and_skips_non_numeric_templates() -> None:
    """Training should build both DeepLog models and record skipped templates."""
    expected_train_event_count = 14
    expected_train_parameter_covered_event_count = 6
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=1,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=2,
            validation_fraction=0.5,
        ),
    )

    with Progress(disable=True) as progress:
        detector.fit(
            [
                _sequence(
                    templates=["A", "T", "A", "T", "A", "T", "SKIP", "SKIP"],
                    params_by_event=[
                        [],
                        ["1.0"],
                        [],
                        ["2.0"],
                        [],
                        ["3.0"],
                        ["x"],
                        ["y"],
                    ],
                    dts_by_event=[None, 10, None, 12, None, 14, None, None],
                    split_label=SplitLabel.TRAIN,
                ),
                _sequence(
                    templates=["A", "T", "A", "T", "A", "T"],
                    params_by_event=[[], ["4.0"], [], ["5.0"], [], ["6.0"]],
                    dts_by_event=[None, 15, None, 18, None, 21],
                    split_label=SplitLabel.TRAIN,
                ),
            ],
            progress=progress,
        )

    assert detector.key_model is not None
    assert "T" in detector.parameter_models
    assert detector.parameter_models["T"].schema.feature_names == [
        "dt_prev_ms",
        "param_0",
    ]
    assert detector.train_event_count == expected_train_event_count
    assert (
        detector.train_parameter_covered_event_count
        == expected_train_parameter_covered_event_count
    )
    assert detector.skipped_parameter_models["SKIP"] == (
        "template has no numeric modelable features"
    )


def test_sequence_prediction_serialises_deeplog_details() -> None:
    """DeepLog sequence predictions should serialise their event-level payload."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
        ),
    )
    detector.key_model = _StaticKeyModel(logits=[-5.0, -5.0, 2.0, 1.0])
    assert detector.key_model is not None
    key_context = _key_context(model=detector.key_model, top_g=1)
    detector.template_to_index = key_context.template_to_index
    detector.index_to_template = key_context.index_to_template
    sequence = _sequence(templates=["A", "B", "D"])

    prediction = detector.predict(sequence).to_prediction_record(sequence)
    encoded = msgspec.json.encode(prediction.to_dict()).decode("utf-8")

    assert '"triggered_by_key_model":true' in encoded
    assert '"findings"' in encoded


def test_deeplog_manifest_reports_parameter_model_metadata() -> None:
    """DeepLog manifests should expose per-template parameter-model metadata."""
    expected_train_covered_event_count = 4
    expected_scored_parameter_event_count = 3
    expected_input_feature_count = 1
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=5,
            batch_size=2,
        ),
    )
    detector.template_to_index = {"T": 0}
    detector.parameter_models["T"] = ParameterModelState(
        template="T",
        schema=ParameterFeatureSchema(
            feature_names=["param_0"],
            numeric_parameter_positions=[0],
            include_elapsed_time=False,
            dropped_parameter_positions=[1],
        ),
        normalisation=NormalisationStats(means=[0.0], stddevs=[1.0]),
        gaussian=GaussianThreshold(
            mean=0.1,
            stddev=0.02,
            lower_bound=0.0,
            upper_bound=0.2,
        ),
        model=_StaticParameterModel(output_vector=[0.0]),
    )
    detector.skipped_parameter_models["SKIPPED"] = (
        "template has no numeric modelable features"
    )
    detector.train_event_count = 10
    detector.train_parameter_covered_event_count = 4
    detector.test_event_count = 8
    detector.scored_parameter_event_count = 3

    manifest = detector.model_manifest(
        sequence_summary=SequenceSummary(
            sequence_count=3,
            train_sequence_count=2,
            test_sequence_count=1,
            train_label_counts={0: 2},
            test_label_counts={0: 1},
        ),
    )

    assert isinstance(manifest, DeepLogManifest)
    assert manifest.detector == "deeplog"
    assert manifest.implementation_scope == "Scoped DeepLog core v1"
    assert manifest.parameter_schema_policy.startswith("strict:")
    assert manifest.parameter_validation_policy.startswith("per-template temporal")
    assert manifest.parameter_detection_enabled is True
    assert manifest.history_size == detector.config.history_size
    assert manifest.trained_parameter_model_count == 1
    assert manifest.skipped_parameter_model_count == 1
    assert (
        manifest.train_parameter_covered_event_count
        == expected_train_covered_event_count
    )
    assert manifest.train_parameter_covered_event_fraction == pytest.approx(0.4)
    assert (
        manifest.scored_parameter_event_count == expected_scored_parameter_event_count
    )
    assert manifest.scored_parameter_event_fraction == pytest.approx(0.375)
    assert manifest.parameter_models[0].template == "T"
    assert manifest.parameter_models[0].feature_count == 1
    assert (
        manifest.parameter_models[0].input_feature_count == expected_input_feature_count
    )
    assert manifest.skipped_parameter_models[0].template == "SKIPPED"
