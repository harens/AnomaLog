# ruff: noqa: PLR0913, PLR2004
"""Tests for the scoped DeepLog experiment implementation."""

from __future__ import annotations

import random
from pathlib import Path

import msgspec
import pytest
import torch
from rich.progress import Progress, TaskID

from anomalog.sequences import SplitLabel, TemplateSequence
from experiments import ConfigError
from experiments.config import load_experiment_bundles
from experiments.models.base import SequenceSummary, decode_experiment_model_config
from experiments.models.deeplog import detector as deeplog_detector
from experiments.models.deeplog import key as deeplog_key
from experiments.models.deeplog.detector import (
    DeepLogDetector,
    DeepLogModelConfig,
)
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
    score_parameter_sequence,
)
from experiments.models.deeplog.parameters.reporting import ParameterCiState
from experiments.models.deeplog.shared import (
    DeepLogManifest,
    DeepLogParameterFinding,
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
ConfigValue = (
    str
    | int
    | float
    | bool
    | None
    | tuple[int, ...]
    | tuple[str, ...]
    | list[int]
    | list[str]
)


def _deep_log_config(**values: ConfigValue) -> DeepLogModelConfig:
    raw_config: dict[str, ConfigValue] = {"name": "deeplog"}
    raw_config.update(values)
    top_g = raw_config.pop("top_g", None)
    if "top_g_values" not in raw_config and isinstance(top_g, int):
        raw_config["top_g_values"] = tuple(range(1, top_g + 1))
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


def test_deeplog_model_config_defaults_top_g_values() -> None:
    """DeepLog should default to the paper's replay cut-offs."""
    config = _deep_log_config(name="deeplog")

    assert config.top_g_values == (1, 3, 5, 7, 9)
    assert max(config.top_g_values) == 9


def test_deeplog_model_config_accepts_full_dataset_next_event_policy() -> None:
    """DeepLog should decode the full-dataset policy for diagnostics."""
    config = _deep_log_config(name="deeplog", vocabulary_policy="full_dataset")

    assert config.vocabulary_policy is VocabularyPolicy.FULL_DATASET


def test_deeplog_model_config_defaults_parameter_detection_enabled() -> None:
    """DeepLog should default experiment configs to key-only scoring."""
    config = _deep_log_config(name="deeplog")

    assert config.parameter_detection_enabled is False
    assert config.key_detection_enabled is True


def test_deeplog_model_config_accepts_parameter_ci_highlight_templates() -> None:
    """DeepLog should decode the OpenStack Figure 9 highlight ordering."""
    config = _deep_log_config(
        name="deeplog",
        parameter_ci_highlight_templates=(
            "VM Started (Lifecycle Event)",
            "During sync_power_state the instance has a pending task (spawning). Skip.",
        ),
    )

    assert config.parameter_ci_highlight_templates == (
        "VM Started (Lifecycle Event)",
        "During sync_power_state the instance has a pending task (spawning). Skip.",
    )


def test_deeplog_model_config_accepts_parameter_only_mode() -> None:
    """DeepLog should decode an explicit parameter-only scoring mode."""
    config = _deep_log_config(
        name="deeplog",
        parameter_detection_enabled=True,
        key_detection_enabled=False,
    )

    assert config.parameter_detection_enabled is True
    assert config.key_detection_enabled is False


def test_deeplog_model_config_defaults_short_session_padding_fidelity() -> None:
    """DeepLog should default to the paper-faithful short-session policy."""
    config = _deep_log_config(name="deeplog")

    assert config.short_session_padding_fidelity is False


def test_deeplog_model_config_accepts_parameter_detection_disabled() -> None:
    """DeepLog should decode explicit key-only HDFS reproduction configs."""
    config = _deep_log_config(
        name="deeplog",
        parameter_detection_enabled=False,
    )

    assert config.parameter_detection_enabled is False


def test_deeplog_model_config_rejects_empty_top_g_values() -> None:
    """DeepLog should require at least one replay cut-off."""
    with pytest.raises(ConfigError, match="length >= 1"):
        _deep_log_config(name="deeplog", top_g_values=())


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
    continuous_context: bool = False,
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
        continuous_context=continuous_context,
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


def _reference_one_hot_histories(
    *,
    histories: list[list[int]],
    vocab_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Mirror the reference one-hot encoder used by the DeepLog regression.

    Args:
        histories (list[list[int]]): Encoded history windows.
        vocab_size (int): Number of known key indexes.
        device (torch.device): Device used for tensor materialisation.

    Returns:
        torch.Tensor: One-hot encoded batch with shape
            ``(batch, history_size, vocab_size)``.
    """
    history_index_tensor = torch.tensor(histories, dtype=torch.long, device=device)
    history_tensor = torch.zeros(
        (len(histories), len(histories[0]), vocab_size),
        dtype=torch.float32,
        device=device,
    )
    history_tensor.scatter_(2, history_index_tensor.unsqueeze(-1), 1.0)
    return history_tensor


def _reference_optimise_key_training_batch(
    *,
    model: KeyLSTM,
    optimizer: torch.optim.Optimizer,
    batch_histories: list[list[int]],
    batch_targets: list[int],
    vocab_size: int,
    device: torch.device,
) -> None:
    """Mirror the reference microbatch update path used in the regression.

    Args:
        model (KeyLSTM): Key model being trained.
        optimizer (torch.optim.Optimizer): Optimiser used for the update.
        batch_histories (list[list[int]]): Encoded history windows.
        batch_targets (list[int]): Matching next-key indexes.
        vocab_size (int): Number of known key indexes.
        device (torch.device): Device used for tensor materialisation.
    """
    batch_size = len(batch_histories)
    microbatch_size = min(batch_size, 64)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer.zero_grad()
    for start in range(0, batch_size, microbatch_size):
        end = start + microbatch_size
        microbatch_histories = batch_histories[start:end]
        microbatch_targets = batch_targets[start:end]
        logits = model(
            _reference_one_hot_histories(
                histories=microbatch_histories,
                vocab_size=vocab_size,
                device=device,
            ),
        )
        loss = criterion(
            logits,
            torch.tensor(
                microbatch_targets,
                dtype=torch.long,
                device=device,
            ),
        )
        scaled_loss = loss * (len(microbatch_histories) / batch_size)
        scaled_loss.backward()
    optimizer.step()


def _reference_fit_key_model(
    *,
    training_corpus: NormalTrainingCorpus,
    config: DeepLogModelConfig,
    device: torch.device,
) -> tuple[KeyLSTM, dict[str, int], dict[int, str]]:
    """Mirror the reference streaming key-model trainer used in the regression.

    Args:
        training_corpus (NormalTrainingCorpus): Normal training corpus used to
            build the reference batches.
        config (DeepLogModelConfig): DeepLog configuration.
        device (torch.device): Device used for training.

    Returns:
        tuple[KeyLSTM, dict[str, int], dict[int, str]]: Reference model and
            vocabulary mappings.
    """
    template_to_index = {
        template: idx for idx, template in enumerate(training_corpus.templates)
    }
    index_to_template = {idx: template for template, idx in template_to_index.items()}
    model = KeyLSTM(
        vocab_size=len(template_to_index),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for _ in range(config.epochs):
        model.train()
        batch_histories: list[list[int]] = []
        batch_targets: list[int] = []
        for sequence in random.sample(
            list(training_corpus.sequences),
            k=len(training_corpus.sequences),
        ):
            for history, target in iter_key_examples(
                sequences=(sequence,),
                template_to_index=template_to_index,
                history_size=config.history_size,
            ):
                batch_histories.append(history)
                batch_targets.append(target)
                if len(batch_histories) < config.batch_size:
                    continue
                _reference_optimise_key_training_batch(
                    model=model,
                    optimizer=optimizer,
                    batch_histories=batch_histories,
                    batch_targets=batch_targets,
                    vocab_size=len(template_to_index),
                    device=device,
                )
                batch_histories = []
                batch_targets = []
        if batch_histories:
            _reference_optimise_key_training_batch(
                model=model,
                optimizer=optimizer,
                batch_histories=batch_histories,
                batch_targets=batch_targets,
                vocab_size=len(template_to_index),
                device=device,
            )
    return model.eval(), template_to_index, index_to_template


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
    assert top_two.actual_rank == 2
    assert top_one.actual_rank == 2
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
    assert finding.actual_rank is None
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
    assert finding.actual_rank is None
    assert finding.top_predictions == []


def test_score_key_sequence_carries_prefix_history_across_chunks() -> None:
    """Continuous stream scoring should reuse the prior chunk's key history."""
    sequence = _sequence(templates=["C"])
    model = _StaticKeyModel(logits=[-5.0, -5.0, 3.0, -5.0])

    finding = score_key_sequence(
        sequence=sequence,
        context=_key_context(model=model, top_g=1),
        prefix_templates=["A", "B"],
    )[0]

    assert finding.history_templates == ["A", "B"]
    assert finding.actual_template == "C"
    assert finding.is_anomalous is False


def test_score_key_sequence_uses_local_indexes_for_unknown_prefix_history() -> None:
    """Unknown carried history should still be keyed to the local event index."""
    sequence = _sequence(templates=["C"])
    model = _StaticKeyModel(logits=[-5.0, -5.0, 3.0, -5.0])

    findings = score_key_sequence(
        sequence=sequence,
        context=_key_context(model=model, top_g=1),
        prefix_templates=["UNSEEN", "B"],
    )

    assert 0 in findings
    finding = findings[0]
    assert finding.event_index == 0
    assert finding.history_templates == ["UNSEEN", "B"]
    assert finding.unknown_history_templates == ["UNSEEN"]
    assert finding.is_anomalous is True


def test_score_parameter_sequence_carries_prefix_history_across_chunks() -> None:
    """Continuous stream scoring should reuse the prior chunk's parameter history."""
    schema = ParameterFeatureSchema(
        feature_names=["param_0"],
        numeric_parameter_positions=[0],
        include_elapsed_time=False,
        dropped_parameter_positions=[],
    )
    state = ParameterModelState(
        template="T",
        schema=schema,
        normalisation=NormalisationStats(means=[0.0], stddevs=[1.0]),
        gaussian=GaussianThreshold(
            mean=0.0,
            stddev=1.0,
            lower_bound=0.0,
            upper_bound=10.0,
        ),
        model=_StaticParameterModel(output_vector=[1.0]),
    )

    findings = score_parameter_sequence(
        sequence=_sequence(
            templates=["T"],
            params_by_event=[["1.0"]],
            split_label=SplitLabel.TEST,
        ),
        parameter_models={"T": state},
        history_size=2,
        eligible_event_indexes={0},
        prefix_events_by_template={"T": [(["1.0"], None), (["1.0"], None)]},
    )

    assert 0 in findings
    finding = findings[0]
    assert finding.observed_vector == [1.0]
    assert finding.is_anomalous is False


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


def test_materialise_key_training_examples_caches_history_windows() -> None:
    """Key training should cache per-sequence indexed histories and targets once."""
    materialise_key_training_examples = (
        deeplog_key._materialise_key_training_examples  # noqa: SLF001
    )
    materialisation_config_type = vars(deeplog_key)[
        "_KeyTrainingExampleMaterialisationConfig"
    ]
    sequence_examples = materialise_key_training_examples(
        sequences=[
            _sequence(
                templates=["A", "B", "C", "D"],
                split_label=SplitLabel.TRAIN,
            ),
            _sequence(
                templates=["D", "C"],
                split_label=SplitLabel.TRAIN,
            ),
        ],
        materialisation_config=materialisation_config_type(
            template_to_index={"A": 0, "B": 1, "C": 2, "D": 3},
            history_size=2,
        ),
        progress=None,
        prepare_task=None,
    )

    assert sequence_examples[0].history_windows.tolist() == [[0, 1], [1, 2]]
    assert sequence_examples[0].target_indexes.tolist() == [2, 3]
    assert sequence_examples[1].history_windows.shape == (0, 2)
    assert sequence_examples[1].target_indexes.shape == (0,)
    assert sequence_examples[0].history_windows.device.type == "cpu"
    assert sequence_examples[1].history_windows.device.type == "cpu"


def test_fit_key_model_materialises_one_hot_histories_per_minibatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepLog should materialise one-hot histories when minibatches run.

    Args:
        monkeypatch (pytest.MonkeyPatch): Records one-hot history construction
            during key-model fitting.
    """
    one_hot_shapes: list[tuple[int, int]] = []
    original_one_hot_history_indexes = deeplog_key._one_hot_history_indexes  # noqa: SLF001

    def _recording_one_hot_history_indexes(
        *,
        history_indexes: torch.Tensor,
        vocab_size: int,
    ) -> torch.Tensor:
        one_hot_shapes.append(
            (
                int(history_indexes.shape[0]),
                int(history_indexes.shape[1]),
            ),
        )
        return original_one_hot_history_indexes(
            history_indexes=history_indexes,
            vocab_size=vocab_size,
        )

    monkeypatch.setattr(
        deeplog_key,
        "_one_hot_history_indexes",
        _recording_one_hot_history_indexes,
    )

    corpus = NormalTrainingCorpus(
        sequences=(
            _sequence(
                templates=["A", "B", "C", "D", "E"],
                split_label=SplitLabel.TRAIN,
            ),
        ),
        templates=("A", "B", "C", "D", "E"),
        event_count=5,
    )
    config = _deep_log_config(
        name="deeplog",
        history_size=1,
        epochs=2,
        batch_size=2,
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

    assert one_hot_shapes == [(2, 1), (2, 1), (2, 1), (2, 1)]


def test_move_key_training_batch_to_device_pins_cuda_transfers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DeepLog should pin cached batches before CUDA transfers.

    Args:
        monkeypatch (pytest.MonkeyPatch): Records pinning and transfer calls.
    """
    pin_calls: list[tuple[int, ...]] = []
    to_calls: list[tuple[str | None, bool | None]] = []

    def _pin_memory(self: torch.Tensor) -> torch.Tensor:
        pin_calls.append(tuple(int(size) for size in self.shape))
        return self

    def _to(
        self: torch.Tensor,
        *,
        device: torch.device,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        to_calls.append((device.type, non_blocking))
        return self

    monkeypatch.setattr(torch.Tensor, "pin_memory", _pin_memory, raising=False)
    monkeypatch.setattr(torch.Tensor, "to", _to, raising=False)

    histories = torch.tensor([[1, 0]], dtype=torch.long)
    targets = torch.tensor([1], dtype=torch.long)

    moved_histories, moved_targets = deeplog_key._move_key_training_batch_to_device(  # noqa: SLF001
        batch_histories=histories,
        batch_targets=targets,
        device=torch.device("cuda"),
    )

    assert moved_histories is histories
    assert moved_targets is targets
    assert pin_calls == [(1, 2), (1,)]
    assert to_calls == [("cuda", True), ("cuda", True)]


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


def test_fit_key_model_streams_batches_without_materialising_all_examples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Key-model training should build only minibatch tensors, not the corpus.

    Args:
        monkeypatch (pytest.MonkeyPatch): Records one-hot batch construction.
    """
    batch_sizes: list[int] = []
    original_forward = KeyLSTM.forward

    def _recording_forward(
        self: KeyLSTM,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        batch_sizes.append(int(inputs.shape[0]))
        return original_forward(self, inputs)

    monkeypatch.setattr(KeyLSTM, "forward", _recording_forward)

    corpus = NormalTrainingCorpus(
        sequences=(
            _sequence(
                templates=["A", "B", "C", "D", "E"],
                split_label=SplitLabel.TRAIN,
            ),
        ),
        templates=("A", "B", "C", "D", "E"),
        event_count=5,
    )
    config = _deep_log_config(
        name="deeplog",
        history_size=1,
        epochs=1,
        batch_size=2,
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

    assert batch_sizes == [2, 2]


def test_fit_key_model_splits_large_training_batches_into_microbatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large key-model batches should be processed in smaller GPU-safe chunks.

    Args:
        monkeypatch (pytest.MonkeyPatch): Records key-model batch shapes during
            training.
    """
    batch_sizes: list[int] = []
    original_forward = KeyLSTM.forward

    def _recording_forward(
        self: KeyLSTM,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        batch_sizes.append(int(inputs.shape[0]))
        return original_forward(self, inputs)

    monkeypatch.setattr(KeyLSTM, "forward", _recording_forward)
    monkeypatch.setattr(
        "experiments.models.deeplog.key._KEY_TRAINING_MICROBATCH_SIZE",
        2,
    )

    corpus = NormalTrainingCorpus(
        sequences=(
            _sequence(
                templates=["A", "B", "C", "D", "E", "F"],
                split_label=SplitLabel.TRAIN,
            ),
        ),
        templates=("A", "B", "C", "D", "E", "F"),
        event_count=6,
    )
    config = _deep_log_config(
        name="deeplog",
        history_size=1,
        epochs=1,
        batch_size=5,
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

    assert batch_sizes == [2, 2, 1]


def test_fit_key_model_retries_with_smaller_microbatches_on_cuda_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Key-model training should back off instead of failing on CUDA OOM.

    Args:
        monkeypatch (pytest.MonkeyPatch): Installs a controlled OOM failure in
            the key-model forward path.
    """
    batch_sizes: list[int] = []

    monkeypatch.setattr(deeplog_key, "_KEY_TRAINING_MICROBATCH_SIZE", 4)

    corpus = NormalTrainingCorpus(
        sequences=(
            _sequence(
                templates=["A", "B", "C", "D", "E", "F"],
                split_label=SplitLabel.TRAIN,
            ),
        ),
        templates=("A", "B", "C", "D", "E", "F"),
        event_count=6,
    )
    config = _deep_log_config(
        name="deeplog",
        history_size=1,
        epochs=1,
        batch_size=5,
        hidden_size=4,
        num_layers=1,
    )

    original_forward = KeyLSTM.forward

    def _flaky_forward(
        self: KeyLSTM,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        batch_sizes.append(int(inputs.shape[0]))
        if inputs.shape[0] > 2:
            msg = "CUDA out of memory. Tried to allocate 1.00 GiB."
            raise RuntimeError(msg)
        return original_forward(self, inputs)

    monkeypatch.setattr(KeyLSTM, "forward", _flaky_forward)

    with Progress(disable=True) as progress:
        fit_key_model(
            training_corpus=corpus,
            config=config,
            device=torch.device("cpu"),
            progress=progress,
        )

    assert batch_sizes == [4, 2, 2, 1]


def test_fit_key_model_matches_reference_streaming_update_path() -> None:
    """The throughput fix should not change the fitted key model materially."""
    corpus = NormalTrainingCorpus(
        sequences=(
            _sequence(
                templates=["A", "B", "C", "D", "E"],
                split_label=SplitLabel.TRAIN,
            ),
            _sequence(
                templates=["B", "C", "D", "E", "A"],
                split_label=SplitLabel.TRAIN,
            ),
        ),
        templates=("A", "B", "C", "D", "E"),
        event_count=10,
    )
    config = _deep_log_config(
        name="deeplog",
        history_size=1,
        epochs=2,
        batch_size=4,
        hidden_size=4,
        num_layers=1,
    )

    random.seed(1234)
    torch.manual_seed(1234)
    reference_model, reference_template_to_index, reference_index_to_template = (
        _reference_fit_key_model(
            training_corpus=corpus,
            config=config,
            device=torch.device("cpu"),
        )
    )
    random.seed(1234)
    torch.manual_seed(1234)
    optimised_model, template_to_index, index_to_template = fit_key_model(
        training_corpus=corpus,
        config=config,
        device=torch.device("cpu"),
        progress=None,
    )

    assert reference_template_to_index == template_to_index
    assert reference_index_to_template == index_to_template
    for reference_param, optimised_param in zip(
        reference_model.parameters(),
        optimised_model.parameters(),
        strict=True,
    ):
        assert torch.allclose(reference_param, optimised_param, atol=5e-3, rtol=1e-5)


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


def test_build_parameter_datasets_merges_continuous_context_series() -> None:
    """Continuous-context sequences should feed one template time series."""
    schema = ParameterFeatureSchema(
        feature_names=["param_0"],
        numeric_parameter_positions=[0],
        include_elapsed_time=False,
        dropped_parameter_positions=[],
    )
    train_pairs, validation_pairs, _ = build_parameter_datasets(
        normal_sequences=[
            _sequence(
                templates=["T", "T", "T"],
                params_by_event=[["1.0"], ["2.0"], ["3.0"]],
                split_label=SplitLabel.TRAIN,
                continuous_context=True,
            ),
            _sequence(
                templates=["T", "T", "T"],
                params_by_event=[["4.0"], ["5.0"], ["6.0"]],
                split_label=SplitLabel.TRAIN,
                continuous_context=True,
            ),
        ],
        template="T",
        schema=schema,
        history_size=2,
        validation_fraction=0.5,
    )

    assert len(train_pairs) == 2
    assert len(validation_pairs) == 2
    assert train_pairs[-1].raw_target == [4.0]
    assert validation_pairs[0].raw_target == [5.0]


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
            parameter_detection_enabled=True,
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


def test_predict_accepts_mixed_entity_chronological_streams() -> None:
    """DeepLog prediction should not require entity-local sequences."""
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

    outcome = detector.predict(
        TemplateSequence(
            events=[
                ("A", [], None),
                ("B", [], None),
                ("D", [], None),
            ],
            label=0,
            entity_ids=["entity-1", "entity-2"],
            window_id=0,
            split_label=SplitLabel.TEST,
        ),
    )

    assert outcome.predicted_label == 1
    assert outcome.triggered_by_key_model is True
    assert outcome.triggered_by_parameter_model is False


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
            parameter_detection_enabled=True,
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


def test_deeplog_sequence_trigger_breakdown_is_disabled_for_stream_batches() -> None:
    """Continuous stream batches should not produce sequence-level aggregation."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
            parameter_detection_enabled=True,
        ),
    )
    detector.key_model = _StaticKeyModel(logits=[-5.0, -5.0, 2.0, 1.0])
    assert detector.key_model is not None
    detector.template_to_index = {
        "A": 0,
        "B": 1,
        "D": 2,
        "T": 3,
    }
    detector.index_to_template = {
        index: template for template, index in detector.template_to_index.items()
    }

    detector.predict(
        _sequence(
            templates=["A", "B", "D"],
            label=0,
            continuous_context=True,
        ),
    )

    metrics = detector.run_metrics(run_metrics={"test_sequence_count": 1})

    assert metrics.sequence_trigger_breakdown is None


def test_deeplog_stream_context_skips_parameter_tails_when_disabled() -> None:
    """Key-only DeepLog should retain only the history needed for scoring."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
            parameter_detection_enabled=False,
        ),
    )
    stream_context = deeplog_detector.DeepLogStreamContext()
    sequence = _sequence(
        templates=["A", "B", "C"],
        continuous_context=True,
    )

    detector._update_stream_context(  # noqa: SLF001
        sequence=sequence,
        stream_context=stream_context,
    )

    assert stream_context.key_templates == ["B", "C"]
    assert stream_context.parameter_events_by_template == {}


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
        "4": 3,
        "5": 3,
    }
    assert next_event_prediction.top_k.accuracy == {
        "1": pytest.approx(1 / 4),
        "2": pytest.approx(3 / 4),
        "3": pytest.approx(3 / 4),
        "4": pytest.approx(3 / 4),
        "5": pytest.approx(3 / 4),
    }


def test_deeplog_top_g_replay_tracks_multiple_g_values_without_refitting() -> None:
    """DeepLog should replay exact-rank key outcomes across all configured g."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g_values=(1, 3, 5),
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
        ),
    )
    detector.key_model = _StaticKeyModel(logits=[3.0, 2.0, 1.0])
    assert detector.key_model is not None
    detector.template_to_index = {
        "A": 0,
        "B": 1,
        "C": 2,
    }
    detector.index_to_template = {
        index: template for template, index in detector.template_to_index.items()
    }

    detector.predict(_sequence(templates=["A", "A", "B"], label=0))
    detector.predict(_sequence(templates=["A", "A", "C"], label=1))
    metrics = detector.run_metrics(run_metrics={"test_sequence_count": 2})
    replay = metrics.top_g_replay

    assert replay is not None
    assert replay.task == "top_g_replay"
    assert replay.configured_top_g == 5
    assert replay.top_g_values == [1, 3, 5]
    assert replay.event_count == 2
    assert replay.sequence_count == 2
    assert replay.normal_sequence_count == 1
    assert replay.anomalous_sequence_count == 1
    assert [point.top_g for point in replay.points] == [1, 3, 5]

    first_point, second_point, third_point = replay.points
    assert first_point.event_hit_count == 0
    assert first_point.event_accuracy == pytest.approx(0.0)
    assert first_point.tp == 1
    assert first_point.tn == 0
    assert first_point.fp == 1
    assert first_point.fn == 0
    assert second_point.top_g == 3
    assert third_point.top_g == 5
    assert first_point.precision == pytest.approx(0.5)
    assert first_point.recall == pytest.approx(1.0)
    assert first_point.f1 == pytest.approx(2 / 3)
    assert first_point.accuracy == pytest.approx(0.5)

    assert second_point.event_hit_count == 2
    assert second_point.event_accuracy == pytest.approx(1.0)
    assert second_point.tp == 0
    assert second_point.tn == 1
    assert second_point.fp == 0
    assert second_point.fn == 1
    assert second_point.precision == pytest.approx(0.0)
    assert second_point.recall == pytest.approx(0.0)
    assert second_point.f1 == pytest.approx(0.0)
    assert second_point.accuracy == pytest.approx(0.5)


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
            parameter_detection_enabled=True,
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
            parameter_detection_enabled=True,
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


def _deeplog_next_event_detector(*, history_size: int) -> DeepLogDetector:
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=history_size,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
        ),
    )
    detector.key_model = _StaticKeyModel(logits=[3.0, 2.0, 1.0])
    detector.template_to_index = {
        "A": 0,
        "B": 1,
        "C": 2,
    }
    detector.index_to_template = {
        index: template for template, index in detector.template_to_index.items()
    }
    return detector


def test_deeplog_next_event_prediction_counts_warmup_for_one_continuous_stream() -> (
    None
):
    """One continuous stream should incur a single history warm-up window."""
    detector = _deeplog_next_event_detector(history_size=10)

    detector.predict(
        _sequence(
            templates=["A"] * 100,
            label=0,
            continuous_context=True,
        ),
    )
    metrics = detector.run_metrics(run_metrics={"test_sequence_count": 1})
    next_event_prediction = metrics.next_event_prediction

    assert next_event_prediction is not None
    assert next_event_prediction.totals.events_seen == 100
    assert next_event_prediction.totals.events_eligible == 90
    assert next_event_prediction.totals.coverage == pytest.approx(0.9)
    assert next_event_prediction.exclusions.insufficient_history == 10
    assert next_event_prediction.segment_diagnostics is not None
    assert next_event_prediction.segment_diagnostics.segment_count == 1
    assert (
        next_event_prediction.segment_diagnostics.expected_insufficient_history_from_segments
        == 10
    )


def test_deeplog_next_event_prediction_counts_independent_segments() -> None:
    """Independent segments should count only eligible warm-up-free events."""
    detector = _deeplog_next_event_detector(history_size=10)

    for length in [100, 50, 5]:
        detector.predict(
            _sequence(
                templates=["A"] * length,
                label=0,
                continuous_context=False,
            ),
        )

    metrics = detector.run_metrics(run_metrics={"test_sequence_count": 3})
    next_event_prediction = metrics.next_event_prediction

    assert next_event_prediction is not None
    assert next_event_prediction.totals.events_seen == 155
    assert next_event_prediction.totals.events_eligible == 130
    assert next_event_prediction.totals.coverage == pytest.approx(130 / 155)
    assert next_event_prediction.exclusions.insufficient_history == 25
    assert next_event_prediction.segment_diagnostics is not None
    assert next_event_prediction.segment_diagnostics.segment_count == 3
    assert (
        next_event_prediction.segment_diagnostics.expected_insufficient_history_from_segments
        == 25
    )


def test_deeplog_short_session_padding_fidelity_scores_last_event() -> None:
    """Legacy short-session fidelity should emit one padded key decision."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=10,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=1,
            short_session_padding_fidelity=True,
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

    outcome = detector.predict(_sequence(templates=["A", "B", "D"]))
    metrics = detector.run_metrics(run_metrics={"test_sequence_count": 1})

    assert outcome.findings
    assert outcome.findings[0].event_index == 2
    assert outcome.findings[0].key_model_finding is not None
    assert metrics.next_event_prediction is not None
    assert metrics.next_event_prediction.totals.events_seen == 3
    assert metrics.next_event_prediction.totals.events_eligible == 1
    assert metrics.next_event_prediction.exclusions.insufficient_history == 2


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
            parameter_detection_enabled=True,
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
            parameter_detection_enabled=True,
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
            parameter_detection_enabled=True,
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
            parameter_detection_enabled=True,
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
            parameter_detection_enabled=True,
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


def test_build_normal_training_corpus_keeps_multi_entity_sequences() -> None:
    """DeepLog should accept mixed-entity chronological streams."""
    progress = Progress(disable=True)

    corpus = build_normal_training_corpus(
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
                training_event_mask=(True, True, False),
            ),
        ],
        progress=progress,
    )

    assert corpus.event_count == 2
    assert corpus.templates == ("A", "B", "C")
    assert corpus.sequences[0].entity_ids == ["entity-1", "entity-2"]


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
            parameter_detection_enabled=True,
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


def test_fit_parameter_only_mode_skips_key_model() -> None:
    """DeepLog should be able to fit the parameter branch without key scoring."""
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
            parameter_detection_enabled=True,
            key_detection_enabled=False,
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

    assert detector.key_model is None
    assert detector.template_to_index == {}
    assert detector.index_to_template == {}
    assert detector.parameter_models


def test_parameter_only_scoring_emits_parameter_ci_report() -> None:
    """Parameter-only DeepLog runs should surface a compact CI report."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=1,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=2,
            device="cpu",
            parameter_detection_enabled=True,
            key_detection_enabled=False,
        ),
    )
    with Progress(disable=True) as progress:
        detector.fit(
            [
                _sequence(
                    templates=["T", "T", "T"],
                    params_by_event=[["1.0"], ["2.0"], ["3.0"]],
                    dts_by_event=[None, 10, 11],
                    split_label=SplitLabel.TRAIN,
                ),
            ],
            progress=progress,
        )
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
            upper_bound=0.2,
        ),
        model=_StaticParameterModel(output_vector=[0.0, 0.0]),
        train_pair_count=4,
        validation_pair_count=2,
    )

    outcome = detector.predict(
        _sequence(
            templates=["T", "T", "T"],
            params_by_event=[["1.0"], ["2.0"], ["3.0"]],
            dts_by_event=[None, 10, 11],
            label=1,
            split_label=SplitLabel.TEST,
        ),
    )
    assert outcome.triggered_by_parameter_model is True

    metrics = detector.run_metrics(
        run_metrics={
            "sequence_count": 1,
            "train_sequence_count": 0,
            "test_sequence_count": 1,
            "ignored_sequence_count": 0,
        },
    )
    assert metrics.parameter_ci_report is not None
    report = metrics.parameter_ci_report
    assert report.paper_approximation is True
    assert report.paper_exact_reproduction is False
    assert "Figure 9" in report.result_note
    assert report.series_scope == "highlighted_subset"
    assert report.highlighted_series_count == 1
    assert report.highlighted_templates == ["T"]
    assert report.total_point_count == 2
    assert report.total_anomalous_point_count == 2
    assert report.series[0].point_count == 2
    assert report.series[0].validation_sample_warning is not None
    assert report.series[0].threshold_summaries[0].confidence == pytest.approx(0.98)
    assert report.series[0].threshold_summaries[0].point_count == 2
    assert report.series[0].threshold_summaries[0].anomalous_point_count == 2
    assert metrics.parameter_ci_trace is not None
    trace = metrics.parameter_ci_trace
    assert trace is not None
    assert "injected overlays" in trace.result_note
    assert trace.series_scope == "highlighted_subset"
    assert trace.highlighted_series_count == 1
    assert trace.total_point_count == 2
    assert trace.total_anomalous_point_count == 2
    assert (
        report.series[0].thresholds.confidence_99
        > report.series[0].thresholds.confidence_98
    )
    assert trace.anomalous_points[0].detected_at_98 in {True, False}


def test_parameter_only_scoring_emits_empty_parameter_ci_report_when_no_points() -> (
    None
):
    """Enabled parameter runs should still serialise an explicit empty report."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=1,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=2,
            parameter_detection_enabled=True,
            key_detection_enabled=False,
        ),
    )
    metrics = detector.run_metrics(
        run_metrics={
            "sequence_count": 1,
            "train_sequence_count": 0,
            "test_sequence_count": 1,
            "ignored_sequence_count": 0,
        },
    )

    assert metrics.parameter_ci_report is not None
    report = metrics.parameter_ci_report
    assert report.series_count == 0
    assert report.highlighted_series_count == 0
    assert report.series_scope == "highlighted_subset"
    assert report.highlighted_templates == []
    assert report.total_point_count == 0
    assert report.total_anomalous_point_count == 0
    assert "Figure 9" in report.result_note
    assert metrics.parameter_ci_trace is not None
    trace = metrics.parameter_ci_trace
    assert trace is not None
    assert trace.series_scope == "highlighted_subset"
    assert trace.highlighted_series_count == 0
    assert trace.series_count == 0
    assert trace.total_point_count == 0
    assert trace.total_anomalous_point_count == 0
    assert "Figure 9" in trace.result_note


def test_parameter_ci_summary_honours_explicit_highlight_order() -> None:
    """Figure 9 summaries should foreground the configured analogue templates."""
    state = ParameterCiState()
    parameter_models = {
        "A": ParameterModelState(
            template="A",
            schema=ParameterFeatureSchema(
                feature_names=["dt_prev_ms"],
                numeric_parameter_positions=[],
                include_elapsed_time=True,
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
            train_pair_count=4,
            validation_pair_count=2,
        ),
        "B": ParameterModelState(
            template="B",
            schema=ParameterFeatureSchema(
                feature_names=["dt_prev_ms"],
                numeric_parameter_positions=[],
                include_elapsed_time=True,
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
            train_pair_count=4,
            validation_pair_count=2,
        ),
    }
    state.record_sequence(
        sequence=_sequence(
            templates=["A", "B"],
            split_label=SplitLabel.TEST,
        ),
        parameter_findings={
            0: DeepLogParameterFinding(
                event_index=0,
                template="A",
                feature_names=["dt_prev_ms"],
                observed_vector=[1.0],
                predicted_vector=[0.5],
                residual_mse=0.1,
                gaussian_mean=0.1,
                gaussian_stddev=0.01,
                gaussian_lower_bound=0.0,
                gaussian_upper_bound=0.2,
                most_anomalous_feature="dt_prev_ms",
                is_anomalous=False,
            ),
            1: DeepLogParameterFinding(
                event_index=1,
                template="B",
                feature_names=["dt_prev_ms"],
                observed_vector=[1.0],
                predicted_vector=[0.5],
                residual_mse=0.2,
                gaussian_mean=0.1,
                gaussian_stddev=0.01,
                gaussian_lower_bound=0.0,
                gaussian_upper_bound=0.2,
                most_anomalous_feature="dt_prev_ms",
                is_anomalous=False,
            ),
        },
        parameter_models=parameter_models,
    )
    state.record_sequence(
        sequence=_sequence(
            templates=["A"],
            split_label=SplitLabel.TEST,
        ),
        parameter_findings={
            0: DeepLogParameterFinding(
                event_index=0,
                template="A",
                feature_names=["dt_prev_ms"],
                observed_vector=[1.0],
                predicted_vector=[0.5],
                residual_mse=0.3,
                gaussian_mean=0.1,
                gaussian_stddev=0.01,
                gaussian_lower_bound=0.0,
                gaussian_upper_bound=0.2,
                most_anomalous_feature="dt_prev_ms",
                is_anomalous=False,
            ),
        },
        parameter_models=parameter_models,
    )

    report = state.snapshot_summary(
        train_sequence_count=1,
        test_sequence_count=2,
        highlighted_templates=("B", "A"),
    )
    assert report is not None
    assert report.series_count == 2
    assert report.highlighted_series_count == 2
    assert report.series_scope == "highlighted_subset"
    assert report.highlighted_templates == ["B", "A"]
    assert [series.template for series in report.series] == ["B", "A"]
    assert report.total_point_count == 3

    trace = state.snapshot_trace(
        train_sequence_count=1,
        test_sequence_count=2,
        highlighted_templates=("B", "A"),
    )
    assert trace is not None
    assert trace.highlighted_templates == ["B", "A"]
    assert [series.template for series in trace.series] == ["B", "A"]


def test_parameter_ci_summary_prefers_event_labels_when_available() -> None:
    """Point labels should follow injected event labels when they exist."""
    state = ParameterCiState()
    parameter_models = {
        "A": ParameterModelState(
            template="A",
            schema=ParameterFeatureSchema(
                feature_names=["dt_prev_ms"],
                numeric_parameter_positions=[],
                include_elapsed_time=True,
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
            train_pair_count=4,
            validation_pair_count=2,
        ),
    }
    state.record_sequence(
        sequence=_sequence(
            templates=["A", "A"],
            split_label=SplitLabel.TEST,
            label=1,
            event_labels=(0, 1),
        ),
        parameter_findings={
            0: DeepLogParameterFinding(
                event_index=0,
                template="A",
                feature_names=["dt_prev_ms"],
                observed_vector=[1.0],
                predicted_vector=[0.5],
                residual_mse=0.1,
                gaussian_mean=0.1,
                gaussian_stddev=0.01,
                gaussian_lower_bound=0.0,
                gaussian_upper_bound=0.2,
                most_anomalous_feature="dt_prev_ms",
                is_anomalous=False,
            ),
            1: DeepLogParameterFinding(
                event_index=1,
                template="A",
                feature_names=["dt_prev_ms"],
                observed_vector=[1.0],
                predicted_vector=[0.5],
                residual_mse=0.3,
                gaussian_mean=0.1,
                gaussian_stddev=0.01,
                gaussian_lower_bound=0.0,
                gaussian_upper_bound=0.2,
                most_anomalous_feature="dt_prev_ms",
                is_anomalous=False,
            ),
        },
        parameter_models=parameter_models,
    )

    report = state.snapshot_summary(
        train_sequence_count=1,
        test_sequence_count=1,
    )
    assert report is not None
    assert report.series[0].anomalous_point_count == 1
    assert report.total_anomalous_point_count == 1
    assert report.series_scope == "highlighted_subset"

    trace = state.snapshot_trace(
        train_sequence_count=1,
        test_sequence_count=1,
    )
    assert trace is not None
    assert [point.label for point in trace.series[0].points] == [0, 1]
    assert trace.series[0].points[1].feature_squared_errors == [0.25]


def test_parameter_only_scoring_carries_entity_history() -> None:
    """Continuous entity windows should carry parameter history between VM ids."""
    detector = DeepLogDetector(
        config=_deep_log_config(
            name="deeplog",
            history_size=2,
            top_g=1,
            hidden_size=4,
            num_layers=1,
            epochs=1,
            batch_size=2,
            parameter_detection_enabled=True,
            key_detection_enabled=False,
        ),
    )
    with Progress(disable=True) as progress:
        detector.fit(
            [
                _sequence(
                    templates=["T", "T", "T", "T"],
                    params_by_event=[
                        ["1.0"],
                        ["2.0"],
                        ["3.0"],
                        ["4.0"],
                    ],
                    dts_by_event=[None, 10, 11, 12],
                    split_label=SplitLabel.TRAIN,
                    continuous_context=True,
                ),
            ],
            progress=progress,
        )

    detector.predict(
        _sequence(
            templates=["T", "T"],
            params_by_event=[["5.0"], ["6.0"]],
            dts_by_event=[None, 13],
            label=1,
            split_label=SplitLabel.TEST,
            continuous_context=True,
        ),
    )
    detector.predict(
        _sequence(
            templates=["T", "T"],
            params_by_event=[["7.0"], ["8.0"]],
            dts_by_event=[None, 14],
            label=1,
            split_label=SplitLabel.TEST,
            continuous_context=True,
        ),
    )

    metrics = detector.run_metrics(
        run_metrics={
            "sequence_count": 2,
            "train_sequence_count": 0,
            "test_sequence_count": 2,
            "ignored_sequence_count": 0,
        },
    )

    assert metrics.parameter_ci_report is not None
    report = metrics.parameter_ci_report
    assert report.total_point_count == 4
    assert report.total_anomalous_point_count == 4
    assert report.highlighted_templates == ["T"]
    assert report.series[0].point_count == 4
    assert report.series[0].validation_sample_warning is not None
    assert report.series[0].threshold_summaries[0].point_count == 4
    assert metrics.parameter_ci_trace is not None
    trace = metrics.parameter_ci_trace
    assert trace is not None
    assert trace.total_point_count == 4
    assert trace.total_anomalous_point_count == 4
    assert trace.series[0].validation_sample_warning is not None


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
            parameter_detection_enabled=True,
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
    assert '"actual_rank":2' in encoded


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
            parameter_detection_enabled=True,
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
    assert manifest.short_session_padding_fidelity is False
    assert manifest.history_size == detector.config.history_size
    assert manifest.top_g == max(detector.config.top_g_values)
    assert manifest.top_g_values == list(detector.config.top_g_values)
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


def test_hdfs_paper_configs_pin_runtime_top_g_to_nine() -> None:
    """HDFS paper-facing DeepLog bundles should score with `g = 9`."""
    config_paths = [
        "experiments/configs/datasets/hdfs/v1_deeplog_paper_entry100k_assign_first.toml",
        "experiments/configs/datasets/hdfs/v1_deeplog_paper_entry100k_split_partial.toml",
        "experiments/configs/datasets/hdfs/wuyifan18_deeplog_preprocessed.toml",
    ]

    for config_path in config_paths:
        bundles = load_experiment_bundles(Path(config_path))
        deeplog_bundle = next(
            bundle for bundle in bundles if bundle.model.detector == "deeplog"
        )
        assert isinstance(deeplog_bundle.model, DeepLogModelConfig)
        assert deeplog_bundle.model.top_g_values == (1, 3, 5, 7, 9)
        assert max(deeplog_bundle.model.top_g_values) == 9
