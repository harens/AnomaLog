"""DeepLog key-model training and scoring helpers.

This file implements the paper's log-key anomaly path:

1. build `(history window -> next key)` training pairs from normal sequences
2. train a stacked LSTM with cross-entropy loss
3. score each inference-time event by checking whether the observed key lands
   in the model's top-`g` predictions

The functions here intentionally stay close to those three paper-level steps so
that a reader can trace the algorithm without first unpacking a generic
training abstraction.
"""

from __future__ import annotations

import random
from collections.abc import Sized
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from experiments.models.deeplog.shared import (
    DeepLogKeyFinding,
    DeepLogTopPrediction,
    KeyLSTM,
    NormalTrainingCorpus,
)
from experiments.models.sequence_masks import training_event_index_mask

_KEY_TRAINING_MICROBATCH_SIZE = 256

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from rich.progress import Progress, TaskID

    from anomalog.sequences import TemplateSequence
    from experiments.models.deeplog.detector import DeepLogModelConfig


@dataclass(frozen=True, slots=True)
class KeyScoringContext:
    """All fitted state needed to score a sequence with the key model.

    Attributes:
        model (KeyLSTM): Fitted next-key model.
        template_to_index (dict[str, int]): Template-to-index vocabulary map.
        index_to_template (dict[int, str]): Reverse vocabulary map.
        history_size (int): Number of prior keys required for each example.
        top_g (int): Number of top predictions treated as normal.
    """

    model: KeyLSTM
    template_to_index: dict[str, int]
    index_to_template: dict[int, str]
    history_size: int
    top_g: int


@dataclass(frozen=True, slots=True)
class _KeyTrainingSequenceExamples:
    """Cached training examples for one replayable sequence.

    Attributes:
        template_indexes (torch.Tensor): Encoded templates for the full
            sequence.
        eligible_target_indexes (torch.Tensor): Zero-based indexes of the
            eligible next-key targets for the sequence.
        history_windows (torch.Tensor): Cached history windows aligned with the
            eligible targets.
        target_indexes (torch.Tensor): Cached next-key targets aligned with the
            history windows.
    """

    template_indexes: torch.Tensor
    eligible_target_indexes: torch.Tensor
    history_windows: torch.Tensor
    target_indexes: torch.Tensor


@dataclass(frozen=True, slots=True)
class _KeyTrainingExampleMaterialisationConfig:
    """Settings needed to cache replayable DeepLog key examples.

    Attributes:
        template_to_index (dict[str, int]): Template vocabulary mapping.
        history_size (int): Number of prior keys required for each example.
    """

    template_to_index: dict[str, int]
    history_size: int


@dataclass(frozen=True, slots=True)
class _KeyTrainingRun:
    """Replayable state and settings for key-model training.

    Attributes:
        criterion (nn.CrossEntropyLoss): Training loss function.
        sequence_examples (tuple[_KeyTrainingSequenceExamples, ...]): Cached
            replayable examples for each training sequence.
        history_size (int): Number of prior keys per example.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        device (torch.device): Torch device used for the run.
        vocab_size (int): Number of known key indexes.
    """

    criterion: nn.CrossEntropyLoss
    sequence_examples: tuple[_KeyTrainingSequenceExamples, ...]
    history_size: int
    epochs: int
    batch_size: int
    device: torch.device
    vocab_size: int


@dataclass(frozen=True, slots=True)
class _KeyEventScoreInput:
    """Materialised inputs for one key-model event score.

    Attributes:
        templates (list[str]): Full template sequence being scored.
        target_index (int): Target index within the scored sequence.
        probabilities (torch.Tensor): Model probabilities for the target
            event.
        actual_rank (int | None): Exact rank of the observed key, if known.
        prefix_length (int): Number of carried-over prefix templates.
    """

    templates: list[str]
    target_index: int
    probabilities: torch.Tensor
    actual_rank: int | None
    prefix_length: int


def fit_key_model(
    *,
    training_corpus: NormalTrainingCorpus,
    config: DeepLogModelConfig,
    device: torch.device,
    progress: Progress | None = None,
) -> tuple[KeyLSTM, dict[str, int], dict[int, str]]:
    """Train DeepLog's stacked-LSTM next-key model.

    Args:
        training_corpus (NormalTrainingCorpus): Replayable normal training state.
        config (DeepLogModelConfig): DeepLog configuration.
        device (torch.device): Torch device used for training/inference.
        progress (Progress | None): Optional progress reporter.

    Returns:
        tuple[KeyLSTM, dict[str, int], dict[int, str]]: Fitted key model and
            its template-index mappings.
    """
    # The paper's key model is trained only on log keys observed in normal
    # training data. We therefore make the vocabulary exactly that set.
    template_to_index = {
        template: idx for idx, template in enumerate(training_corpus.templates)
    }
    index_to_template = {idx: template for template, idx in template_to_index.items()}

    prepare_task: TaskID | None = None
    if progress is not None:
        total = (
            len(training_corpus.sequences)
            if isinstance(training_corpus.sequences, Sized)
            else None
        )
        prepare_task = progress.add_task(
            "Preparing DeepLog key examples",
            total=total,
        )
    try:
        sequence_examples = _materialise_key_training_examples(
            sequences=training_corpus.sequences,
            materialisation_config=_KeyTrainingExampleMaterialisationConfig(
                template_to_index=template_to_index,
                history_size=config.history_size,
            ),
            progress=progress,
            prepare_task=prepare_task,
        )
    finally:
        if progress is not None and prepare_task is not None:
            progress.remove_task(prepare_task)

    model = KeyLSTM(
        vocab_size=len(template_to_index),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    _train_key_model(
        model=model,
        optimizer=optimizer,
        training_run=_KeyTrainingRun(
            criterion=nn.CrossEntropyLoss(),
            sequence_examples=sequence_examples,
            history_size=config.history_size,
            epochs=config.epochs,
            batch_size=config.batch_size,
            device=device,
            vocab_size=len(template_to_index),
        ),
        progress=progress,
    )
    return model.eval(), template_to_index, index_to_template


def _materialise_key_training_examples(
    *,
    sequences: Iterable[TemplateSequence],
    materialisation_config: _KeyTrainingExampleMaterialisationConfig,
    progress: Progress | None,
    prepare_task: TaskID | None,
) -> tuple[_KeyTrainingSequenceExamples, ...]:
    """Cache the replayable key-model examples for each sequence.

    Args:
        sequences (Iterable[TemplateSequence]): Normal training sequences.
        materialisation_config (_KeyTrainingExampleMaterialisationConfig):
            Template vocabulary mapping and history window length.
        progress (Progress | None): Optional progress reporter.
        prepare_task (TaskID | None): Progress task used while scanning.

    Returns:
        tuple[_KeyTrainingSequenceExamples, ...]: Cached sequence examples.

    Raises:
        ValueError: If the corpus contains no history/next-key examples.
    """
    template_to_index = materialisation_config.template_to_index
    history_size = materialisation_config.history_size
    sequence_examples: list[_KeyTrainingSequenceExamples] = []
    has_examples = False
    for sequence in sequences:
        template_indexes = torch.tensor(
            [template_to_index[template] for template in sequence.templates],
            dtype=torch.long,
        )
        eligible_target_indexes = torch.tensor(
            [
                target_index
                for target_index in training_event_index_mask(sequence)
                if target_index >= history_size
            ],
            dtype=torch.long,
        )
        if template_indexes.numel() > history_size and eligible_target_indexes.numel():
            has_examples = True
            history_windows = template_indexes.unfold(
                0,
                history_size,
                1,
            ).index_select(
                0,
                eligible_target_indexes - history_size,
            )
            target_indexes = template_indexes.index_select(
                0,
                eligible_target_indexes,
            )
        else:
            history_windows = torch.empty(
                (0, history_size),
                dtype=torch.long,
            )
            target_indexes = torch.empty((0,), dtype=torch.long)
        sequence_examples.append(
            _KeyTrainingSequenceExamples(
                template_indexes=template_indexes,
                eligible_target_indexes=eligible_target_indexes,
                history_windows=history_windows,
                target_indexes=target_indexes,
            ),
        )
        if progress is not None and prepare_task is not None:
            progress.advance(prepare_task)
    if not has_examples:
        msg = "DeepLog key model requires at least one history/next-key example."
        raise ValueError(msg)
    return tuple(sequence_examples)


def iter_key_examples(
    *,
    sequences: Iterable[TemplateSequence],
    template_to_index: dict[str, int],
    history_size: int,
    eligible_target_indexes: Iterable[int] | None = None,
) -> Iterator[tuple[list[int], int]]:
    """Yield DeepLog `(history -> next-key)` training pairs.

    Args:
        sequences (Iterable[TemplateSequence]): Normal train sequences.
        template_to_index (dict[str, int]): Key vocabulary.
        history_size (int): Number of prior keys per example.
        eligible_target_indexes (Iterable[int] | None): Optional sequence-local
            target indexes that may contribute training examples.

    Yields:
        tuple[list[int], int]: Encoded history and target key index.
    """
    eligible_indexes = (
        set(eligible_target_indexes) if eligible_target_indexes is not None else None
    )
    for sequence in sequences:
        template_indexes = [
            template_to_index[template] for template in sequence.templates
        ]
        if len(template_indexes) <= history_size:
            continue
        for start in range(len(template_indexes) - history_size):
            target_index = start + history_size
            if eligible_indexes is not None and target_index not in eligible_indexes:
                continue
            yield (
                template_indexes[start : start + history_size],
                template_indexes[target_index],
            )


def _train_key_model(
    *,
    model: KeyLSTM,
    optimizer: torch.optim.Optimizer,
    training_run: _KeyTrainingRun,
    progress: Progress | None,
) -> None:
    """Train the DeepLog key model over in-memory indexed history windows.

    Args:
        model (KeyLSTM): Fitted key model being trained.
        optimizer (torch.optim.Optimizer): Optimiser used for training.
        training_run (_KeyTrainingRun): Replayable sequences and training
            settings.
        progress (Progress | None): Optional progress reporter.
    """
    effective_batch_size = max(1, training_run.batch_size)
    epoch_task = None
    if progress is not None:
        epoch_task = progress.add_task(
            "Training DeepLog key model",
            total=training_run.epochs,
        )

    for _ in range(training_run.epochs):
        model.train()
        for batch_histories, batch_targets in _iter_key_training_batches(
            training_run=training_run,
            batch_size=effective_batch_size,
        ):
            _optimise_key_training_batch(
                model=model,
                optimizer=optimizer,
                training_run=training_run,
                batch_histories=batch_histories,
                batch_targets=batch_targets,
            )
        if progress is not None and epoch_task is not None:
            progress.advance(epoch_task)

    if progress is not None and epoch_task is not None:
        progress.update(epoch_task, completed=training_run.epochs, visible=False)


def _iter_key_training_batches(
    *,
    training_run: _KeyTrainingRun,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield minibatches of DeepLog key examples from replayable sequences.

    Args:
        training_run (_KeyTrainingRun): Replayable sequences and settings.
        batch_size (int): Maximum number of examples per yielded batch.

    Yields:
        tuple[torch.Tensor, torch.Tensor]: Key-history windows and matching
            target indexes for one minibatch.
    """
    training_sequences = list(training_run.sequence_examples)
    random.shuffle(training_sequences)
    batch_histories: list[torch.Tensor] = []
    batch_targets: list[torch.Tensor] = []
    batch_count = 0
    for sequence_examples in training_sequences:
        if sequence_examples.target_indexes.numel() == 0:
            continue
        history_windows = sequence_examples.history_windows
        target_indexes = sequence_examples.target_indexes
        example_count = int(target_indexes.shape[0])
        start_index = 0
        while start_index < example_count:
            remaining = batch_size - batch_count
            take = min(remaining, example_count - start_index)
            batch_histories.append(history_windows[start_index : start_index + take])
            batch_targets.append(target_indexes[start_index : start_index + take])
            batch_count += take
            start_index += take
            if batch_count < batch_size:
                continue
            yield torch.cat(batch_histories, dim=0), torch.cat(batch_targets, dim=0)
            batch_histories = []
            batch_targets = []
            batch_count = 0
    if batch_histories:
        yield torch.cat(batch_histories, dim=0), torch.cat(batch_targets, dim=0)


def _optimise_key_training_batch(
    *,
    model: KeyLSTM,
    optimizer: torch.optim.Optimizer,
    training_run: _KeyTrainingRun,
    batch_histories: torch.Tensor,
    batch_targets: torch.Tensor,
) -> None:
    """Optimise the key model on one minibatch.

    Args:
        model (KeyLSTM): Key model being trained.
        optimizer (torch.optim.Optimizer): Optimiser used for the update.
        training_run (_KeyTrainingRun): Replayable sequences and settings.
        batch_histories (torch.Tensor): Indexed history windows.
        batch_targets (torch.Tensor): Matching next-key indexes.

    Raises:
        RuntimeError: If the configured device still exhausts memory at the
            smallest supported microbatch size.
    """
    batch_size = int(batch_histories.shape[0])
    batch_histories, batch_target_indexes = _move_key_training_batch_to_device(
        batch_histories=batch_histories,
        batch_targets=batch_targets,
        device=training_run.device,
    )
    microbatch_size = min(batch_size, _KEY_TRAINING_MICROBATCH_SIZE)
    while True:
        optimizer.zero_grad()
        try:
            for start in range(0, batch_size, microbatch_size):
                end = start + microbatch_size
                logits = model(
                    _one_hot_history_indexes(
                        history_indexes=batch_histories[start:end],
                        vocab_size=training_run.vocab_size,
                    ),
                )
                loss = training_run.criterion(
                    logits,
                    batch_target_indexes[start:end],
                )
                scaled_loss = loss * ((end - start) / batch_size)
                scaled_loss.backward()
        except RuntimeError as exc:
            if not _is_cuda_oom_error(exc) or microbatch_size <= 1:
                raise
            if training_run.device.type == "cuda":
                torch.cuda.empty_cache()
            microbatch_size = max(1, microbatch_size // 2)
            continue
        optimizer.step()
        return


def _move_key_training_batch_to_device(
    *,
    batch_histories: torch.Tensor,
    batch_targets: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move one cached DeepLog key batch onto the training device.

    The CPU-cached integer history windows stay out of GPU memory until a
    minibatch is about to be optimised. When the target device is CUDA,
    pinning the host tensors first keeps the transfer path asynchronous without
    changing the training semantics.

    Args:
        batch_histories (torch.Tensor): Batched indexed history windows.
        batch_targets (torch.Tensor): Matching target indexes.
        device (torch.device): Training device.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Device-ready histories and targets.
    """
    if device.type == "cuda":
        batch_histories = batch_histories.pin_memory()
        batch_targets = batch_targets.pin_memory()
        return (
            batch_histories.to(device=device, non_blocking=True),
            batch_targets.to(device=device, non_blocking=True),
        )
    return (
        batch_histories.to(device=device),
        batch_targets.to(device=device),
    )


def _is_cuda_oom_error(exc: RuntimeError) -> bool:
    """Return whether a runtime error looks like a CUDA memory exhaustion.

    Args:
        exc (RuntimeError): Runtime error raised by the training step.

    Returns:
        bool: True when the exception message matches a CUDA OOM failure.
    """
    return "out of memory" in str(exc).lower()


def score_key_sequence(  # noqa: C901
    *,
    sequence: TemplateSequence,
    context: KeyScoringContext,
    prefix_templates: list[str] | None = None,
    include_short_session_padding_fallback: bool = False,
) -> dict[int, DeepLogKeyFinding]:
    """Score one sequence with the DeepLog key model.

    Args:
        sequence (TemplateSequence): Sequence to score.
        context (KeyScoringContext): Fitted key-model state and settings.
        prefix_templates (list[str] | None): Optional templates from the
            immediately preceding chronological context.
        include_short_session_padding_fallback (bool): Whether to score one
            padded top-`g` decision for short standalone sessions.

    Returns:
        dict[int, DeepLogKeyFinding]: Event index to key-model finding.
    """
    findings: dict[int, DeepLogKeyFinding] = {}
    templates = sequence.templates
    prefix_templates = [] if prefix_templates is None else prefix_templates
    combined_templates = prefix_templates + templates
    prefix_length = len(prefix_templates)
    if len(templates) <= context.history_size:
        if len(combined_templates) <= context.history_size:
            if not include_short_session_padding_fallback or not templates:
                return findings
            short_finding = _score_short_session_with_padding(
                templates=templates,
                context=context,
            )
            if short_finding is not None:
                findings[short_finding.event_index] = short_finding
            return findings
        templates = combined_templates
        prefix_length = len(prefix_templates)
    elif prefix_templates:
        templates = combined_templates
        prefix_length = len(prefix_templates)

    known_history_indexes: list[list[int]] = []
    known_target_indexes: list[int] = []
    for target_index in range(context.history_size, len(templates)):
        history_templates = templates[
            target_index - context.history_size : target_index
        ]
        if target_index < prefix_length:
            continue
        local_target_index = target_index - prefix_length
        unknown_history_templates = [
            template
            for template in history_templates
            if template not in context.template_to_index
        ]
        if unknown_history_templates:
            # We fail closed here. Passing an unseen history through a synthetic
            # token would ask the model to make a confident prediction for a
            # situation it was never trained on.
            findings[local_target_index] = DeepLogKeyFinding(
                event_index=local_target_index,
                history_templates=history_templates,
                unknown_history_templates=unknown_history_templates,
                actual_template=templates[target_index],
                actual_probability=None,
                actual_rank=None,
                is_anomalous=True,
                is_oov=templates[target_index] not in context.template_to_index,
                top_predictions=[],
            )
            continue
        known_target_indexes.append(local_target_index)
        known_history_indexes.append(
            [context.template_to_index[template] for template in history_templates],
        )

    if not known_history_indexes:
        return findings

    history_tensor = _one_hot_histories(
        histories=known_history_indexes,
        vocab_size=len(context.template_to_index),
        device=next(context.model.parameters()).device,
    )
    with torch.inference_mode():
        probabilities_by_event = torch.softmax(
            context.model(history_tensor),
            dim=1,
        ).cpu()
    rank_positions_by_event = torch.argsort(
        torch.argsort(probabilities_by_event, dim=1, descending=True),
        dim=1,
    )

    for event_position, (target_index, probabilities) in enumerate(
        zip(known_target_indexes, probabilities_by_event, strict=True),
    ):
        findings[target_index] = _score_key_event(
            score_input=_KeyEventScoreInput(
                templates=templates,
                target_index=target_index,
                probabilities=probabilities,
                actual_rank=(
                    None
                    if templates[target_index] not in context.template_to_index
                    else int(
                        rank_positions_by_event[
                            event_position,
                            context.template_to_index[templates[target_index]],
                        ],
                    )
                    + 1
                ),
                prefix_length=prefix_length,
            ),
            context=context,
        )
    return findings


def _score_short_session_with_padding(  # noqa: PLR0914
    *,
    templates: list[str],
    context: KeyScoringContext,
) -> DeepLogKeyFinding | None:
    """Score one short standalone sequence with left-padded key history.

    This mirrors the original DeepLog reference script behaviour where sessions
    shorter than `window_size + 1` are padded during prediction so they still
    contribute one top-`g` anomaly decision.

    Args:
        templates (list[str]): Sequence templates for one standalone session.
        context (KeyScoringContext): Fitted key-model state and settings.

    Returns:
        DeepLogKeyFinding | None: Decision for the last event, or `None` when
            no target can be formed.
    """
    if not templates:
        return None
    target_index = len(templates) - 1
    history_templates = templates[:-1]
    padded_history_indexes = [-1] * context.history_size
    history_tail = history_templates[-context.history_size :]
    start = context.history_size - len(history_tail)
    for offset, template in enumerate(history_tail):
        index = context.template_to_index.get(template)
        if index is None:
            continue
        padded_history_indexes[start + offset] = index

    history_tensor = torch.zeros(
        (1, context.history_size, len(context.template_to_index)),
        dtype=torch.float32,
        device=next(context.model.parameters()).device,
    )
    for pos, idx in enumerate(padded_history_indexes):
        if idx >= 0:
            history_tensor[0, pos, idx] = 1.0

    with torch.inference_mode():
        probabilities = torch.softmax(context.model(history_tensor), dim=1).cpu()[0]
    rank_positions = torch.argsort(
        torch.argsort(probabilities.unsqueeze(0), dim=1, descending=True),
        dim=1,
    )

    top_probabilities, top_indexes = _top_key_predictions(
        probabilities=probabilities,
        vocabulary_size=len(context.template_to_index),
        top_g=context.top_g,
    )
    top_predictions = [
        DeepLogTopPrediction(
            template=context.index_to_template[int(index)],
            probability=float(probability),
        )
        for probability, index in zip(
            top_probabilities.tolist(),
            top_indexes.tolist(),
            strict=True,
        )
    ]
    actual_template = templates[target_index]
    actual_index = context.template_to_index.get(actual_template)
    is_oov = actual_index is None
    actual_rank = (
        None if actual_index is None else int(rank_positions[0, actual_index]) + 1
    )
    actual_probability = (
        None if actual_index is None else float(probabilities[actual_index])
    )
    top_index_set = {int(index) for index in top_indexes.tolist()}
    return DeepLogKeyFinding(
        event_index=target_index,
        history_templates=history_templates,
        unknown_history_templates=[
            template
            for template in history_tail
            if template not in context.template_to_index
        ],
        actual_template=actual_template,
        actual_probability=actual_probability,
        actual_rank=actual_rank,
        is_anomalous=is_oov or (actual_index not in top_index_set),
        is_oov=is_oov,
        top_predictions=top_predictions,
    )


def _score_key_event(
    *,
    score_input: _KeyEventScoreInput,
    context: KeyScoringContext,
) -> DeepLogKeyFinding:
    """Build one key-model finding from predicted next-key probabilities.

    Args:
        score_input (_KeyEventScoreInput): Materialised event-scoring inputs.
        context (KeyScoringContext): Fitted key-model state and settings.

    Returns:
        DeepLogKeyFinding: Serialised decision payload for one target event.
    """
    absolute_target_index = score_input.target_index + score_input.prefix_length
    history_templates = score_input.templates[
        absolute_target_index - context.history_size : absolute_target_index
    ]
    unknown_history_templates = [
        template
        for template in history_templates
        if template not in context.template_to_index
    ]
    actual_template = score_input.templates[absolute_target_index]
    actual_index = context.template_to_index.get(actual_template)
    top_probabilities, top_indexes = _top_key_predictions(
        probabilities=score_input.probabilities,
        vocabulary_size=len(context.template_to_index),
        top_g=context.top_g,
    )
    top_predictions = [
        DeepLogTopPrediction(
            template=context.index_to_template[int(index)],
            probability=float(probability),
        )
        for probability, index in zip(
            top_probabilities.tolist(),
            top_indexes.tolist(),
            strict=True,
        )
    ]
    is_oov = actual_index is None
    actual_probability = (
        None if actual_index is None else float(score_input.probabilities[actual_index])
    )
    top_index_set = {int(index) for index in top_indexes.tolist()}
    return DeepLogKeyFinding(
        event_index=score_input.target_index,
        history_templates=history_templates,
        unknown_history_templates=unknown_history_templates,
        actual_template=actual_template,
        actual_probability=actual_probability,
        actual_rank=score_input.actual_rank,
        is_anomalous=is_oov or (actual_index not in top_index_set),
        is_oov=is_oov,
        top_predictions=top_predictions,
    )


def _top_key_predictions(
    *,
    probabilities: torch.Tensor,
    vocabulary_size: int,
    top_g: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the configured top-`g` candidate keys for one event.

    Args:
        probabilities (torch.Tensor): Per-key probabilities for one event.
        vocabulary_size (int): Number of known key indexes.
        top_g (int): Number of top predictions to return.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Top probabilities and their indexes.
    """
    top_k = min(top_g, vocabulary_size)
    return torch.topk(probabilities, k=top_k)


def _one_hot_histories(
    *,
    histories: list[list[int]],
    vocab_size: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Encode a batch of key-history windows as one-hot tensors.

    Args:
        histories (list[list[int]]): Encoded history windows.
        vocab_size (int): Number of known key indexes.
        device (torch.device | None): Optional tensor device.

    Returns:
        torch.Tensor: One-hot encoded batch with shape
            ``(batch, history_size, vocab_size)``.

    Raises:
        ValueError: If no histories are provided.
    """
    if not histories:
        msg = "at least one history is required"
        raise ValueError(msg)

    history_index_tensor = torch.tensor(histories, dtype=torch.long, device=device)
    return _one_hot_history_indexes(
        history_indexes=history_index_tensor,
        vocab_size=vocab_size,
    )


def _one_hot_history_indexes(
    *,
    history_indexes: torch.Tensor,
    vocab_size: int,
) -> torch.Tensor:
    """Encode a batch of indexed key histories as one-hot tensors.

    Args:
        history_indexes (torch.Tensor): Batched history indexes with shape
            ``(batch, history_size)``.
        vocab_size (int): Number of known key indexes.

    Returns:
        torch.Tensor: One-hot encoded batch with shape
            ``(batch, history_size, vocab_size)``.
    """
    batch_size, history_size = history_indexes.shape
    history_tensor = torch.zeros(
        (batch_size, history_size, vocab_size),
        dtype=torch.float32,
        device=history_indexes.device,
    )
    history_tensor.scatter_(2, history_indexes.unsqueeze(-1), 1.0)
    return history_tensor
