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
class _KeyTrainingRun:
    """Materialised in-memory tensors and settings for key-model training.

    Attributes:
        inputs (torch.Tensor): One-hot encoded training histories.
        targets (torch.Tensor): Target key indexes.
        criterion (nn.CrossEntropyLoss): Training loss function.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        device (torch.device): Torch device used for the run.
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    criterion: nn.CrossEntropyLoss
    epochs: int
    batch_size: int
    device: torch.device


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

    Raises:
        ValueError: If no history-target examples can be constructed.
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
    examples: list[tuple[list[int], int]] = []
    try:
        for sequence in training_corpus.sequences:
            examples.extend(
                iter_key_examples(
                    sequences=(sequence,),
                    template_to_index=template_to_index,
                    history_size=config.history_size,
                ),
            )
            if progress is not None and prepare_task is not None:
                progress.advance(prepare_task)
    finally:
        if progress is not None and prepare_task is not None:
            progress.remove_task(prepare_task)
    if not examples:
        msg = "DeepLog key model requires at least one history/next-key example."
        raise ValueError(msg)

    history_inputs = _one_hot_histories(
        histories=[history for history, _ in examples],
        vocab_size=len(template_to_index),
    )
    targets = torch.tensor(
        [target for _, target in examples],
        dtype=torch.long,
    )

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
            inputs=history_inputs,
            targets=targets,
            criterion=nn.CrossEntropyLoss(),
            epochs=config.epochs,
            batch_size=config.batch_size,
            device=device,
        ),
        progress=progress,
    )
    return model.eval(), template_to_index, index_to_template


def iter_key_examples(
    *,
    sequences: Iterable[TemplateSequence],
    template_to_index: dict[str, int],
    history_size: int,
) -> Iterator[tuple[list[int], int]]:
    """Yield DeepLog `(history -> next-key)` training pairs.

    Args:
        sequences (Iterable[TemplateSequence]): Normal train sequences.
        template_to_index (dict[str, int]): Key vocabulary.
        history_size (int): Number of prior keys per example.

    Yields:
        tuple[list[int], int]: Encoded history and target key index.
    """
    for sequence in sequences:
        template_indexes = [
            template_to_index[template] for template in sequence.templates
        ]
        if len(template_indexes) <= history_size:
            continue
        for start in range(len(template_indexes) - history_size):
            yield (
                template_indexes[start : start + history_size],
                template_indexes[start + history_size],
            )


def _train_key_model(
    *,
    model: KeyLSTM,
    optimizer: torch.optim.Optimizer,
    training_run: _KeyTrainingRun,
    progress: Progress | None,
) -> None:
    """Train the DeepLog key model over in-memory one-hot windows.

    Args:
        model (KeyLSTM): Fitted key model being trained.
        optimizer (torch.optim.Optimizer): Optimiser used for training.
        training_run (_KeyTrainingRun): In-memory inputs, targets, and
            training settings.
        progress (Progress | None): Optional progress reporter.
    """
    dataset_size = len(training_run.inputs)
    effective_batch_size = min(training_run.batch_size, dataset_size)
    all_inputs = training_run.inputs.to(training_run.device)
    all_targets = training_run.targets.to(training_run.device)
    epoch_task = None
    if progress is not None:
        epoch_task = progress.add_task(
            "Training DeepLog key model",
            total=training_run.epochs,
        )

    for _ in range(training_run.epochs):
        model.train()
        permutation = torch.randperm(dataset_size, device=training_run.device)
        for start in range(0, dataset_size, effective_batch_size):
            batch_indexes = permutation[start : start + effective_batch_size]
            optimizer.zero_grad()
            logits = model(all_inputs[batch_indexes])
            loss = training_run.criterion(logits, all_targets[batch_indexes])
            loss.backward()
            optimizer.step()
        if progress is not None and epoch_task is not None:
            progress.advance(epoch_task)

    if progress is not None and epoch_task is not None:
        progress.update(epoch_task, completed=training_run.epochs, visible=False)


def score_key_sequence(
    *,
    sequence: TemplateSequence,
    context: KeyScoringContext,
) -> dict[int, DeepLogKeyFinding]:
    """Score one sequence with the DeepLog key model.

    Args:
        sequence (TemplateSequence): Sequence to score.
        context (KeyScoringContext): Fitted key-model state and settings.

    Returns:
        dict[int, DeepLogKeyFinding]: Event index to key-model finding.
    """
    findings: dict[int, DeepLogKeyFinding] = {}
    templates = sequence.templates
    if len(templates) <= context.history_size:
        return findings

    known_history_indexes: list[list[int]] = []
    known_target_indexes: list[int] = []
    for target_index in range(context.history_size, len(templates)):
        history_templates = templates[
            target_index - context.history_size : target_index
        ]
        unknown_history_templates = [
            template
            for template in history_templates
            if template not in context.template_to_index
        ]
        if unknown_history_templates:
            # We fail closed here. Passing an unseen history through a synthetic
            # token would ask the model to make a confident prediction for a
            # situation it was never trained on.
            findings[target_index] = DeepLogKeyFinding(
                event_index=target_index,
                history_templates=history_templates,
                unknown_history_templates=unknown_history_templates,
                actual_template=templates[target_index],
                actual_probability=None,
                is_anomalous=True,
                is_oov=templates[target_index] not in context.template_to_index,
                top_predictions=[],
            )
            continue
        known_target_indexes.append(target_index)
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

    for target_index, probabilities in zip(
        known_target_indexes,
        probabilities_by_event,
        strict=True,
    ):
        findings[target_index] = _score_key_event(
            templates=templates,
            target_index=target_index,
            probabilities=probabilities,
            context=context,
        )
    return findings


def _score_key_event(
    *,
    templates: list[str],
    target_index: int,
    probabilities: torch.Tensor,
    context: KeyScoringContext,
) -> DeepLogKeyFinding:
    """Build one key-model finding from predicted next-key probabilities.

    Args:
        templates (list[str]): Full template sequence being scored.
        target_index (int): Absolute index of the target event in `templates`.
        probabilities (torch.Tensor): Model probabilities for the next-key
            vocabulary at this target position.
        context (KeyScoringContext): Fitted key-model state and settings.

    Returns:
        DeepLogKeyFinding: Serialised decision payload for one target event.
    """
    history_templates = templates[target_index - context.history_size : target_index]
    unknown_history_templates = [
        template
        for template in history_templates
        if template not in context.template_to_index
    ]
    actual_template = templates[target_index]
    actual_index = context.template_to_index.get(actual_template)
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
    is_oov = actual_index is None
    actual_probability = (
        None if actual_index is None else float(probabilities[actual_index])
    )
    top_index_set = {int(index) for index in top_indexes.tolist()}
    return DeepLogKeyFinding(
        event_index=target_index,
        history_templates=history_templates,
        unknown_history_templates=unknown_history_templates,
        actual_template=actual_template,
        actual_probability=actual_probability,
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

    batch_size = len(histories)
    history_size = len(histories[0])
    history_index_tensor = torch.tensor(histories, dtype=torch.long, device=device)
    history_tensor = torch.zeros(
        (batch_size, history_size, vocab_size),
        dtype=torch.float32,
        device=device,
    )
    history_tensor.scatter_(2, history_index_tensor.unsqueeze(-1), 1.0)
    return history_tensor
