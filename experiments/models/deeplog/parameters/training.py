"""Training and calibration for DeepLog parameter models.

This module owns the offline fitting loop:

- iterate over template schemas
- build the per-template datasets
- train one LSTM per template
- calibrate one Gaussian threshold per template

Keeping that flow in one place makes it easier to answer "how does a fitted
`ParameterModelState` come into existence?" without jumping between unrelated
helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from experiments.models.deeplog.parameters.dataset import (
    ParameterTrainingPair,
    build_parameter_datasets,
    fit_gaussian_threshold,
    masked_mse,
    masked_regression_loss,
)
from experiments.models.deeplog.parameters.schema import build_parameter_schemas
from experiments.models.deeplog.shared import (
    NormalTrainingCorpus,
    ParameterFeatureSchema,
    ParameterLSTM,
    ParameterModelState,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rich.progress import Progress

    from anomalog.sequences import TemplateSequence
    from experiments.models.deeplog.detector import DeepLogModelConfig


@dataclass(frozen=True, slots=True)
class _ParameterTrainingRun:
    """Materialized in-memory tensors and settings for one template model.

    Attributes:
        inputs (torch.Tensor): Normalised training histories.
        targets (torch.Tensor): Normalised target vectors.
        masks (torch.Tensor): Observation masks for the target vectors.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        device (torch.device): Torch device used for the run.
    """

    inputs: torch.Tensor
    targets: torch.Tensor
    masks: torch.Tensor
    epochs: int
    batch_size: int
    device: torch.device


def fit_parameter_models(
    *,
    training_corpus: NormalTrainingCorpus,
    config: DeepLogModelConfig,
    device: torch.device,
    progress: Progress | None = None,
) -> tuple[dict[str, ParameterModelState], dict[str, str]]:
    """Train one DeepLog parameter-value model per template when possible.

    Args:
        training_corpus (NormalTrainingCorpus): Normal training data and templates.
        config (DeepLogModelConfig): DeepLog model configuration.
        device (torch.device): Torch device for model training.
        progress (Progress | None): Optional progress reporter.

    Returns:
        tuple[dict[str, ParameterModelState], dict[str, str]]: Fitted models
            and per-template skip reasons.
    """
    schemas = build_parameter_schemas(
        normal_sequences=training_corpus.sequences,
        all_templates=training_corpus.templates,
        include_elapsed_time=config.include_elapsed_time,
    )
    parameter_models: dict[str, ParameterModelState] = {}
    skipped_models: dict[str, str] = {}

    template_task = None
    if progress is not None:
        template_task = progress.add_task(
            "Training DeepLog parameter models",
            total=len(schemas),
        )
    for template, schema in schemas.items():
        state, skip_reason = fit_parameter_model(
            template=template,
            schema=schema,
            normal_sequences=training_corpus.sequences,
            config=config,
            device=device,
        )
        if skip_reason is not None:
            skipped_models[template] = skip_reason
        elif state is not None:
            parameter_models[template] = state
        if progress is not None and template_task is not None:
            progress.advance(template_task)

    if progress is not None and template_task is not None:
        progress.update(template_task, completed=len(schemas), visible=False)
    return parameter_models, skipped_models


def fit_parameter_model(
    *,
    template: str,
    schema: ParameterFeatureSchema,
    normal_sequences: Iterable[TemplateSequence],
    config: DeepLogModelConfig,
    device: torch.device,
) -> tuple[ParameterModelState | None, str | None]:
    """Fit one per-template parameter model or return a skip reason.

    Args:
        template (str): Template being modeled.
        schema (ParameterFeatureSchema): Feature schema for the template.
        normal_sequences (Iterable[TemplateSequence]): Normal training sequences.
        config (DeepLogModelConfig): DeepLog model configuration.
        device (torch.device): Torch device for model training.

    Returns:
        tuple[ParameterModelState | None, str | None]: Fitted model state on
            success, otherwise a human-readable skip reason.
    """
    if not schema.feature_names:
        return None, "template has no numeric modelable features"

    train_pairs, validation_pairs, normalisation = build_parameter_datasets(
        normal_sequences=normal_sequences,
        template=template,
        schema=schema,
        history_size=config.history_size,
        validation_fraction=config.validation_fraction,
    )
    if not train_pairs:
        return None, "not enough training examples after validation split"
    if not validation_pairs:
        return None, "not enough validation examples for Gaussian calibration"

    feature_count = len(schema.feature_names)
    model = ParameterLSTM(
        input_size=feature_count,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=feature_count,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    train_parameter_model(
        model=model,
        optimizer=optimizer,
        training_run=_ParameterTrainingRun(
            inputs=torch.tensor(
                [pair.history_inputs for pair in train_pairs],
                dtype=torch.float32,
                device=device,
            ),
            targets=torch.tensor(
                [pair.target for pair in train_pairs],
                dtype=torch.float32,
                device=device,
            ),
            masks=torch.tensor(
                [pair.target_mask for pair in train_pairs],
                dtype=torch.bool,
                device=device,
            ),
            epochs=config.epochs,
            batch_size=config.batch_size,
            device=device,
        ),
    )
    model.eval()

    residuals = parameter_pair_residuals(
        model=model,
        pairs=validation_pairs,
        batch_size=config.batch_size,
    )
    gaussian = fit_gaussian_threshold(
        residuals=residuals,
        confidence=config.gaussian_confidence,
    )
    return (
        ParameterModelState(
            template=template,
            schema=schema,
            normalisation=normalisation,
            gaussian=gaussian,
            model=model,
        ),
        None,
    )


def train_parameter_model(
    *,
    model: ParameterLSTM,
    optimizer: torch.optim.Optimizer,
    training_run: _ParameterTrainingRun,
) -> None:
    """Train one template-specific parameter model.

    Args:
        model (ParameterLSTM): Template-specific parameter model.
        optimizer (torch.optim.Optimizer): Optimiser used for training.
        training_run (_ParameterTrainingRun): In-memory training tensors and
            settings.
    """
    dataset_size = len(training_run.inputs)
    effective_batch_size = min(training_run.batch_size, dataset_size)
    for _ in range(training_run.epochs):
        model.train()
        permutation = torch.randperm(dataset_size, device=training_run.device)
        for start in range(0, dataset_size, effective_batch_size):
            batch_indexes = permutation[start : start + effective_batch_size]
            optimizer.zero_grad()
            predictions = model(training_run.inputs[batch_indexes])
            loss = masked_regression_loss(
                outputs=predictions,
                targets=training_run.targets[batch_indexes],
                mask=training_run.masks[batch_indexes],
            )
            loss.backward()
            optimizer.step()


def parameter_pair_residual(
    *,
    model: ParameterLSTM,
    pair: ParameterTrainingPair,
) -> float:
    """Return one validation residual for Gaussian calibration.

    Args:
        model (ParameterLSTM): Fitted parameter model.
        pair (ParameterTrainingPair): Validation pair to score.

    Returns:
        float: Masked mean squared residual for the pair.
    """
    with torch.inference_mode():
        predicted = (
            model(
                torch.tensor(
                    [pair.history_inputs],
                    dtype=torch.float32,
                    device=next(model.parameters()).device,
                ),
            )
            .cpu()
            .squeeze(0)
            .tolist()
        )
    return masked_mse(observed=pair.target, predicted=predicted, mask=pair.target_mask)


def parameter_pair_residuals(
    *,
    model: ParameterLSTM,
    pairs: list[ParameterTrainingPair],
    batch_size: int,
) -> list[float]:
    """Return validation residuals for one parameter model.

    Args:
        model (ParameterLSTM): Fitted parameter model.
        pairs (list[ParameterTrainingPair]): Validation pairs to score.
        batch_size (int): Batch size used for residual computation.

    Returns:
        list[float]: Masked mean squared residuals for the validation pairs.
    """
    if not pairs:
        return []
    model_device = next(model.parameters()).device
    residuals: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        inputs = torch.tensor(
            [pair.history_inputs for pair in batch_pairs],
            dtype=torch.float32,
            device=model_device,
        )
        with torch.inference_mode():
            predicted_batch = model(inputs).cpu().tolist()
        for pair, predicted in zip(batch_pairs, predicted_batch, strict=True):
            residuals.append(
                masked_mse(
                    observed=pair.target,
                    predicted=predicted,
                    mask=pair.target_mask,
                ),
            )
    return residuals
