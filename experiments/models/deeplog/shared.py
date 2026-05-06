"""Shared DeepLog config, state, and corpus helpers.

The goal of this module is to hold the small amount of state that is genuinely
shared across the DeepLog implementation:

- lightweight serialisable metadata types
- fitted per-template parameter-model state
- model classes used by both training and inference code
- training-corpus preparation helpers

It intentionally does not contain the actual fitting logic. Keeping the
algorithmic steps in `key.py`, `parameters.py`, and `detector.py` makes the
paper-to-code mapping easier to follow.
"""

from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass
from typing import TYPE_CHECKING

import msgspec
import torch
from torch import nn

from anomalog.parsers.structured.contracts import is_anomalous_label
from anomalog.sequences import SplitLabel
from experiments.models.base import ModelManifest, require_entity_local_sequences

if TYPE_CHECKING:
    from collections.abc import Iterable

    from rich.progress import Progress

    from anomalog.sequences import TemplateSequence

DT_FEATURE_NAME = "dt_prev_ms"
EPSILON = 1e-8
MIN_TEMPORAL_PARAMETER_PAIRS = 2


@dataclass(frozen=True, slots=True)
class GaussianThreshold:
    """Gaussian calibration stats for DeepLog parameter residuals.

    Attributes:
        mean (float): Mean validation residual.
        stddev (float): Standard deviation of validation residuals.
        lower_bound (float): Lower anomaly threshold bound.
        upper_bound (float): Upper anomaly threshold bound.
    """

    mean: float
    stddev: float
    lower_bound: float
    upper_bound: float


@dataclass(frozen=True, slots=True)
class ParameterFeatureSchema:
    """Stable per-template numeric feature schema for DeepLog parameter models.

    Attributes:
        feature_names (list[str]): Ordered numeric feature names.
        numeric_parameter_positions (list[int]): Original parameter indices kept
            as numeric features.
        include_elapsed_time (bool): Whether elapsed time is included as a feature.
        dropped_parameter_positions (list[int]): Original parameter indices
            rejected as non-numeric or unsupported.
    """

    feature_names: list[str]
    numeric_parameter_positions: list[int]
    include_elapsed_time: bool
    dropped_parameter_positions: list[int]


@dataclass(frozen=True, slots=True)
class NormalisationStats:
    """Per-feature normalisation statistics for one template model.

    Attributes:
        means (list[float]): Per-feature training means.
        stddevs (list[float]): Per-feature training standard deviations.
    """

    means: list[float]
    stddevs: list[float]


@dataclass(slots=True)
class ParameterModelState:
    """Runtime state for one DeepLog per-template parameter-value model.

    Attributes:
        template (str): Template this parameter model is scoped to.
        schema (ParameterFeatureSchema): Stable feature schema for the model.
        normalisation (NormalisationStats): Per-feature normalisation stats.
        gaussian (GaussianThreshold): Residual calibration thresholds.
        model (ParameterLSTM): Fitted parameter model.
    """

    template: str
    schema: ParameterFeatureSchema
    normalisation: NormalisationStats
    gaussian: GaussianThreshold
    model: ParameterLSTM


class ParameterModelManifestEntry(msgspec.Struct, frozen=True):
    """Serialisable manifest entry for one template-specific parameter model.

    Attributes:
        template (str): Template this parameter model is scoped to.
        feature_count (int): Total number of predicted features.
        input_feature_count (int): Number of input features fed to the model.
        feature_names (list[str]): Ordered feature names.
        numeric_parameter_positions (list[int]): Original numeric parameter
            positions retained in the model.
        dropped_parameter_positions (list[int]): Original parameter positions
            omitted from the model.
        gaussian_mean (float): Mean validation residual.
        gaussian_stddev (float): Standard deviation of validation residuals.
        gaussian_lower_bound (float): Lower anomaly threshold bound.
        gaussian_upper_bound (float): Upper anomaly threshold bound.
    """

    template: str
    feature_count: int
    input_feature_count: int
    feature_names: list[str]
    numeric_parameter_positions: list[int]
    dropped_parameter_positions: list[int]
    gaussian_mean: float
    gaussian_stddev: float
    gaussian_lower_bound: float
    gaussian_upper_bound: float


class SkippedParameterModelEntry(msgspec.Struct, frozen=True):
    """Serialisable reason for skipping a template-specific parameter model.

    Attributes:
        template (str): Template whose parameter model was skipped.
        reason (str): Human-readable reason the model was not trained.
    """

    template: str
    reason: str


class DeepLogTopPrediction(msgspec.Struct, frozen=True):
    """Top-`g` key prediction candidate from DeepLog's next-event model.

    This corresponds to the paper's log-key anomaly detection component, which
    treats an event as normal when the observed next log key is among the
    highest-probability predicted candidates.

    Attributes:
        template (str): Predicted log-key template.
        probability (float): Model probability assigned to the template.
    """

    template: str
    probability: float


class DeepLogKeyFinding(msgspec.Struct, frozen=True):
    """Event-level finding from DeepLog's stacked-LSTM key predictor.

    This captures the paper's next-log-key anomaly decision for one target
    event, including the observed template, whether it was out of vocabulary,
    how unknown history items were handled, and the ranked top-`g` candidates
    produced by the model.

    Attributes:
        event_index (int): Index of the event within the sequence.
        history_templates (list[str]): History window used for the prediction.
        unknown_history_templates (list[str]): Templates in the history that
            were unseen during training.
        actual_template (str): Observed next log-key template.
        actual_probability (float | None): Probability assigned to the
            observed template, if available.
        is_anomalous (bool): Whether the key-model decision is anomalous.
        is_oov (bool): Whether the observed template was out of vocabulary.
        top_predictions (list[DeepLogTopPrediction]): Ranked top predictions.
    """

    event_index: int
    history_templates: list[str]
    unknown_history_templates: list[str]
    actual_template: str
    actual_probability: float | None
    is_anomalous: bool
    is_oov: bool
    top_predictions: list[DeepLogTopPrediction]


class DeepLogParameterFinding(msgspec.Struct, frozen=True):
    """Event-level finding from a per-template DeepLog parameter model.

    This corresponds to the paper's parameter-value anomaly detection model:
    one LSTM per log key, residual scored by MSE, and thresholded by a
    Gaussian fitted on validation residuals.

    Attributes:
        event_index (int): Index of the event within the sequence.
        template (str): Template observed at that event.
        feature_names (list[str]): Ordered parameter feature names.
        observed_vector (list[float | None]): Observed raw feature values.
        predicted_vector (list[float | None]): Predicted raw feature values.
        residual_mse (float): Masked mean squared residual.
        gaussian_mean (float): Mean validation residual.
        gaussian_stddev (float): Standard deviation of validation residuals.
        gaussian_lower_bound (float): Lower Gaussian threshold bound.
        gaussian_upper_bound (float): Upper Gaussian threshold bound.
        most_anomalous_feature (str | None): Feature with the largest error.
        is_anomalous (bool): Whether the residual breaches the threshold.
    """

    event_index: int
    template: str
    feature_names: list[str]
    observed_vector: list[float | None]
    predicted_vector: list[float | None]
    residual_mse: float
    gaussian_mean: float
    gaussian_stddev: float
    gaussian_lower_bound: float
    gaussian_upper_bound: float
    most_anomalous_feature: str | None
    is_anomalous: bool


class DeepLogEventFinding(msgspec.Struct, frozen=True):
    """Combined DeepLog event-level anomaly finding.

    DeepLog reports anomalies at the event level, then aggregates them into a
    sequence-level prediction for the surrounding AnomaLog experiment contract.
    Keeping both key-model and parameter-model findings here makes it possible
    to inspect which stage fired.

    Attributes:
        event_index (int): Index of the event within the sequence.
        template (str): Template observed at that event.
        key_model_finding (DeepLogKeyFinding | None): Key-model evidence, if any.
        parameter_model_finding (DeepLogParameterFinding | None): Parameter-model
            evidence, if any.
    """

    event_index: int
    template: str
    key_model_finding: DeepLogKeyFinding | None = None
    parameter_model_finding: DeepLogParameterFinding | None = None


class DeepLogManifest(ModelManifest, frozen=True):
    """Serialisable manifest for the scoped DeepLog implementation.

    Attributes:
        implementation_scope (str): Human-readable scope/variant identifier.
        parameter_schema_policy (str): Policy used to derive parameter schemas.
        parameter_validation_policy (str): Policy used when validating parameter
            model inputs.
        history_size (int): Key-model history length.
        top_g (int): Number of top key predictions treated as normal.
        num_layers (int): LSTM layer count used by DeepLog models.
        hidden_size (int): Shared LSTM hidden size.
        epochs (int): Training epochs.
        batch_size (int): Training batch size.
        learning_rate (float): Optimiser learning rate.
        validation_fraction (float): Fraction reserved for validation.
        gaussian_confidence (float): Confidence level used for Gaussian bounds.
        include_elapsed_time (bool): Whether elapsed time is modeled as a
            parameter feature.
        train_key_vocabulary_size (int): Key-model vocabulary size from training.
        trained_parameter_model_count (int): Number of per-template parameter
            models trained successfully.
        skipped_parameter_model_count (int): Number of per-template parameter
            models skipped.
        train_parameter_covered_event_count (int): Training events covered by
            parameter models.
        train_parameter_covered_event_fraction (float): Covered-event fraction on
            the training corpus.
        scored_parameter_event_count (int): Scored inference events covered by
            parameter models.
        scored_parameter_event_fraction (float): Covered-event fraction during
            scoring.
        parameter_models (list[ParameterModelManifestEntry]): Manifest entries for
            trained parameter models.
        skipped_parameter_models (list[SkippedParameterModelEntry]): Reasons for
            skipped parameter models.
    """

    implementation_scope: str
    parameter_schema_policy: str
    parameter_validation_policy: str
    history_size: int
    top_g: int
    num_layers: int
    hidden_size: int
    epochs: int
    batch_size: int
    learning_rate: float
    validation_fraction: float
    gaussian_confidence: float
    include_elapsed_time: bool
    train_key_vocabulary_size: int
    trained_parameter_model_count: int
    skipped_parameter_model_count: int
    train_parameter_covered_event_count: int
    train_parameter_covered_event_fraction: float
    scored_parameter_event_count: int
    scored_parameter_event_fraction: float
    parameter_models: list[ParameterModelManifestEntry]
    skipped_parameter_models: list[SkippedParameterModelEntry]


@dataclass(frozen=True, slots=True)
class NormalTrainingCorpus:
    """Replayable normal-only training state for the offline DeepLog detector.

    Attributes:
        sequences (tuple[TemplateSequence, ...]): Normal training sequences kept
            for replay across DeepLog submodels.
        templates (tuple[str, ...]): Sorted unique training templates.
        event_count (int): Total number of events across normal training sequences.
    """

    sequences: tuple[TemplateSequence, ...]
    templates: tuple[str, ...]
    event_count: int


def training_event_mask_for_sequence(sequence: TemplateSequence) -> tuple[bool, ...]:
    """Return the DeepLog training-target eligibility mask for one sequence.

    Args:
        sequence (TemplateSequence): Sequence whose training-target eligibility
            mask should be derived.

    Returns:
        tuple[bool, ...]: Per-event eligibility mask.

    When a sequence carries an explicit `training_event_mask`, that mask is the
    source of truth. Otherwise the legacy whole-sequence policy is preserved:
    only normal sequences contribute training targets.
    """
    explicit_mask = sequence.training_event_mask
    if explicit_mask is not None:
        return explicit_mask
    if is_anomalous_label(sequence.label):
        return tuple(False for _ in sequence.events)
    return tuple(True for _ in sequence.events)


def evaluation_event_mask_for_sequence(sequence: TemplateSequence) -> tuple[bool, ...]:
    """Return the DeepLog evaluation-target mask for one sequence.

    Args:
        sequence (TemplateSequence): Sequence whose scoring-target eligibility
            mask should be derived.

    Returns:
        tuple[bool, ...]: Per-event scoring eligibility mask.

    When a sequence carries an explicit `evaluation_event_mask`, that mask is
    the source of truth. Otherwise the legacy split-label policy is preserved:
    only test sequences contribute evaluation targets.
    """
    explicit_mask = sequence.evaluation_event_mask
    if explicit_mask is not None:
        return explicit_mask
    if sequence.split_label is not SplitLabel.TEST:
        return tuple(False for _ in sequence.events)
    return tuple(True for _ in sequence.events)


def training_event_index_mask(sequence: TemplateSequence) -> list[int]:
    """Return the eligible DeepLog training target indexes for one sequence.

    Args:
        sequence (TemplateSequence): Sequence whose eligible target indexes
            should be derived.

    Returns:
        list[int]: Zero-based indexes of eligible training targets.
    """
    return [
        event_index
        for event_index, is_eligible in enumerate(
            training_event_mask_for_sequence(sequence),
        )
        if is_eligible
    ]


def evaluation_event_index_mask(sequence: TemplateSequence) -> list[int]:
    """Return the eligible DeepLog evaluation target indexes for one sequence.

    Args:
        sequence (TemplateSequence): Sequence whose eligible target indexes
            should be derived.

    Returns:
        list[int]: Zero-based indexes of eligible evaluation targets.
    """
    return [
        event_index
        for event_index, is_eligible in enumerate(
            evaluation_event_mask_for_sequence(sequence),
        )
        if is_eligible
    ]


class KeyLSTM(nn.Module):
    """Stacked-LSTM next-log-key predictor.

    The paper describes the key model as taking a window of one-hot encoded
    log-key vectors and predicting the next key with a softmax output layer.
    This module follows that structure directly:

    - input shape: `(batch, history_size, vocab_size)`
    - recurrent core: stacked LSTM layers
    - output shape: `(batch, vocab_size)` logits for the next key

    Args:
        vocab_size (int): Size of the one-hot key vocabulary.
        hidden_size (int): Hidden width of each LSTM layer.
        num_layers (int): Number of stacked LSTM layers.
    """

    def __init__(self, *, vocab_size: int, hidden_size: int, num_layers: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return logits over the next log key.

        Args:
            inputs (torch.Tensor): One-hot key histories with shape
                `(batch, history_size, vocab_size)`.

        Returns:
            torch.Tensor: Next-key logits with shape `(batch, vocab_size)`.
        """
        outputs, _ = self.lstm(inputs)
        return self.output(outputs[:, -1, :])


class ParameterLSTM(nn.Module):
    """Template-specific DeepLog parameter-value predictor.

    Each template/log key gets its own model. The model consumes a short
    history of normalised parameter vectors for that template and predicts the
    next normalised parameter vector for the same template.

    Args:
        input_size (int): Number of features in each history vector.
        hidden_size (int): Hidden width of each LSTM layer.
        num_layers (int): Number of stacked LSTM layers.
        output_size (int): Number of features predicted for the next event.
    """

    def __init__(
        self,
        *,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the predicted next parameter vector.

        Args:
            inputs (torch.Tensor): Parameter histories with shape
                `(batch, history_size, input_size)`.

        Returns:
            torch.Tensor: Predicted next vectors with shape `(batch, output_size)`.
        """
        outputs, _ = self.lstm(inputs)
        return self.output(outputs[:, -1, :])


def build_normal_training_corpus(
    train_sequences: Iterable[TemplateSequence],
    *,
    progress: Progress,
) -> NormalTrainingCorpus:
    """Collect replayable normal-target training state for offline DeepLog.

    Args:
        train_sequences (Iterable[TemplateSequence]): Train-split sequences.
        progress (Progress): Progress reporter.

    Returns:
        NormalTrainingCorpus: Cached training state for replay.

    Raises:
        ValueError: If no normal sequences are available for training.
    """
    # The paper trains DeepLog from normal execution only. For raw-entry
    # chronological streams we preserve mixed chunks as context but filter
    # training targets at the event level, so we materialise the replayable
    # corpus once and let the model-specific builders consume the eligibility
    # mask for each event.
    total = len(train_sequences) if isinstance(train_sequences, Sized) else None
    prepare_task = progress.add_task(
        "Preparing DeepLog training corpus",
        total=total,
    )
    normal_sequences: list[TemplateSequence] = []
    template_set: set[str] = set()
    event_count = 0
    try:
        for sequence in train_sequences:
            require_entity_local_sequences((sequence,), detector_name="DeepLog")
            eligible_indexes = training_event_index_mask(sequence)
            if not eligible_indexes:
                progress.advance(prepare_task)
                continue
            normal_sequences.append(sequence)
            event_count += len(eligible_indexes)
            template_set.update(sequence.templates)
            progress.advance(prepare_task)
    finally:
        progress.remove_task(prepare_task)
    if not normal_sequences:
        msg = "DeepLog requires at least one eligible training target."
        raise ValueError(msg)
    return NormalTrainingCorpus(
        sequences=tuple(normal_sequences),
        templates=tuple(sorted(template_set)),
        event_count=event_count,
    )
