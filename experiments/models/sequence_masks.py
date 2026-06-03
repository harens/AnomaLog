"""Shared event-mask helpers for experiment-layer sequence contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anomalog.parsers.structured.contracts import is_anomalous_label
from anomalog.sequences import SplitLabel

if TYPE_CHECKING:
    from anomalog.sequences import TemplateSequence


def training_event_mask_for_sequence(sequence: TemplateSequence) -> tuple[bool, ...]:
    """Return the training-target eligibility mask for one sequence.

    Args:
        sequence (TemplateSequence): Sequence whose training-target eligibility
            mask should be derived.

    Returns:
        tuple[bool, ...]: Per-event training eligibility mask.

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
    """Return the evaluation-target eligibility mask for one sequence.

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
    """Return the eligible training target indexes for one sequence.

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
    """Return the eligible evaluation target indexes for one sequence.

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
