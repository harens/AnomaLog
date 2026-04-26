"""Tests for public sequence representation helpers."""

import pytest

from anomalog.representations import (
    SequentialRepresentation,
    TemplateCountRepresentation,
    TemplatePhraseRepresentation,
)
from anomalog.sequences import SplitLabel, TemplateSequence


@pytest.mark.allow_no_new_coverage
def test_template_count_representation_preserves_sequence_counts() -> None:
    """Template-count representations preserve the sequence count map."""
    # This locks down the concrete public API name and return shape after the
    # representation split; nearby uncovered branches are phrase/sequential paths.
    sequence = TemplateSequence(
        events=[
            ("Node failed", [], None),
            ("Node failed", [], 50),
            ("Disk warning", [], 75),
        ],
        label=1,
        entity_ids=["node-a"],
        window_id=7,
        split_label=SplitLabel.TEST,
    )
    representation = TemplateCountRepresentation()

    assert representation.represent(sequence) == {
        "Node failed": 2,
        "Disk warning": 1,
    }


@pytest.mark.allow_no_new_coverage
def test_template_phrase_representation_extracts_ngrams() -> None:
    """Phrase representations include normalized templates and token n-grams."""
    # This protects the exact phrase feature contract; nearby uncovered branches
    # are validation and edge-case handling rather than the nominal mapping.
    sequence = TemplateSequence(
        events=[
            ("Node failed block 42", [], None),
            ("Disk warning", [], 50),
        ],
        label=0,
        entity_ids=["node-a"],
        window_id=1,
        split_label=SplitLabel.TRAIN,
    )
    representation = TemplatePhraseRepresentation(
        phrase_ngram_min=1,
        phrase_ngram_max=2,
    )

    assert representation.represent(sequence) == {
        "node failed block 42": 1,
        "node": 1,
        "failed": 1,
        "block": 1,
        "42": 1,
        "node failed": 1,
        "failed block": 1,
        "block 42": 1,
        "disk warning": 2,
        "disk": 1,
        "warning": 1,
    }


def test_template_phrase_representation_rejects_invalid_ngram_bounds() -> None:
    """Phrase representations validate n-gram settings eagerly."""
    with pytest.raises(ValueError, match="at least 1"):
        TemplatePhraseRepresentation(
            phrase_ngram_min=0,
        )

    with pytest.raises(ValueError, match=">= phrase_ngram_min"):
        TemplatePhraseRepresentation(
            phrase_ngram_min=2,
            phrase_ngram_max=1,
        )


def test_template_phrase_representation_handles_templates_without_tokens() -> None:
    """Phrase representations retain normalized templates when tokenization is empty."""
    sequence = TemplateSequence(
        events=[
            (" <*> ", [], None),
        ],
        label=0,
        entity_ids=["node-a"],
        window_id=3,
        split_label=SplitLabel.TRAIN,
    )

    representation = TemplatePhraseRepresentation(
        phrase_ngram_min=2,
        phrase_ngram_max=3,
    )

    assert representation.represent(sequence) == {"<*>": 1}


def test_template_phrase_representation_skips_oversized_ngrams() -> None:
    """Phrase extraction should skip n-gram sizes larger than the token stream."""
    sequence = TemplateSequence(
        events=[("warning", [], None)],
        label=0,
        entity_ids=["node-a"],
        window_id=4,
        split_label=SplitLabel.TRAIN,
    )

    representation = TemplatePhraseRepresentation(
        phrase_ngram_min=2,
        phrase_ngram_max=3,
    )

    assert representation.represent(sequence) == {"warning": 1}


@pytest.mark.allow_no_new_coverage
def test_template_phrase_representation_splits_non_alphanumeric_boundaries() -> None:
    """Phrase extraction should tokenize on any non-alphanumeric separator."""
    # This is a regression check for punctuation/underscore token boundaries.
    # The representation module is already fully covered, so there is no nearby
    # uncovered branch to exercise instead.
    sequence = TemplateSequence(
        events=[("Node_A-42/failure", [], None)],
        label=1,
        entity_ids=["node-a"],
        window_id=5,
        split_label=SplitLabel.TEST,
    )

    representation = TemplatePhraseRepresentation(
        phrase_ngram_min=1,
        phrase_ngram_max=2,
    )

    assert representation.represent(sequence) == {
        "node_a-42/failure": 1,
        "node": 1,
        "a": 1,
        "42": 1,
        "failure": 1,
        "node a": 1,
        "a 42": 1,
        "42 failure": 1,
    }


@pytest.mark.allow_no_new_coverage
def test_sequential_representation_preserves_template_order() -> None:
    """Sequential representations keep the original template ordering."""
    # This locks down the public sequential representation contract after
    # dropping parameter-appending behavior; the implementation is a direct
    # property passthrough, so nearby uncovered branches are not relevant here.
    sequence = TemplateSequence(
        events=[
            ("Node failed", [], None),
            ("Disk warning", ["sda"], 50),
        ],
        label=0,
        entity_ids=["node-a"],
        window_id=9,
        split_label=SplitLabel.TEST,
    )

    assert SequentialRepresentation().represent(sequence) == [
        "Node failed",
        "Disk warning",
    ]
