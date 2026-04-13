"""Sequence representation types for downstream anomaly detectors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from anomalog.sequences import SequenceBuilder, SplitLabel, TemplateSequence

TRepresentation = TypeVar("TRepresentation")


class SequenceRepresentation(Protocol[TRepresentation]):
    """Protocol for converting full grouped sequences into model inputs.

    Implementations receive the complete `TemplateSequence`, including event
    timings, extracted parameters, entity IDs, labels, and split metadata, and
    may choose whichever fields are relevant for a detector.
    """

    name: ClassVar[str]

    def represent(self, sequence: TemplateSequence) -> TRepresentation:
        """Convert one grouped sequence into a representation payload."""


@dataclass(slots=True, frozen=True)
class SequenceSample(Generic[TRepresentation]):
    """Model-ready data derived from a `TemplateSequence`.

    `TemplateSequence` is the grouped log window; `SequenceSample` is the
    representation-specific payload passed to a detector.
    """

    data: TRepresentation
    label: int
    entity_ids: list[str]
    split_label: SplitLabel
    window_id: int

    @classmethod
    def from_sequence(
        cls,
        sequence: TemplateSequence,
        *,
        data: TRepresentation,
    ) -> SequenceSample[TRepresentation]:
        """Build a model-ready sample from one template sequence."""
        return cls(
            data=data,
            label=sequence.label,
            entity_ids=sequence.entity_ids,
            split_label=sequence.split_label,
            window_id=sequence.window_id,
        )

    def as_labeled_example(self) -> tuple[TRepresentation, int]:
        """Return a generic `(x, y)` example pair."""
        return self.data, self.label


@dataclass(slots=True, frozen=True)
class SequenceRepresentationView(Generic[TRepresentation]):
    """Lazy iterable over represented sequence samples.

    The representation stage is the point where a model decides which parts of
    `TemplateSequence` matter; the full sequence object is passed through to the
    representation implementation on each iteration.
    """

    sequences: SequenceBuilder
    representation: SequenceRepresentation[TRepresentation]

    def __iter__(self) -> Iterator[SequenceSample[TRepresentation]]:
        """Yield represented sequence samples."""
        for sequence in self.sequences:
            yield SequenceSample.from_sequence(
                sequence,
                data=self.representation.represent(sequence),
            )

    def iter_labeled_examples(self) -> Iterator[tuple[TRepresentation, int]]:
        """Yield `(x, y)` pairs only, intentionally dropping split metadata."""
        for sample in self:
            yield sample.as_labeled_example()


@dataclass(slots=True, frozen=True)
class SequentialRepresentation(SequenceRepresentation[list[str]]):
    """Ordered template-only representation for sequential models."""

    name: ClassVar[str] = "sequential"

    def represent(self, sequence: TemplateSequence) -> list[str]:
        """Return the ordered template stream for one sequence."""
        return sequence.templates


@dataclass(slots=True, frozen=True)
class TemplateCountRepresentation(SequenceRepresentation[Counter[str]]):
    """Count-based representation that intentionally uses template text only."""

    name: ClassVar[str] = "template_counts"

    def represent(self, sequence: TemplateSequence) -> Counter[str]:
        """Return one template-count vector."""
        return Counter(sequence.templates)


@dataclass(slots=True, frozen=True)
class TemplatePhraseRepresentation(SequenceRepresentation[Counter[str]]):
    """Phrase-count representation derived from template text only."""

    name: ClassVar[str] = "template_phrases"
    phrase_ngram_min: int = 1
    phrase_ngram_max: int = 2

    def __post_init__(self) -> None:
        """Validate phrase extraction settings."""
        if self.phrase_ngram_min < 1:
            msg = "phrase_ngram_min must be at least 1."
            raise ValueError(msg)
        if self.phrase_ngram_max < self.phrase_ngram_min:
            msg = "phrase_ngram_max must be >= phrase_ngram_min."
            raise ValueError(msg)

    def represent(self, sequence: TemplateSequence) -> Counter[str]:
        """Return one phrase-count vector."""
        phrase_counts: Counter[str] = Counter()
        for template in sequence.templates:
            phrase_counts.update(
                _extract_template_phrases(
                    template,
                    phrase_ngram_min=self.phrase_ngram_min,
                    phrase_ngram_max=self.phrase_ngram_max,
                ),
            )
        return phrase_counts


def _extract_template_phrases(
    template: str,
    *,
    phrase_ngram_min: int,
    phrase_ngram_max: int,
) -> list[str]:
    """Extract normalized template and token n-gram phrases."""
    normalized_template = " ".join(template.split()).strip().lower()
    phrases: list[str] = []
    if normalized_template:
        phrases.append(normalized_template)

    tokens = list(_iter_template_tokens(template))
    if not tokens:
        return phrases

    for ngram_size in range(phrase_ngram_min, phrase_ngram_max + 1):
        if len(tokens) < ngram_size:
            continue
        phrases.extend(
            " ".join(tokens[start_idx : start_idx + ngram_size])
            for start_idx in range(len(tokens) - ngram_size + 1)
        )
    return phrases


def _iter_template_tokens(template: str) -> list[str]:
    """Split a template into lowercase alphanumeric tokens."""
    tokens: list[str] = []
    current: list[str] = []
    for char in template:
        if char.isalnum():
            current.append(char.lower())
            continue
        if current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tokens
