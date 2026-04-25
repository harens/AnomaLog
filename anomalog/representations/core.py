"""Sequence representation types for downstream anomaly detectors."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeVar

from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterator

    from anomalog.sequences import SequenceBuilder, SplitLabel, TemplateSequence

TRepresentation = TypeVar("TRepresentation")


class SequenceRepresentation(Protocol[TRepresentation]):
    """Protocol for converting full grouped sequences into model inputs.

    Implementations receive the complete `TemplateSequence`, including event
    timings, extracted parameters, entity IDs, labels, and split metadata, and
    may choose whichever fields are relevant for a detector.

    Attributes:
        name (ClassVar[str]): Stable registry/config name for the representation.
    """

    name: ClassVar[str]

    def represent(self, sequence: TemplateSequence) -> TRepresentation:
        """Convert one grouped sequence into a representation payload.

        Args:
            sequence (TemplateSequence): Full grouped sequence carrying events,
                labels, entity ids, and split metadata.

        Returns:
            TRepresentation: Detector-specific representation of the sequence.
        """


@dataclass(slots=True, frozen=True)
class SequenceSample(Generic[TRepresentation]):
    """Model-ready data derived from a `TemplateSequence`.

    `TemplateSequence` is the grouped log window; `SequenceSample` is the
    representation-specific payload passed to a detector.

    Attributes:
        data (TRepresentation): Detector-ready representation payload.
        label (int): Sequence-level anomaly label derived from the source window.
        entity_ids (list[str]): Unique entity ids present in the source window.
        split_label (SplitLabel): Train/test split assigned during sequence
            building.
        window_id (int): Stable window identifier within the sequence builder.
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
        """Build a model-ready sample from one template sequence.

        Args:
            sequence (TemplateSequence): Source grouped sequence carrying labels
                and metadata.
            data (TRepresentation): Representation payload derived from the
                sequence.

        Returns:
            SequenceSample[TRepresentation]: Sample carrying the represented
                payload together with the original sequence metadata.
        """
        return cls(
            data=data,
            label=sequence.label,
            entity_ids=sequence.entity_ids,
            split_label=sequence.split_label,
            window_id=sequence.window_id,
        )

    def as_labeled_example(self) -> tuple[TRepresentation, int]:
        """Return a generic `(x, y)` example pair.

        Returns:
            tuple[TRepresentation, int]: Representation payload and label.
        """
        return self.data, self.label


@dataclass(slots=True, frozen=True)
class SequenceRepresentationView(Generic[TRepresentation]):
    """Lazy iterable over represented sequence samples.

    The representation stage is the point where a model decides which parts of
    `TemplateSequence` matter; the full sequence object is passed through to the
    representation implementation on each iteration.

    Attributes:
        sequences (SequenceBuilder): Underlying sequence builder producing
            `TemplateSequence` objects lazily.
        representation (SequenceRepresentation[TRepresentation]): Representation
            applied to each yielded sequence.
    """

    sequences: SequenceBuilder
    representation: SequenceRepresentation[TRepresentation]

    def __iter__(self) -> Iterator[SequenceSample[TRepresentation]]:
        """Yield represented sequence samples.

        Yields:
            SequenceSample[TRepresentation]: One represented sample per input
                template sequence.
        """
        for sequence in self.sequences:
            yield SequenceSample.from_sequence(
                sequence,
                data=self.representation.represent(sequence),
            )

    def iter_labeled_examples(self) -> Iterator[tuple[TRepresentation, int]]:
        """Yield `(x, y)` pairs only, intentionally dropping split metadata.

        Yields:
            tuple[TRepresentation, int]: Representation payload and label pairs.
        """
        for sample in self:
            yield sample.as_labeled_example()


@dataclass(slots=True, frozen=True)
class SequentialRepresentation(SequenceRepresentation[list[str]]):
    """Ordered template-only representation for sequential models.

    Attributes:
        name (ClassVar[str]): Registry/config name for the representation.
    """

    name: ClassVar[str] = "sequential"

    @override
    def represent(self, sequence: TemplateSequence) -> list[str]:
        """Return the ordered template stream for one sequence.

        Args:
            sequence (TemplateSequence): Sequence whose template order should be
                preserved exactly.

        Returns:
            list[str]: Ordered template stream for the sequence.
        """
        return sequence.templates


@dataclass(slots=True, frozen=True)
class TemplateCountRepresentation(SequenceRepresentation[Counter[str]]):
    """Count-based representation that intentionally uses template text only.

    Attributes:
        name (ClassVar[str]): Registry/config name for the representation.
    """

    name: ClassVar[str] = "template_counts"

    @override
    def represent(self, sequence: TemplateSequence) -> Counter[str]:
        """Return one template-count vector.

        Args:
            sequence (TemplateSequence): Sequence whose template frequencies are
                being counted.

        Returns:
            Counter[str]: Template-frequency vector for the sequence.
        """
        return Counter(sequence.templates)


@dataclass(slots=True, frozen=True)
class TemplatePhraseRepresentation(SequenceRepresentation[Counter[str]]):
    """Phrase-count representation derived from template text only.

    This expands each template into normalsed full-template phrases and token
    n-grams. The representation deliberately ignores parameters and timing so
    phrase-based detectors react only to recurring message wording.

    Attributes:
        name (ClassVar[str]): Registry/config name for the representation.
        phrase_ngram_min (int): Smallest token n-gram size to emit.
        phrase_ngram_max (int): Largest token n-gram size to emit.
    """

    name: ClassVar[str] = "template_phrases"
    phrase_ngram_min: int = 1
    phrase_ngram_max: int = 2

    def __post_init__(self) -> None:
        """Validate phrase extraction settings.

        Raises:
            ValueError: If the configured n-gram bounds are invalid.
        """
        if self.phrase_ngram_min < 1:
            msg = "phrase_ngram_min must be at least 1."
            raise ValueError(msg)
        if self.phrase_ngram_max < self.phrase_ngram_min:
            msg = "phrase_ngram_max must be >= phrase_ngram_min."
            raise ValueError(msg)

    @override
    def represent(self, sequence: TemplateSequence) -> Counter[str]:
        """Return one phrase-count vector.

        Args:
            sequence (TemplateSequence): Sequence whose template phrases should be
                counted.

        Returns:
            Counter[str]: Phrase-frequency vector for the sequence.
        """
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
    """Extract normalised template and token n-gram phrases.

    Args:
        template (str): Template text to tokenize and normalise.
        phrase_ngram_min (int): Minimum token n-gram size to include.
        phrase_ngram_max (int): Maximum token n-gram size to include.

    Returns:
        list[str]: Normalised full-template phrases plus token n-grams.
    """
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
    """Split a template into lowercase alphanumeric tokens.

    Args:
        template (str): Template text to split into tokens.

    Returns:
        list[str]: Tokens extracted from the template text.
    """
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
