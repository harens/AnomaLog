"""Abstractions for templated datasets and template parser contracts."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, TypeAlias, runtime_checkable

from prefect.assets import Asset

from anomalog.cache import CachePathsConfig
from anomalog.parsers.structured.contracts import StructuredSink
from anomalog.sequences import (
    ChronologicalStreamSequenceBuilder,
    EntitySequenceBuilder,
    FixedSequenceBuilder,
    FixedWindowBasis,
    TimeSequenceBuilder,
)

if TYPE_CHECKING:
    from anomalog.labels import AnomalyLabelLookup

UntemplatedText: TypeAlias = str
LogTemplate: TypeAlias = str
ExtractedParameters: TypeAlias = Sequence[str]


# TODO: Add visualisation methods
@runtime_checkable
class TemplateParser(Protocol):
    """Interface for template mining implementations.

    Implementations are initialised with an optional dataset name so runtime
    caches can be scoped per dataset when needed.

    Attributes:
        name (ClassVar[str]): Stable registry/config name for the parser.
        is_identity_parser (ClassVar[bool]): Whether the parser leaves the raw
            text unchanged. Parsers default to `False`; identity parsers
            override this to `True` so sequence builders can use dense raw
            source fast paths without guessing from class names.
        dataset_name (str | None): Optional dataset identifier used to scope
            runtime caches or persisted parser state.
    """

    name: ClassVar[str]
    is_identity_parser: ClassVar[bool] = False
    dataset_name: str | None

    def inference(
        self,
        unstructured_text: UntemplatedText,
    ) -> tuple[LogTemplate, ExtractedParameters]:
        """Infer the normalsed template and extracted parameters for one line.

        Args:
            unstructured_text (UntemplatedText): Raw untemplated message text.

        Returns:
            tuple[LogTemplate, ExtractedParameters]: Template text plus any
                extracted parameter values derived from the message.
        """

    def train(
        self,
        untemplated_text_iterator: Callable[[], Iterator[UntemplatedText]],
        *,
        asset_deps: list[Asset] | None = None,
    ) -> None:
        """Train the parser on the dataset's untemplated message stream.

        Args:
            untemplated_text_iterator (Callable[[], Iterator[UntemplatedText]]):
                Zero-argument iterator factory over untemplated message text.
            asset_deps (list[Asset] | None): Optional upstream asset
                dependencies to incorporate into the training cache key.
        """


@dataclass(slots=True, frozen=True)
class TemplatedDataset:
    """Structured dataset paired with a trained template parser and labels.

    This is the post-build dataset view returned to callers. It keeps the
    sink-backed structured rows, trained template inference function, and
    anomaly labels aligned so sequence builders can derive consistent windows
    without exposing Prefect or runtime orchestration details.

    Attributes:
        sink (StructuredSink): Structured sink that owns persisted parsed rows.
        cache_paths (CachePathsConfig): Data/cache roots associated with the
            build that produced this dataset.
        template_parser (TemplateParser): Trained template parser used for
            template inference over structured rows.
        anomaly_labels (AnomalyLabelLookup): Normalised anomaly label lookups
            attached to the dataset.
    """

    sink: StructuredSink
    cache_paths: CachePathsConfig
    template_parser: TemplateParser
    anomaly_labels: "AnomalyLabelLookup"

    def sequences(self) -> EntitySequenceBuilder:
        """Return the default entity-grouped sequence builder.

        Returns:
            EntitySequenceBuilder: Default entity-grouped sequence view.
        """
        return EntitySequenceBuilder.from_dataset(self)

    def group_by_entity(self) -> EntitySequenceBuilder:
        """Group sequences by entity id.

        Returns:
            EntitySequenceBuilder: Entity-grouped sequence view.
        """
        return self.sequences()

    def group_by_chronological_stream(
        self,
        *,
        chunk_size: int = 100_000,
        continuous_context: bool = True,
    ) -> ChronologicalStreamSequenceBuilder:
        """Group sequences into deterministic chronological stream chunks.

        Args:
            chunk_size (int): Maximum number of raw entries per emitted chunk.
            continuous_context (bool): Whether emitted chunks should carry
                model state across chunk boundaries.

        Returns:
            ChronologicalStreamSequenceBuilder: Chronological stream view.
        """
        return ChronologicalStreamSequenceBuilder(
            sink=self.sink,
            infer_template=self.template_parser.inference,
            label_for_group=self.anomaly_labels.label_for_group,
            chunk_size=chunk_size,
            continuous_context=continuous_context,
        )

    def group_by_fixed_window(
        self,
        window_size: int,
        step_size: int | None = None,
        *,
        window_basis: FixedWindowBasis = FixedWindowBasis.COMPACTED_ROWS,
        window_alignment_offset: int = 0,
    ) -> FixedSequenceBuilder:
        """Group sequences in fixed-size windows.

        Args:
            window_size (int): Number of rows in each emitted window.
            step_size (int | None): Optional step between successive windows.
                Defaults to `window_size`.
            window_basis (FixedWindowBasis): Positional basis used to build the
                fixed windows.
            window_alignment_offset (int): Raw-position offset before the first
                window when `window_basis` is `raw_positions`.

        Returns:
            FixedSequenceBuilder: Fixed-window sequence view.
        """
        return FixedSequenceBuilder(
            sink=self.sink,
            infer_template=self.template_parser.inference,
            label_for_group=self.anomaly_labels.label_for_group,
            window_size=window_size,
            step=step_size,
            window_basis=window_basis,
            window_alignment_offset=window_alignment_offset,
        )

    def group_by_time_window(
        self,
        time_span_ms: int,
        step_span_ms: int | None = None,
    ) -> TimeSequenceBuilder:
        """Group sequences using time-based sliding windows.

        Args:
            time_span_ms (int): Width of each emitted time window in
                milliseconds.
            step_span_ms (int | None): Optional step between successive windows.
                Defaults to `time_span_ms`.

        Returns:
            TimeSequenceBuilder: Time-window sequence view.
        """
        return TimeSequenceBuilder(
            sink=self.sink,
            infer_template=self.template_parser.inference,
            label_for_group=self.anomaly_labels.label_for_group,
            time_span_ms=time_span_ms,
            step=step_span_ms,
        )
