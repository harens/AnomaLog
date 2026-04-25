"""Public fluent dataset builder API."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from anomalog._runtime.models import TemplatedDatasetBuildRequest
from anomalog._runtime.services import build_templated_dataset
from anomalog.cache import CachePathsConfig, clear_dataset_cache
from anomalog.parsers.structured import ParquetStructuredSink
from anomalog.parsers.template import Drain3Parser

if TYPE_CHECKING:
    from anomalog.labels import AnomalyLabelReader
    from anomalog.parsers.structured import StructuredParser, StructuredSink
    from anomalog.parsers.template import TemplatedDataset, TemplateParser
    from anomalog.sources import DatasetSource


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Immutable fluent builder for configuring a dataset pipeline.

    The builder captures dataset preprocessing choices without executing any
    pipeline stage immediately. Each fluent method returns a new spec so callers
    can share partially configured specs safely and only trigger orchestration at
    `build()` time.

    Attributes:
        dataset_name (str): Stable dataset identifier used for cache roots and
            materialised outputs.
        source (DatasetSource | None): Source that materialises or locates the raw
            dataset contents before parsing.
        structured_parser (StructuredParser | None): Parser that turns raw log
            lines into structured records.
        structured_sink (type[StructuredSink]): Sink implementation responsible
            for persisting structured rows and later grouped iteration.
        cache_paths (CachePathsConfig): Data/cache roots used by the build.
        anomaly_label_reader (AnomalyLabelReader | None): Optional anomaly label
            reader bound after structured parsing.
        template_parser (type[TemplateParser]): Template parser type used to mine
            message templates from structured records.
    """

    dataset_name: str
    source: DatasetSource | None = None
    structured_parser: StructuredParser | None = None
    structured_sink: type[StructuredSink] = field(
        default=ParquetStructuredSink,
    )
    cache_paths: CachePathsConfig = field(default_factory=CachePathsConfig)
    anomaly_label_reader: AnomalyLabelReader | None = None
    template_parser: type[TemplateParser] = field(default=Drain3Parser)

    def from_source(self, source: DatasetSource) -> DatasetSpec:
        """Bind the raw dataset source for later materialisation.

        Args:
            source (DatasetSource): Source strategy that knows how to provide the
                raw logs for this dataset.

        Returns:
            DatasetSpec: New spec with the supplied source attached.
        """
        return replace(self, source=source)

    def parse_with(self, structured_parser: StructuredParser) -> DatasetSpec:
        """Bind the structured parser that defines log-line semantics.

        Args:
            structured_parser (StructuredParser): Parser used during build to
                convert raw lines into structured records.

        Returns:
            DatasetSpec: New spec with the supplied parser attached.
        """
        return replace(self, structured_parser=structured_parser)

    def store_with(self, structured_sink: type[StructuredSink]) -> DatasetSpec:
        """Override the structured sink implementation for this dataset.

        Args:
            structured_sink (type[StructuredSink]): Sink type that owns
                persistence and grouped access for structured rows.

        Returns:
            DatasetSpec: New spec with the supplied sink type attached.
        """
        return replace(self, structured_sink=structured_sink)

    def label_with(self, anomaly_label_reader: AnomalyLabelReader) -> DatasetSpec:
        """Attach an anomaly label reader to enrich the built dataset.

        Args:
            anomaly_label_reader (AnomalyLabelReader): Reader used to resolve
                per-line or per-entity anomaly labels after parsing.

        Returns:
            DatasetSpec: New spec with the supplied label reader attached.
        """
        return replace(self, anomaly_label_reader=anomaly_label_reader)

    def template_with(self, template_parser: type[TemplateParser]) -> DatasetSpec:
        """Select the template parser implementation used during build.

        Args:
            template_parser (type[TemplateParser]): Template parser type trained on
                the structured dataset before the templated view is returned.

        Returns:
            DatasetSpec: New spec with the supplied template parser type attached.
        """
        return replace(self, template_parser=template_parser)

    def with_cache_paths(self, cache_paths: CachePathsConfig) -> DatasetSpec:
        """Override the default data and cache roots for this dataset.

        Args:
            cache_paths (CachePathsConfig): Explicit roots to use for source
                materialisation and derived local artifacts.

        Returns:
            DatasetSpec: New spec with the supplied cache paths attached.
        """
        return replace(self, cache_paths=cache_paths)

    def build(self) -> TemplatedDataset:
        """Build and return the templated dataset view.

        Returns:
            TemplatedDataset: Built dataset with structured rows, labels, and
                templates attached.
        """
        return build_templated_dataset(self._compile())

    def clear_cache(self) -> None:
        """Delete all local cached artifacts for this dataset.

        Raises:
            ValueError: If the dataset name is empty.
        """
        if not self.dataset_name:
            msg = "DatasetSpec.clear_cache() requires a non-empty dataset name."
            raise ValueError(msg)
        clear_dataset_cache(
            self.dataset_name,
            cache_paths=self.cache_paths,
        )

    def _compile(self) -> TemplatedDatasetBuildRequest:
        source, structured_parser = self._validate()
        template_parser = self.template_parser

        return TemplatedDatasetBuildRequest(
            dataset_name=self.dataset_name,
            source=source,
            structured_parser=structured_parser,
            structured_sink=self.structured_sink,
            cache_paths=self.cache_paths,
            anomaly_label_reader=self.anomaly_label_reader,
            template_parser=template_parser,
        )

    def _validate(self) -> tuple[DatasetSource, StructuredParser]:
        if not self.dataset_name:
            msg = "DatasetSpec.build() requires a non-empty dataset name."
            raise ValueError(msg)
        if self.source is None:
            msg = "DatasetSpec.build() requires a source."
            raise ValueError(msg)
        if self.structured_parser is None:
            msg = "DatasetSpec.build() requires a structured parser."
            raise ValueError(msg)

        return self.source, self.structured_parser
