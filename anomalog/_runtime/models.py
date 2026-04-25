"""Internal build request models for flow-backed dataset construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anomalog.cache import CachePathsConfig
    from anomalog.labels import AnomalyLabelReader
    from anomalog.parsers.structured import StructuredParser, StructuredSink
    from anomalog.parsers.template import TemplateParser
    from anomalog.sources import DatasetSource


@dataclass(frozen=True, slots=True)
class TemplatedDatasetBuildRequest:
    """Fully-specified internal dataset build request.

    This keeps the runtime flow boundary narrow: builder-time validation happens
    before execution, and the flow receives one immutable payload with every
    dependency resolved. That avoids runtime code guessing defaults from partial
    public builder state.

    Attributes:
        dataset_name (str): Stable dataset identifier used for cache and output
            paths.
        source (DatasetSource): Materialisation strategy that must already be
            chosen by the public builder.
        structured_parser (StructuredParser): Parser used to turn raw log lines
            into structured records.
        structured_sink (type[StructuredSink]): Sink implementation that owns
            persistence and grouped iteration semantics.
        cache_paths (CachePathsConfig): Resolved data/cache roots for this build.
        anomaly_label_reader (AnomalyLabelReader | None): Optional reader for
            dataset-level anomaly labels after structured parsing.
        template_parser (type[TemplateParser]): Template parser type to train and
            bind to the structured dataset.
    """

    dataset_name: str
    source: DatasetSource
    structured_parser: StructuredParser
    structured_sink: type[StructuredSink]
    cache_paths: CachePathsConfig
    anomaly_label_reader: AnomalyLabelReader | None
    template_parser: type[TemplateParser]
