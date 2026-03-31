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
    """Fully-specified internal dataset build request."""

    dataset_name: str
    source: DatasetSource
    structured_parser: StructuredParser
    structured_sink: type[StructuredSink]
    cache_paths: CachePathsConfig
    anomaly_label_reader: AnomalyLabelReader | None
    template_parser: type[TemplateParser]
