"""StructuredDataset wraps parsed logs prior to template mining."""

from dataclasses import dataclass

from anomalog.cache import CachePathsConfig
from anomalog.labels import AnomalyLabelLookup
from anomalog.parsers.structured.contracts import (
    UNTEMPLATED_FIELD,
    StructuredSink,
)
from anomalog.parsers.template.dataset import TemplatedDataset, TemplateParser


# TODO: Better way of managing the flow of data through the
# various stages of processing? Maybe a more explicit pipeline definition?
@dataclass(slots=True)
class StructuredDataset:
    """Structured dataset with labels ready for template mining.

    This is the narrow handoff between structured parsing and template mining.
    It keeps sink-backed structured rows and normalised anomaly label lookups
    together so downstream stages cannot accidentally mix artifacts from
    different dataset builds.

    Attributes:
        sink (StructuredSink): Structured sink that owns persisted parsed rows.
        cache_paths (CachePathsConfig): Data/cache roots associated with the
            current dataset build.
        anomaly_labels (AnomalyLabelLookup): Normalised per-line and per-group
            anomaly label accessors for the structured dataset.
    """

    sink: StructuredSink
    cache_paths: CachePathsConfig
    anomaly_labels: AnomalyLabelLookup

    def mine_templates_with(self, template_parser: TemplateParser) -> TemplatedDataset:
        """Train a template parser and return a templated dataset view.

        Args:
            template_parser (TemplateParser): Template parser to train over the
                structured dataset.

        Returns:
            TemplatedDataset: Structured dataset paired with the trained parser.
        """
        template_parser.train(
            lambda: (
                row.untemplated_message_text
                for row in self.sink.iter_structured_lines(
                    columns=[UNTEMPLATED_FIELD],
                )()
            ),
        )

        return TemplatedDataset(
            sink=self.sink,
            cache_paths=self.cache_paths,
            template_parser=template_parser,
            anomaly_labels=self.anomaly_labels,
        )
