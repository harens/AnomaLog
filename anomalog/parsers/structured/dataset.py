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
    """Structured dataset with labels ready for template mining."""

    sink: StructuredSink
    cache_paths: CachePathsConfig
    anomaly_labels: AnomalyLabelLookup

    def mine_templates_with(self, template_parser: TemplateParser) -> TemplatedDataset:
        """Train a template parser and return a templated dataset view."""
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
