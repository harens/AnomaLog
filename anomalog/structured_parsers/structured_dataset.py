from dataclasses import dataclass

from anomalog.anomaly_label_reader import AnomalyLabelLookup
from anomalog.cache import CachePathsConfig
from anomalog.structured_parsers.contracts import (
    UNTEMPLATED_FIELD,
    StructuredSink,
)
from anomalog.template_parsers.templated_dataset import TemplatedDataset, TemplateParser


# TODO: Better way of managing the flow of data through the
# various stages of processing? Maybe a more explicit pipeline definition?
@dataclass(slots=True)
class StructuredDataset:
    sink: StructuredSink
    cache_paths: CachePathsConfig
    anomaly_labels: AnomalyLabelLookup

    def mine_templates_with(self, template_parser: TemplateParser) -> TemplatedDataset:
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
