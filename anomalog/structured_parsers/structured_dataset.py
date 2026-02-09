from dataclasses import dataclass

from anomalog.cache import CachePathsConfig
from anomalog.structured_parsers.contracts import StructuredSink
from anomalog.template_parsers.templated_dataset import TemplatedDataset, TemplateParser


@dataclass(slots=True, frozen=True)
class StructuredDataset:
    sink: StructuredSink
    cache_paths: CachePathsConfig

    def mine_templates_with(self, template_parser: TemplateParser) -> TemplatedDataset:
        template_parser.train(self.sink.read_unstructured_free_text())

        return TemplatedDataset(
            sink=self.sink,
            cache_paths=self.cache_paths,
            template_parser=template_parser,
        )
