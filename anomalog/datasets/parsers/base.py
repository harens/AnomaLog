from typing import Protocol, runtime_checkable

from anomalog.datasets.dataset import Dataset


class ParsedDataset: ...


@runtime_checkable
class DatasetParser(Protocol):
    def parse(self, dataset: Dataset) -> ParsedDataset: ...
