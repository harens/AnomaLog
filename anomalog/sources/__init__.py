"""Dataset source abstractions for fetching log data."""

from anomalog.sources.deeplog_preprocessed import PostProcessedSource

from .contracts import DatasetSource
from .local import LocalDirSource, LocalZipSource
from .remote_zip import RemoteZipSource

__all__ = [
    "DatasetSource",
    "LocalDirSource",
    "LocalZipSource",
    "PostProcessedSource",
    "RemoteZipSource",
]
