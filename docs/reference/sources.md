# Sources

Sources define how a dataset is materialised locally before parsing starts.

Use them to distinguish between checked-in local data, local archives, and
remote benchmark downloads while keeping the dataset pipeline explicit.

```pycon
>>> from pathlib import Path
>>> from anomalog.sources import LocalDirSource, LocalZipSource, RemoteZipSource
>>> LocalDirSource.name, LocalZipSource.name, RemoteZipSource.name
('local_dir', 'local_zip', 'remote_zip')
>>> LocalDirSource(Path("logs"), raw_logs_relpath=Path("demo.log")).raw_logs_relpath
PosixPath('demo.log')
```

## `anomalog.sources`

::: anomalog.sources
