# Labels

Labels connect anomaly annotations to the parsed dataset.

This page covers the built-in label readers and lookup helpers used to expose
line-level or group-level anomaly labels during dataset building and sequence
construction.

```pycon
>>> from pathlib import Path
>>> from anomalog.labels import CSVReader
>>> reader = CSVReader(relative_path=Path("labels.csv"))
>>> reader.entity_column, reader.label_column
('entity_id', 'anomalous')
>>> reader.with_context(dataset_root=Path("."), sink=None).dataset_root
PosixPath('.')
```

## `anomalog.labels`

::: anomalog.labels
