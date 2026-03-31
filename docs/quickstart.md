# Quickstart

## Built-in dataset

```python
from anomalog.presets import bgl

dataset = bgl.build()
```

## Custom dataset

```python
from pathlib import Path

from anomalog import DatasetSpec
from anomalog.labels import CSVReader
from anomalog.parsers import HDFSV1Parser
from anomalog.sources import LocalZipSource

dataset = (
    DatasetSpec("my-hdfs")
    .from_source(LocalZipSource(Path("HDFS_v1.zip"), raw_logs_relpath=Path("HDFS.log")))
    .parse_with(HDFSV1Parser())
    .label_with(
        CSVReader(
            relative_path=Path("preprocessed/anomaly_label.csv"),
            entity_column="BlockId",
            label_column="Label",
        ),
    )
    .build()
)
```

## Group sequences

```python
sequences = (
    dataset.group_by_entity()
    .with_train_fraction(0.8)
)
```
