# AnomaLog

[![Codecov](https://img.shields.io/codecov/c/github/harens/anomalog?style=flat-square&logo=codecov)](https://app.codecov.io/gh/harens/AnomaLog)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/harens/anomalog/ci.yml?style=flat-square&logo=github&label=tests)](https://github.com/harens/AnomaLog/actions/workflows/ci.yml)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/harens/anomalog/deploy-docs.yml?style=flat-square&logo=googledocs&logoColor=white&label=docs)](https://harens.github.io/AnomaLog/)
![GitHub License](https://img.shields.io/github/license/harens/AnomaLog?style=flat-square&color=blue)


An orchestration-driven research framework for reproducible log anomaly detection pipelines. Converts raw logs into deterministic, template-mapped sequences ready for controlled detector experiments.

Built on [Prefect](https://www.prefect.io/), AnomaLog emphasises end-to-end reproducibility from raw log ingestion to model-ready sequences.

## Motivation

Many log anomaly detection implementations focus primarily on modelling techniques while omitting the full preprocessing pipeline. Parsing details are often described but not fully reproducible from code, and experiments frequently rely on preprocessed datasets without documenting raw log handling.

“The same dataset” is not always the same once parsing choices, windowing rules, entity grouping, and leakage controls are considered.

AnomaLog provides a cache-aware, pipeline-first framework that treats log preprocessing as a first-class research artifact. Each stage, from raw ingestion → parsing → template mining → sequencing, is modular and reproducible, rather than one-off scripts with hidden assumptions.

This enables controlled ablation studies, fair model comparisons, and fully repeatable experiments from raw logs. Researchers can focus on modeling choices rather than reverse-engineering preprocessing and experiment glue.

## Key Features

- **Deterministic pipeline execution.**
  Workflow stages are fingerprinted and cached so only modified components are recomputed.

- **Protocol-driven modularity.**
  All preprocessing stages implement explicit protocol interfaces, enabling parsers, template miners (e.g. [Drain3](https://github.com/logpai/Drain3)), and sequencing strategies to be swapped without altering downstream logic.

- **Explicit sequencing strategies.**
  Entity-based, fixed-length, and time-windowed sequences are built with deterministic split controls.

- **Dataset-first workflows.**
  Built-in benchmark presets and custom datasets share the same public interface.

- **Scalable, artifact-first storage.**
  Structured events are persisted in Parquet by default so expensive parsing can be reused.

## Research Usage

Unlike model-centric repositories that assume preprocessed inputs,
AnomaLog makes preprocessing part of the research surface. A typical workflow is:

1) Materialise a templated dataset (raw → structured → templates).
2) Generate deterministic sequences under an explicit split protocol.
3) Plug in any detector that consumes TemplateSequence.

Determinism is a property of the pipeline, not the random number generator. Event ordering is defined by the default dataset backend and preserved through sequencing. This allows for reproducible train/test splits across runs without requiring random seeds.

```python
from anomalog import SplitLabel
from anomalog.presets import bgl

dataset = bgl.build()
sequence_view = dataset.group_by_entity().with_train_fraction(0.2)

for seq in sequence_view:
    if seq.split_label == SplitLabel.TRAIN:
        ...
```

## Custom Dataset Definition

To add a dataset, define a `DatasetSpec` by specifying the source, structured parser, optional label alignment, and template parser. This makes dataset provenance and preprocessing assumptions explicit and versionable.

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

### Built-in presets

```python
from anomalog.presets import bgl, hdfs_v1

bgl_dataset = bgl.build()
hdfs_dataset = hdfs_v1.build()
```

## Preprocessing Ablation Studies

Preprocessing decisions such as the template miner, label alignment, and grouping strategy can be treated as experimental variables rather than hidden implementation details.

```python
from anomalog.parsers import Drain3Parser, IdentityTemplateParser
from anomalog.presets import bgl

drain_dataset = bgl.template_with(Drain3Parser).build()
identity_dataset = bgl.template_with(IdentityTemplateParser).build()
```
