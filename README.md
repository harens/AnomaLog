# AnomaLog

[![Codecov](https://img.shields.io/codecov/c/github/harens/anomalog?style=flat-square&logo=codecov)](https://app.codecov.io/gh/harens/AnomaLog)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/harens/anomalog/ci.yml?style=flat-square&logo=github&label=tests)](https://github.com/harens/AnomaLog/actions/workflows/ci.yml)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/harens/anomalog/deploy-docs.yml?style=flat-square&logo=googledocs&logoColor=white&label=docs)](https://harens.github.io/AnomaLog/)
![GitHub License](https://img.shields.io/github/license/harens/AnomaLog?style=flat-square&color=blue)

AnomaLog is a Python library for researchers working on log anomaly detection.
It turns raw logs into deterministic, template-mapped sequences so preprocessing choices can be made explicit, compared fairly, and reproduced from raw data onward.

The library exists because benchmark results in log anomaly detection often depend on hidden preprocessing decisions: how logs are parsed, how labels are aligned, how templates are mined, and how events are grouped into sequences. AnomaLog treats that preprocessing pipeline as part of the research artifact rather than as disposable glue code.

## Pipeline at a glance

AnomaLog models preprocessing as a stable pipeline:

1. Raw log ingestion and dataset sourcing
2. Structured parsing into typed log records
3. Deterministic structured storage
4. Template mining and template assignment
5. Sequence building into model-ready outputs

The public API is centered on the fluent builder:

```python
DatasetSpec(...).from_source(...).parse_with(...).label_with(...).template_with(...).build()
```

That produces a templated dataset view which can then be grouped into sequences with explicit train/test semantics.

## Installation

[![PyPI - Version](https://img.shields.io/pypi/v/anomalog?logo=pypi&logoColor=white&color=blue&style=flat-square)](https://pypi.org/project/anomalog/)
[![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fharens%2FAnomaLog%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/anomalog/)

```bash
pip install anomalog
```

## Quickstart

Use a built-in preset. AnomaLog currently ships ready-made presets for popular benchmark datasets including `BGL` and `HDFS_v1`.

```python
from anomalog.presets import bgl

dataset = bgl.build()
sequences = dataset.group_by_entity().with_train_fraction(0.8)

for sequence in sequences:
    print(sequence.window_id, sequence.split_label, sequence.label)
```

Define a custom dataset:

```python
from pathlib import Path

from anomalog import DatasetSpec
from anomalog.labels import CSVReader
from anomalog.parsers import HDFSV1Parser
from anomalog.sources import LocalZipSource

dataset = (
    DatasetSpec("my-hdfs")
    .from_source(
        LocalZipSource(
            Path("HDFS_v1.zip"),
            raw_logs_relpath=Path("HDFS.log"),
        ),
    )
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

Swap preprocessing stages explicitly:

```python
from anomalog.parsers import Drain3Parser, IdentityTemplateParser
from anomalog.presets import bgl

drain_dataset = bgl.template_with(Drain3Parser).build()
identity_dataset = bgl.template_with(IdentityTemplateParser).build()
```

## What AnomaLog enables

AnomaLog is designed primarily for:

- researchers comparing anomaly detectors under controlled preprocessing assumptions
- researchers running preprocessing ablations over parsers, template miners, and grouping strategies
- users who want to materialise benchmark or custom datasets from raw logs without building one-off pipelines
- contributors who need a clear separation between public API, runtime orchestration, and dataset views

## Documentation

The full documentation is organized by task:

- [Getting started](https://harens.github.io/AnomaLog/getting-started/) for the end-to-end workflow, pipeline stages, reproducibility model, and first examples
- [Reference](https://harens.github.io/AnomaLog/reference/) for the codebase map and mkdocstrings-backed API pages

## Experiments

The repository also includes a config-driven experiment layer under `experiments/` for reproducible detector runs and preprocessing ablations.

```bash
python -m experiments.runners.run_experiment \
  --config experiments/configs/runs/bgl_template_frequency.toml
```

See `experiments/README.md` for the experiment layout and artifact format.

## Development

Contributor setup and local commands are documented in [Development](https://harens.github.io/AnomaLog/development/).
