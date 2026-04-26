# AnomaLog

[![Codecov](https://img.shields.io/codecov/c/github/harens/anomalog?style=flat-square&logo=codecov)](https://app.codecov.io/gh/harens/AnomaLog)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/harens/anomalog/ci.yml?style=flat-square&logo=github&label=tests)](https://github.com/harens/AnomaLog/actions/workflows/ci.yml)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/harens/anomalog/deploy-docs.yml?style=flat-square&logo=googledocs&logoColor=white&label=docs)](https://harens.github.io/AnomaLog/)
![GitHub License](https://img.shields.io/github/license/harens/AnomaLog?style=flat-square&color=blue)

**AnomaLog turns raw logs into reproducible, model-ready datasets for anomaly detection.**

It is designed for research workflows where preprocessing is not incidental, but part of the experimental artifact. Parsing, template mining, labeling, and sequence construction are made explicit, composable, and reproducible.

Benchmark results in log anomaly detection often depend on hidden preprocessing decisions. AnomaLog surfaces those decisions and makes them first-class, enabling fair comparison and repeatable experiments.

Some typical use cases include:

- Comparing anomaly detectors under controlled preprocessing assumptions
- Running ablations over parsers or template miners
- Reproducing published benchmarks from raw logs

### ⚡ 10-second example

```python
from anomalog.presets import bgl
from anomalog.representations import TemplatePhraseRepresentation

samples = (
    bgl.build()
    .group_by_entity()
    .with_train_fraction(0.8)
    .represent_with(TemplatePhraseRepresentation())
)
```

This constructs model-ready features from raw logs with a fully specified, reproducible preprocessing pipeline.

## Installation

[![PyPI - Version](https://img.shields.io/pypi/v/anomalog?logo=pypi&logoColor=white&color=blue&style=flat-square)](https://pypi.org/project/anomalog/)
[![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fharens%2FAnomaLog%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/anomalog/)

```bash
pip install anomalog
```

## Pipeline at a glance

AnomaLog models preprocessing as a deterministic pipeline:

1. **Source** - raw log ingestion and dataset sourcing
2. **Parsing** - structured parsing into typed log records
3. **Templating** - template mining and assignment
4. **Sequencing** - grouping logs into windows
5. **Representation** - model-ready feature extraction

The public API is centered on the fluent builder:

```python
dataset = (
    DatasetSpec(...)
    .from_source(...)
    .parse_with(...)
    .label_with(...)
    .template_with(...)
    .build()
)
```

That produces a templated dataset, which can then be grouped into sequences and converted into model-ready representations.

## Quickstart

Our presets include popular log anomaly datasets like BGL and HDFS v1.

A typical workflow is:

1. Load a preset dataset (e.g. BGL).
2. Build structured logs with parsing and templating.
3. Group logs into labeled sequences.
4. Convert sequences into model-ready features.

```python
>>> from anomalog import SplitLabel
>>> from anomalog.parsers import IdentityTemplateParser
>>> from anomalog.presets import bgl
>>> from anomalog.representations import TemplatePhraseRepresentation

>>> dataset = bgl.build()
# Presets are ordinary DatasetSpec objects, so preprocessing choices stay visible
>>> bgl.template_parser.name
'drain3'
# Deterministically group logs into sequences with explicit train/test semantics
>>> sequences = dataset.group_by_entity().with_train_fraction(0.8)

# Each event stores (template, parameters, timing delta)
>>> next(iter(sequences))
TemplateSequence(
    events=[
        ('RAS KERNEL FATAL <:*:> <:*:> <:*:>', ['data', 'storage', 'interrupt'], None),
        ('RAS KERNEL FATAL <:*:> <:*:> <:*:>', ['instruction', 'address:', '0x00004ed8'], 523407),
        ...
    ],
    label=1,
    entity_ids=['R00-M0-N0-C:J08-U01'],
    split_label=<SplitLabel.TRAIN: 'train'>
)

# Convert sequences into n-gram features for modeling
>>> train_samples = sequences.represent_with(
    TemplatePhraseRepresentation(phrase_ngram_min=1, phrase_ngram_max=2),
)
>>> next(sample for sample in train_samples if sample.split_label is SplitLabel.TRAIN)
SequenceSample(data=Counter({'ras': 49, 'kernel': 49, 'ras kernel': 49, ...})
               label=1,
               entity_ids=['R00-M0-N0-C:J08-U01'],
               split_label=<SplitLabel.TRAIN: 'train'>,
               window_id=0)

# Ablation: disable template mining and use raw log lines directly
>>> ablated_dataset = bgl.template_with(IdentityTemplateParser).build()
```

Built-in presets are ordinary `DatasetSpec` objects, so the exact source,
parser, label, and template-mining choices stay visible in code and can be
modified for ablations instead of being hidden behind preprocessed artifacts.

You can also define a custom dataset:

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

## Documentation

The full documentation is organised by task:

- [Getting started](https://harens.github.io/AnomaLog/getting-started/) for the end-to-end workflow, grouping choices, representation stage, and split semantics
- [Experiments](https://harens.github.io/AnomaLog/experiments/) for config-driven detector runs and recorded artifacts
- [Reference](https://harens.github.io/AnomaLog/reference/) for the codebase map and API pages

## Experiments

The repository also includes a config-driven experiment layer under
`experiments/` for model experimentation on top of AnomaLog preprocessing.

Built-in experiment detectors include Template Frequency, [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier),
[River](https://riverml.xyz/), and a scoped [DeepLog](https://dl.acm.org/doi/10.1145/3133956.3134015) and [DeepCASE](https://ieeexplore.ieee.org/abstract/document/9833671) reimplementation.

```bash
uv run python -m experiments.runners.run_experiment \
  --config experiments/configs/runs/bgl_template_frequency.toml
```

Experiment runs reuse AnomaLog's dataset-side caches and write deterministic
result directories, but detector training and test scoring are intentionally
rerun for new config fingerprints.

See [`experiments/README.md`](experiments/README.md) for the experiment layout and artifact format.

## Development

Contributor setup and local commands are documented in [Development](https://harens.github.io/AnomaLog/development/).
