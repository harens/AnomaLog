# Getting Started

Install AnomaLog, start from a checked-in preset, and turn raw logs into reproducible model inputs.

## :material-download: Install

[![PyPI - Version](https://img.shields.io/pypi/v/anomalog?logo=pypi&logoColor=white&color=blue&style=flat-square)](https://pypi.org/project/anomalog/)
[![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fharens%2FAnomaLog%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/anomalog/)

```bash
pip install anomalog
```

## :material-chart-timeline-variant: Mental model

AnomaLog treats preprocessing as part of the research artifact rather than as
setup code around the model.

The public API keeps the preprocessing stages explicit:

```python
from anomalog import DatasetSpec

dataset = (
    DatasetSpec("...")  # (1)!
    .from_source(...)  # (2)!
    .parse_with(...)  # (3)!
    .label_with(...)  # (4)!
    .template_with(...)  # (5)!
    .build()
)
```

1. Choose a stable dataset name.
2. Decide where the raw logs come from.
3. Parse the dataset-specific log format.
4. Attach anomaly labels when they are not already present inline.
5. Choose the template miner.

`.build()` returns a templated dataset. Sequence construction happens
after that step, and representation happens after sequence construction:

1. build a templated dataset
2. group events into `TemplateSequence` windows
3. choose a representation for the detector family you want to run

That separation is deliberate. Sequence construction decides which events belong together in one example. Representation then decides how that example is encoded for a model.

A few terms used throughout the docs:

- A **template** is a canonical message pattern shared by many concrete log lines.
- A **sequence** is a grouped window of log events that becomes one model input.
- A **representation** is the model-facing form of that sequence, such as an ordered list or a count vector.

For the full stage-by-stage explanation, see [Pipeline Concepts](pipeline-concepts.md).

## :material-rocket-launch-outline: Start from a preset

The easiest starting point is a preset dataset specification from
`anomalog.presets`.

AnomaLog includes ready-made presets for benchmark datasets including [BGL](https://github.com/logpai/loghub/blob/master/BGL/README.md) and [HDFS v1](https://github.com/logpai/loghub/blob/master/HDFS/README.md).

Each preset is an ordinary `DatasetSpec`, so its preprocessing choices remain visible, inspectable, and modifiable:

```python
>>> from anomalog.presets import bgl
>>> from anomalog.parsers import IdentityTemplateParser

>>> bgl.source.url
'https://zenodo.org/records/8196385/files/BGL.zip'
>>> bgl.template_parser.name
'drain3'
# Ablation: disable template mining and use raw log lines directly
>>> ablated_dataset = bgl.template_with(IdentityTemplateParser).build()
```

That matters because presets are not opaque shortcuts. They are checked-in
builder definitions that you can inspect, keep fixed for a baseline, and modify one stage at a time for ablations.

## :material-play-circle-outline: Build a dataset

Build a templated dataset directly from the preset:

```python
from anomalog.presets import bgl

dataset = bgl.build()
```

This materialises the preset pipeline and returns a templated dataset.

If you want explicit control instead of a preset, define a `DatasetSpec` directly:

```python
from pathlib import Path

from anomalog import DatasetSpec
from anomalog.labels import CSVReader
from anomalog.parsers import Drain3Parser, HDFSV1Parser
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
    .template_with(Drain3Parser)
    .build()
)
```

The same fluent builder is used in both cases. Presets simply provide a
checked-in starting `DatasetSpec`.

## :material-timeline-text-outline: Group into sequences

Once `.build()` returns a templated dataset, choose how downstream models
should see the log stream.

For benchmarks such as BGL and HDFS, entity grouping is often the right
starting point:

```python
from anomalog import SplitLabel
from anomalog.presets import bgl

dataset = bgl.build()
sequences = dataset.group_by_entity().with_train_fraction(0.8)

for sequence in sequences:
    if sequence.split_label is SplitLabel.TRAIN:
        print(sequence.window_id, sequence.label, sequence.templates[:3])
```

```text
0 0 [
    "RAS KERNEL INFO instruction cache parity error corrected",
    "RAS KERNEL INFO data cache parity error corrected",
    "RAS KERNEL INFO data cache parity error corrected",
]
```

AnomaLog also supports fixed-size and time-based windows when the research
question is not entity-centric:

```python
fixed_sequences = dataset.group_by_fixed_window(window_size=128, step_size=64)
time_sequences = dataset.group_by_time_window(
    time_span_ms=60_000,
    step_span_ms=30_000,
)
```

All grouping modes produce `TemplateSequence` objects. See
[Sequences](reference/sequences.md) for the full object shape and
[Pipeline Concepts](pipeline-concepts.md#sequence-construction) for grouping
tradeoffs.

## :material-vector-polyline: Choose a representation

`TemplateSequence` is still model-agnostic. The representation layer converts a
sequence into the input shape expected by a detector.

```python
from anomalog.representations import (
    SequentialRepresentation,
    TemplateCountRepresentation,
    TemplatePhraseRepresentation,
)

builder = dataset.group_by_fixed_window(window_size=3).with_train_fraction(0.8)

sequential = builder.represent_with(SequentialRepresentation())
template_counts = builder.represent_with(TemplateCountRepresentation())
template_phrases = builder.represent_with(
    TemplatePhraseRepresentation(phrase_ngram_min=1, phrase_ngram_max=2),
)
```

Use the representation that matches the model family:

- `SequentialRepresentation` for ordered template streams
- `TemplateCountRepresentation` for sparse template counts
- `TemplatePhraseRepresentation` for sparse phrase counts extracted from template text

Custom representations are not limited to template text. They receive the full
`TemplateSequence`, so they can use event timing deltas, parameters, entity
IDs, or split metadata.

For more detail, see [Representations](reference/representations.md) and
[Pipeline Concepts](pipeline-concepts.md#representations).

## :material-sign-direction: What next

- Read [Pipeline Concepts](pipeline-concepts.md) for the full stage-by-stage explanation and reproducibility model
- See [Experiments](experiments.md) for config-driven detector runs and result artifacts
- See [Reference](reference/index.md) for the API pages and module map
- See [Development](development.md) for contributor setup and implementation-facing storage details
