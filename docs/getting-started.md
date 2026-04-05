# Getting Started

Install AnomaLog, build a dataset, and turn raw logs into sequences you can use in experiments.

## :material-download: Install

[![PyPI - Version](https://img.shields.io/pypi/v/anomalog?logo=pypi&logoColor=white&color=blue&style=flat-square)](https://pypi.org/project/anomalog/)
[![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fharens%2FAnomaLog%2Frefs%2Fheads%2Fmain%2Fpyproject.toml&style=flat-square&logo=python&logoColor=white)](https://pypi.org/project/anomalog/)


```bash
pip install anomalog
```

## :material-chart-timeline-variant: Mental model

AnomaLog is built around a reproducible preprocessing pipeline:

1. Source raw logs
2. Parse them into structured records
3. Store the structured data
4. Attach anomaly labels
5. Mine message templates
6. Build sequences for downstream models

In the public API, the first five stages are configured with `DatasetSpec`, and the sixth stage happens on the built dataset view:

```python
dataset = (
    DatasetSpec("...")  # (1)!
    .from_source(...)  # (2)!
    .parse_with(...)  # (3)!
    .label_with(...)  # (4)!
    .template_with(...)  # (5)!
    .build()
)
```

1. Give the pipeline a stable dataset name.
2. Choose where the raw data comes from.
3. Choose the parser for that specific dataset format.
4. Provide anomaly labels if they are not already present inline.
5. Choose the template miner.

The return value of `.build()` is a templated dataset view. You then group it into sequences with methods such as `group_by_entity()`, `group_by_fixed_window(...)`, and `group_by_time_window(...)`.

## :material-rocket-launch-outline: First dataset: use a built-in preset

The easiest starting point is a preset dataset specification from `anomalog.presets`.

AnomaLog includes ready-made presets for popular benchmark datasets, including:

- `bgl`
- `hdfs_v1`

```python
from anomalog.presets import bgl

spec = bgl
dataset = spec.build()
```

`dataset` is a templated dataset view. You can turn it into sequences immediately:

```python
from anomalog import SplitLabel
from anomalog.presets import bgl

dataset = bgl.build()
sequences = dataset.group_by_entity().with_train_fraction(0.8)

for sequence in sequences:
    if sequence.split_label is SplitLabel.TRAIN:
        print(sequence.window_id, sequence.label, sequence.templates[:3])
```

Because `bgl` is itself a checked-in `DatasetSpec`, the source, parser, labels, and template miner choices are explicit and versioned in code.

If you want to inspect or modify that configuration, start from the preset spec itself:

```python
from anomalog.parsers import IdentityTemplateParser
from anomalog.presets import bgl

identity_dataset = bgl.template_with(IdentityTemplateParser).build()
```

See [Sequences](reference/sequences.md) for the full `TemplateSequence` shape.

## :material-play-circle-outline: A custom dataset, step by step

Use `DatasetSpec` when you want explicit control over sourcing, parsing, labels, and template mining.

```python
from pathlib import Path

from anomalog import DatasetSpec
from anomalog.labels import CSVReader
from anomalog.parsers import Drain3Parser, HDFSV1Parser
from anomalog.sources import LocalZipSource

dataset = (
    DatasetSpec("my-hdfs")  # (1)!
    .from_source(  # (2)!
        LocalZipSource(
            Path("HDFS_v1.zip"),
            raw_logs_relpath=Path("HDFS.log"),
        ),
    )
    .parse_with(HDFSV1Parser())  # (3)!
    .label_with(  # (4)!
        CSVReader(
            relative_path=Path("preprocessed/anomaly_label.csv"),
            entity_column="BlockId",
            label_column="Label",
        ),
    )
    .template_with(Drain3Parser)  # (5)!
    .build()
)
```

1. Name the dataset pipeline you want to build.
2. Choose how the raw logs are materialised locally.
3. Pass a structured parser instance for the dataset format you are working with.
4. Attach anomaly labels when they are not already embedded in the structured data.
5. Pass a template parser class. [`Drain3Parser`](reference/parsers.md#anomalog.parsers.Drain3Parser) is the default template miner.

## :material-stairs: What each stage means

### :material-folder-download-outline: Source raw logs

`from_source(...)` tells AnomaLog where the dataset comes from.

Typical choices are:

- [`RemoteZipSource`](reference/sources.md#anomalog.sources.RemoteZipSource) for a benchmark dataset downloaded from a URL
- [`LocalZipSource`](reference/sources.md#anomalog.sources.LocalZipSource) for a local archive
- [`LocalDirSource`](reference/sources.md#anomalog.sources.LocalDirSource) for a directory that already contains the logs

### :material-file-code-outline: Parse a specific log format

`parse_with(...)` is where you tell AnomaLog how to interpret a specific dataset format.

A structured parser is responsible for extracting components such as:

- timestamp
- entity identifier
- message text before templating
- inline anomaly label when the format contains one

This is why `parse_with(...)` takes a parser instance such as `HDFSV1Parser()` or `BGLParser()`: each parser understands a particular log format.

For example, an HDFS parser turns a raw line such as:

```text
081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-160 src: /10.0.0.1:54106 dest: /10.0.0.2:50010
```

into structured components such as:

- timestamp: `081109 203518` converted into Unix milliseconds
- entity ID: `blk_-160`
- message text before templating: `INFO dfs.DataNode$DataXceiver: Receiving block blk_-160 src: /10.0.0.1:54106 dest: /10.0.0.2:50010`

That structured representation is what the later template and sequence stages operate on.

### :material-database-outline: Store structured data in a sink

After parsing, AnomaLog writes the structured records to a sink. By default, that sink is [`ParquetStructuredSink`](reference/parsers.md#anomalog.parsers.ParquetStructuredSink).

The sink is the persisted representation of the structured dataset. Keeping this stage explicit means later steps can reuse the structured data instead of reparsing raw logs.

[Parquet](https://parquet.apache.org/) is used by default because it is a compact, columnar on-disk format that works well for repeated reads over structured records.

If you want to choose the sink explicitly, use `store_with(...)`. It takes a sink class:

```python
.store_with(ParquetStructuredSink)
```

For example:

```python
from anomalog.parsers import HDFSV1Parser, ParquetStructuredSink

spec = (
    DatasetSpec("my-hdfs")
    .from_source(...)
    .parse_with(HDFSV1Parser())
    .store_with(ParquetStructuredSink)
)
```

### :material-label-outline: Attach anomaly labels

Every dataset build needs anomaly labels from one of two places:

- inline labels emitted by the parser and stored in the structured sink
- an explicit reader such as [`CSVReader`](reference/labels.md#anomalog.labels.CSVReader)

If a dataset has no inline labels and no label reader, `.build()` fails.

The distinction is important:

- [`InlineReader`](reference/labels.md#anomalog.labels.InlineReader) can expose both per-line and per-group labels from the structured sink
- [`CSVReader`](reference/labels.md#anomalog.labels.CSVReader) currently provides group or entity-level labels only, not per-line labels

Use `CSVReader` when the anomaly annotations are stored separately from the raw logs, for example in a dataset-provided CSV.

### :material-shape-outline: Mine templates

A template is the canonical message pattern behind many concrete log lines.

For example, these two lines:

- `Received block blk_123 from node_7`
- `Received block blk_987 from node_3`

share the same template:

- `Received block <*> from <*>`

Template mining is the step that maps many concrete messages onto those shared patterns and extracts the variable parts as parameters.

`template_with(...)` chooses the template miner. It takes a class, not an instance.

[`Drain3Parser`](reference/parsers.md#anomalog.parsers.Drain3Parser) is the default. [Drain3](https://github.com/logpai/Drain3) is a streaming implementation built on the original Drain approach to log template mining, and AnomaLog uses it as the default template parser for turning raw messages into reusable patterns.

## :material-timeline-text-outline: Build sequences

Once `.build()` returns a templated dataset, you choose how to group events into downstream sequences.

```python
entity_sequences = dataset.group_by_entity()
fixed_sequences = dataset.group_by_fixed_window(window_size=128, step_size=64)
time_sequences = dataset.group_by_time_window(time_span_ms=60_000, step_span_ms=30_000)
```

Those grouping methods return a `SequenceBuilder`, which yields `TemplateSequence` objects.

For entity-based grouping, you can also restrict the training split to normal entities only:

```python
trainable = (
    dataset.group_by_entity()
    .with_train_fraction(0.8)
    .with_train_on_normal_entities_only()
)
```

## :material-source-branch: Reproducibility and caching

AnomaLog is designed so that the dataset pipeline itself is the reproducible artifact.

In practice, that means:

- the dataset definition lives in a `DatasetSpec`
- raw data materialisation is tied to the configured source
- structured data is persisted and reused instead of reparsed every time
- template mining results are cached and reused when the inputs have not changed
- sequence grouping is deterministic for a fixed built dataset and grouping configuration
- train/test splits come from deterministic ordering and grouping rules rather than from a random number generator

There is no seed you need to set just to make the preprocessing split reproducible. The ordering rules in the sink and sequence builder define the split behavior.

[Prefect](https://www.prefect.io/) is used internally for task materialisation and cache reuse, but normal user code should think in terms of dataset stages rather than Prefect mechanics.

For the storage layout and caching internals, see [Development](development.md).

## :material-sign-direction: Next steps

- See [Reference](reference/index.md) for the module map and API pages
- See [Development](development.md) for storage and caching internals
