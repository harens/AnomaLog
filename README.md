# AnomaLog

[![Codecov](https://img.shields.io/codecov/c/github/harens/anomalog?style=flat-square&logo=codecov)](https://app.codecov.io/gh/harens/AnomaLog)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/harens/anomalog/ci.yml?style=flat-square&logo=github&label=tests)](https://github.com/harens/AnomaLog/actions/workflows/ci.yml)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/harens/anomalog/deploy-docs.yml?style=flat-square&logo=googledocs&logoColor=white&label=docs)](https://harens.github.io/AnomaLog/)
![GitHub License](https://img.shields.io/github/license/harens/AnomaLog?style=flat-square&color=blue)


An orchestration-driven research framework for reproducible log anomaly detection pipelines. Converts raw logs into deterministic, template-mapped sequences ready for controlled detector experiments.

Built on [Prefect](https://www.prefect.io/), AnomaLog emphasises end-to-end reproducibility - from raw log ingestion to model-ready sequences.

## Motivation

Many log anomaly detection implementations focus primarily on modeling techniques while omitting the full preprocessing pipeline. Parsing details are often described but not fully reproducible from code, and experiments frequently rely on preprocessed datasets without documenting raw log handling.

“The same dataset” is not always the same once parsing choices, windowing rules, entity grouping, and leakage controls are considered.

AnomaLog provides a cache-aware, pipeline-first framework that treats log preprocessing as a first-class research artifact. Each stage, from raw ingestion → parsing → template mining → sequencing, is modular and reproducible, rather than one-off scripts with hidden assumptions.

This enables controlled ablation studies, fair model comparisons, and fully repeatable experiments from raw logs. Researchers can focus on modeling choices rather than reverse-engineering preprocessing and experiment glue.

## Key Features

- **Deterministic pipeline execution.**
  Workflow stages are fingerprinted and cached, ensuring that only modified components are recomputed. This enables rapid iteration while preserving experiment traceability and artifact lineage.

- **Protocol-driven modularity.**
  All preprocessing stages implement explicit protocol interfaces, enabling parsers, template miners (e.g. [Drain3](https://github.com/logpai/Drain3)), and sequencing strategies to be swapped without altering downstream logic.

- **Explicit sequencing strategies.**
  Supports entity-based, fixed-length, and time-windowed sequences with deterministic train/test controls, including “train-on-normal-entities-only” protocols.

- **Dataset-first workflows.**
  Built-in flows for common log anomaly benchmarks (MD5-verified Zenodo mirrors) with a consistent schema for custom datasets and inline or external labels.

- **Scalable, artifact-first storage.**
  Structured events are persisted in columnar [Parquet](https://parquet.apache.org/) format, decoupling expensive parsing from downstream modeling. Entity-level bucketing bounds peak memory and enables streaming-scale preprocessing of large log corpora.

## Research Usage

Unlike model-centric repositories that assume preprocessed inputs,
AnomaLog makes preprocessing part of the research surface. A typical workflow is:

1) Materialise a templated dataset (raw → structured → templates).
2) Generate deterministic sequences under an explicit split protocol.
3) Plug in any detector that consumes TemplateSequence.

Determinism is a property of the pipeline, not the random number generator. Event ordering is defined by the default dataset backend and preserved through sequencing. This allows for reproducible train/test splits across runs without requiring random seeds.

```python
from anomalog.datasets import build_bgl_dataset
from anomalog.sequences import SplitLabel

dataset = build_bgl_dataset()
sequence_view = dataset.group_by_entity().split_train_fraction(0.2)

for seq in sequence_view:
    if seq.split_label == SplitLabel.TRAIN:
        # seq.events: [(template: str, parameters: list[str], dt_prev_ms: int | None), ...]
        # seq.counts: Counter[str] of template IDs
        # seq.label: int ground-truth label
        # seq.entity_ids: list[str] entities present in the window
        # seq.window_id: stable window identifier
        ...
```

### Custom Dataset Definition

To add a dataset, define a `RawDataset` by specifying (i) the source, (ii) a structured parser, and (iii) optional label alignment. This makes dataset provenance and preprocessing assumptions explicit and versionable.

<!---
TODO: Make import paths nicer
-->

```python
# Below BGL definition provided by default in anomalog.datasets

from pathlib import Path

from anomalog.sources import RemoteZipSource
from anomalog.sources.raw_dataset import RawDataset
from anomalog.structured_parsers.parsers import BGLParser

# BGL definition below provided and importable from anomalog.dataset
bgl = RawDataset(
    dataset_name="BGL",
    # Swap custom dataset sources from local ZIPs/dirs or online.
    source=RemoteZipSource(
        url="https://zenodo.org/records/8196385/files/BGL.zip",
        md5_checksum="4452953c470f2d95fcb32d5f6e733f7a",
    ),
    # Regex parser for an individual line
    structured_parser=BGLParser(),
    # Optional: align anomaly labels by entity/session ID.
    # anomaly_label_reader=CSVReader(
    #     relative_path=Path("preprocessed/anomaly_label.csv"),
    #     entity_column="Node",
    #     label_column="Label",
    # ),
)
```

### Preprocessing Ablation Studies

Preprocessing decisions (e.g. template miner, windowing strategy, label alignment) can be treated as experimental variables rather than hidden implementation details. This allows controlled ablation studies where only a single preprocessing component is changed while the remainder of the pipeline remains identical.

<!---
TODO: Actually show Ablation
-->

```python
# Below BGL flow provided by default in anomalog.datasets

from prefect import flow

from anomalog.datasets import bgl  # Or use the custom BGL defined above
from anomalog.template_parsers import Drain3Parser

# BGL builder below provided and importable from anomalog.dataset
@flow
def build_bgl_dataset() -> TemplatedDataset:
    # Each stage can optionally take a parameter for a different pluggable implementation
    return (
        bgl.fetch_if_needed()
        .extract_structured_components()
        .mine_templates_with(Drain3Parser("BGL"))
    )
```
