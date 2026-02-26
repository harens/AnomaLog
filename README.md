## AnomaLog

An orchestration-driven research framework for reproducible log anomaly detection pipelines. Converts raw logs into deterministic, template-mapped sequences ready for controlled detector experiments.

Built on [Prefect](https://www.prefect.io/), AnomaLog emphasises end-to-end reproducibility - from raw log ingestion to model-ready sequences.

### Motivation

Many log anomaly detection implementations focus primarily on modeling techniques while omitting the full preprocessing pipeline. Parsing details are often described but not fully reproducible from code, and experiments frequently rely on preprocessed datasets without documenting raw log handling.

“The same dataset” is not always the same once parsing choices, windowing rules, entity grouping, and leakage controls are considered.

AnomaLog provides a cache-aware, pipeline-first framework that treats log preprocessing as a first-class research artifact. Each stage, from raw ingestion → parsing → template mining → sequencing, is modular and reproducible, rather than one-off scripts with hidden assumptions.

This enables controlled ablation studies, fair model comparisons, and fully repeatable experiments from raw logs. Researchers can focus on modeling choices rather than reverse-engineering preprocessing and experiment glue.

### Key Features

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
