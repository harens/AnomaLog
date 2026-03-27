---
title: AnomaLog
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# AnomaLog

[![Codecov](https://img.shields.io/codecov/c/github/harens/anomalog?style=flat-square&logo=codecov)](https://app.codecov.io/gh/harens/AnomaLog)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/harens/anomalog/ci.yml?style=flat-square&logo=github&label=tests)](https://github.com/harens/AnomaLog/actions/workflows/ci.yml)
[![GitHub License](https://img.shields.io/github/license/harens/AnomaLog?style=flat-square&color=blue)](https://github.com/harens/AnomaLog/blob/main/LICENSE)

Reproducible log anomaly detection pipelines, from raw logs to deterministic, template-mapped sequences

[Get started](quickstart.md){ .md-button .md-button--primary }
[About](about.md){ .md-button }

</div>

<div class="grid cards" markdown>

-   :material-database: **Raw logs to model-ready data**
    Go from ingestion and parsing to template-mapped, sequence-ready artifacts in one reproducible pipeline.

-   :material-source-branch: **Deterministic by design**
    Fingerprinted stages, caching, and stable artifact lineage make runs repeatable and easier to compare.

-   :material-puzzle: **Modular pipeline stages**
    Swap parsers, template miners, and sequence builders without rewriting downstream workflows.

-   :material-chart-timeline-variant: **Explicit sequence construction**
    Build entity-based, fixed-length, or time-windowed sequences with clear split and leakage controls.

-   :material-database-search: **Benchmark and custom datasets**
    Use built-in dataset workflows or bring your own data through a consistent schema and labelling model.

-   :material-file-table: **Reusable intermediate artifacts**
    Persist structured events as reusable artifacts, with Parquet as the default backend and pluggable storage abstractions.

</div>
