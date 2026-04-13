---
title: AnomaLog
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# AnomaLog

[![PyPI - Version](https://img.shields.io/pypi/v/anomalog?logo=pypi&logoColor=white&color=blue&style=flat-square)](https://pypi.org/project/anomalog/)
[![Codecov](https://img.shields.io/codecov/c/github/harens/anomalog?style=flat-square&logo=codecov)](https://app.codecov.io/gh/harens/AnomaLog)
[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/harens/anomalog/ci.yml?style=flat-square&logo=github&label=tests)](https://github.com/harens/AnomaLog/actions/workflows/ci.yml)
[![GitHub License](https://img.shields.io/github/license/harens/AnomaLog?style=flat-square&color=blue)](https://github.com/harens/AnomaLog/blob/main/LICENSE)

Reproducible log anomaly detection pipelines, from raw logs to deterministic, template-mapped sequences.

[Get Started](getting-started.md){ .md-button .md-button--primary }
[Pipeline Concepts](pipeline-concepts.md){ .md-button }
[Reference](reference/index.md){ .md-button }

</div>

<div class="grid cards" markdown>

-   :material-database: **End-to-end preprocessing**
    Go from raw logs to parsed, templated, and sequence-ready artifacts in one reproducible pipeline.

-   :material-source-branch: **Deterministic by design**
    Fingerprinted stages, caching, and stable artifact lineage make runs repeatable and easier to compare.

-   :material-puzzle: **Modular pipeline stages**
    Swap parsers, template miners, and sequence builders without rewriting downstream workflows.

-   :material-chart-timeline-variant: **Explicit sequence construction**
    Build entity-based, fixed-length, or time-windowed sequences with clear split and leakage controls.

-   :material-database-search: **Benchmark and custom datasets**
    Use built-in dataset workflows or bring your own data through a consistent schema and labelling model.

-   :material-file-table: **Reusable intermediate artifacts**
    Reuse structured intermediate data across runs instead of rebuilding everything from raw logs.

</div>

## :material-lightbulb-outline: Why AnomaLog exists

Many log anomaly detection results are difficult to compare because the preprocessing pipeline is underspecified. Two experiments may claim to use the same dataset while differing in parsing rules, label alignment, template mining, grouping, or split behavior.

AnomaLog treats those preprocessing decisions as explicit pipeline stages rather than hidden scripts or fixed artifacts. The goal is not only convenience, but to make comparisons, ablations, and reruns more defensible.

## :material-chart-timeline-variant: How it works

AnomaLog structures preprocessing as an explicit, reproducible pipeline:

1. Define a dataset source
2. Parse and template logs
3. Group events into sequences
4. Represent sequences for modeling

```python
from anomalog.parsers import IdentityTemplateParser
from anomalog.presets import bgl
from anomalog.representations import TemplateCountRepresentation

# Deterministic, composable preprocessing pipeline
samples = (
    bgl.template_with(IdentityTemplateParser)
    .build()
    .group_by_entity()
    .with_train_fraction(0.8)
    .represent_with(TemplateCountRepresentation())
)
```

See [Getting Started](getting-started.md) for the onboarding walkthrough and
[Pipeline Concepts](pipeline-concepts.md) for the full mental model.

## :material-sign-direction: Start here

- [Getting Started](getting-started.md) - Install AnomaLog and run your first pipeline
- [Pipeline Concepts](pipeline-concepts.md) - Understand stages, grouping, representations, and reproducibility
- [Experiments](experiments.md) - Run config-driven detector experiments
- [API Reference](reference/index.md) - Browse interfaces, built-ins, and module docs
- [Development](development.md) - Set up the repo and run checks locally
