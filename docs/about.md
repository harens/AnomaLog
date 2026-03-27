# About AnomaLog

!!! abstract "Overview"
    **AnomaLog** is a framework for reproducible log anomaly detection pipelines, from raw logs to model-ready sequences.

    It treats preprocessing as a **first-class research artifact**, rather than an implicit or one-off step.

!!! note "Out of scope"
    It does not prescribe specific models, but provides the foundation for evaluating them rigorously.

---

## Why it exists

Most work in log anomaly detection focuses on modelling, while preprocessing is often:

- only partially described and not reproducible
- implemented as ad-hoc scripts
- replaced entirely by preprocessed datasets

!!! warning "The hidden problem"
    Two experiments using “the same dataset” may differ in preprocessing decisions that are rarely standardised or made explicit:

    - parsing logic
    - template mining configuration
    - sequence construction rules
    - train/test splits
    - leakage controls

    As a result, observed performance differences are often confounded by uncontrolled preprocessing variation rather than modelling improvements.

---
## Pipeline

<div class="pipeline" markdown>

<span class="pipeline-step">
Raw logs
:material-information-outline:{ title="Standardised ingestion ensures datasets are acquired and loaded consistently" }
</span>

<span class="pipeline-arrow">→</span>

<span class="pipeline-step">
Parsing
:material-information-outline:{ title="Convert raw logs into structured, ordered events" }
</span>

<span class="pipeline-arrow">→</span>

<span class="pipeline-step">
Templates
:material-information-outline:{ title="Map messages to reusable templates while preserving structure" }
</span>

<span class="pipeline-arrow">→</span>

<span class="pipeline-step">
Sequencing
:material-information-outline:{ title="Events are grouped into sequences with explicit splitting rules, including entity-based, fixed-length, and time-windowed strategies with clear control over leakage." }
</span>

<span class="pipeline-arrow">→</span>

<span class="pipeline-step">
Artifacts
:material-information-outline:{ title="Produce stable artifacts for downstream experiments" }
</span>

</div>
Each stage produces a stable, reusable artifact. This makes preprocessing explicit and modular, allowing experiments to be defined in terms of *which stage changed*, rather than implicitly altering the entire pipeline.

---

## What this enables

<div class="grid cards" markdown>

-   **Stage-level ablations**
    Modify individual pipeline components (e.g. parsing or sequencing) and isolate their effect on model performance.

-   **Faithful comparisons**
    Ensure performance differences reflect modelling choices rather than hidden preprocessing variation.

-   **Deterministic re-execution**
    Reproduce experiments end-to-end from raw logs with consistent ordering, transformations, and splits.

-   **Artifact reuse**
    Avoid recomputation by reusing persisted intermediate outputs across experiments.

</div>
---

## Takeaway

AnomaLog shifts log anomaly detection from model-centric experimentation to also include **pipeline-centric experimentation**.

By making preprocessing explicit, versionable, and reproducible, it enables controlled comparisons and more reliable conclusions.
