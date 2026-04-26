# Reference

This section is the technical map for AnomaLog.

Use it for two things:

- navigating the public API
- orienting yourself in the codebase before drilling into the generated reference pages

## :material-book-open-page-variant-outline: Reference pages

<div class="grid cards" markdown>

- [Top-level API](top-level.md)
  `DatasetSpec` and `SplitLabel`

- [Presets](presets.md)
  Built-in dataset specifications

- [Parsers](parsers.md)
  Structured parsers, template parsers, sinks, and `TemplatedDataset`

- [Sources](sources.md)
  Dataset source implementations

- [Labels](labels.md)
  Label readers and lookups

- [Sequences](sequences.md)
  Sequence builders and sequence objects

- [Representations](representations.md)
  Model-facing sequence representations and lazy representation views

- [Experiments](experiments.md)
  Config-driven experiment configs, model runners, and result helpers

</div>

## :material-compass-outline: Codebase layout

- `anomalog/__init__.py` exposes the small top-level public API
- `anomalog/dataset.py` contains the fluent `DatasetSpec` builder
- `anomalog/presets.py` defines built-in dataset specifications
- `anomalog/parsers/` contains structured and template parser abstractions
- `anomalog/sources/` contains dataset materialisation logic
- `anomalog/labels.py` contains label readers and lookups
- `anomalog/sequences.py` contains sequence grouping and split behavior
- `anomalog/representations/` contains model-facing sequence representations
- `anomalog/_runtime/` contains internal orchestration code
- `experiments/` contains the repository-local experiment configs, runners, and result helpers

## :material-flask-outline: Experiment layer

The `experiments/` directory is useful for reproducible model experimentation
after preprocessing, but it is separate from the core public API of the
library. See [Experiments](experiments.md) for the module reference and
[`docs/experiments.md`](../experiments.md) for the workflow overview.

!!! note
    The reference pages document symbols. Start with [Getting Started](../getting-started.md) for the onboarding path and [Pipeline Concepts](../pipeline-concepts.md) for the stage-by-stage mental model.
