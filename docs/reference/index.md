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

</div>

## :material-compass-outline: Codebase layout

- `anomalog/__init__.py` exposes the small top-level public API
- `anomalog/dataset.py` contains the fluent `DatasetSpec` builder
- `anomalog/presets.py` defines built-in dataset specifications
- `anomalog/parsers/` contains structured and template parser abstractions
- `anomalog/sources/` contains dataset materialisation logic
- `anomalog/labels.py` contains label readers and lookups
- `anomalog/sequences.py` contains sequence grouping and split behavior
- `anomalog/_runtime/` contains internal orchestration code

## :material-flask-outline: Experiment layer

The `experiments/` directory is useful for reproducible detector runs, but it is separate from the core public API of the library.

!!! note
    The reference pages document symbols. For the workflow and mental model, start with [Getting Started](../getting-started.md).
