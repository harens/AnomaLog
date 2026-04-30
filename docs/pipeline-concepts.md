# Pipeline Concepts

This page expands on the pipeline introduced in [Getting Started](getting-started.md).

Use it when you want the full mental model for staged preprocessing, sequence
construction, representations, and reproducibility.

## :material-layers-outline: Mental model

AnomaLog is easiest to understand if you treat preprocessing as part of the
research artifact rather than as setup code around the model.

The library makes each preprocessing stage explicit and pluggable, so you can
swap one stage at a time for ablation studies:

1. Source raw logs
2. Parse them into structured records
3. Store the structured data
4. Attach anomaly labels
5. Mine message templates
6. Build sequences for downstream models
7. Represent those sequences for a detector or learner

In the public API, the first five stages are configured with `DatasetSpec`, and
sequence construction happens on the built dataset view:

```python
from anomalog import DatasetSpec

dataset = (
    DatasetSpec("...")
    .from_source(...)
    .parse_with(...)
    .label_with(...)
    .template_with(...)
    .build()
)
```

The return value of `.build()` is a templated dataset view. You then decide how
to group events into windows such as entities or sliding windows, and finally
choose a representation that matches the detector family you want to run.

The important design point is that each stage is a separate choice. You can
change the source, parser, label reader, template miner, grouping strategy, or
representation without rewriting the rest of the pipeline.

## :material-stairs: Dataset stages

### :material-folder-download-outline: Source raw logs

`from_source(...)` tells AnomaLog where the dataset comes from.

Typical choices are:

- [`RemoteZipSource`](reference/sources.md#anomalog.sources.RemoteZipSource) for a benchmark dataset downloaded from a URL
- [`LocalZipSource`](reference/sources.md#anomalog.sources.LocalZipSource) for a local archive
- [`LocalDirSource`](reference/sources.md#anomalog.sources.LocalDirSource) for a directory that already contains the logs

### :material-file-code-outline: Parse a specific log format

`parse_with(...)` is where you tell AnomaLog how to interpret a specific log
format.

A structured parser is responsible for extracting components such as:

- timestamp
- entity identifier
- message text before templating
- inline anomaly label when the format contains one

This is why `parse_with(...)` takes a parser instance such as `HDFSV1Parser()`
or `BGLParser()`: each parser understands a particular log format.

For example, an HDFS parser turns a raw line such as:

```text
081109 203518 143 INFO dfs.DataNode$DataXceiver: Receiving block blk_-160 src: /10.0.0.1:54106 dest: /10.0.0.2:50010
```

into structured components such as:

- timestamp: `081109 203518` converted into Unix milliseconds
- entity ID: `blk_-160`
- message text before templating: `INFO dfs.DataNode$DataXceiver: Receiving block blk_-160 src: /10.0.0.1:54106 dest: /10.0.0.2:50010`

That structured representation is what the later template and sequence stages
operate on.

### :material-database-outline: Store structured data in a sink

After parsing, AnomaLog persists the parsed records in a structured sink so later stages can reuse them without reparsing raw logs. By default,
that sink is [`ParquetStructuredSink`](reference/parsers.md#anomalog.parsers.ParquetStructuredSink).

The sink is the persisted representation of the structured dataset. Keeping
this stage explicit means later steps can reuse the structured data instead of
reparsing raw logs.

[Parquet](https://parquet.apache.org/) is used by default because it is a
compact, columnar on-disk format that works well for repeated reads over
structured records.

If you want to choose the sink explicitly, use `store_with(...)`. It takes a
sink class:

```python
.store_with(ParquetStructuredSink)
```

For example:

```python
from anomalog import DatasetSpec
from anomalog.parsers import HDFSV1Parser, ParquetStructuredSink

spec = (
    DatasetSpec("my-hdfs")
    .from_source(...)
    .parse_with(HDFSV1Parser())
    .store_with(ParquetStructuredSink)
)
```

### :material-label-outline: Attach anomaly labels

Every dataset build must obtain anomaly labels from one of two places:

- inline labels emitted by the parser and stored in the structured sink
- an explicit reader such as [`CSVReader`](reference/labels.md#anomalog.labels.CSVReader)

If a dataset has no inline labels and no label reader, `.build()` fails.

The distinction is important:

- [`InlineReader`](reference/labels.md#anomalog.labels.InlineReader) can expose both per-line and per-group labels from the structured sink
- [`CSVReader`](reference/labels.md#anomalog.labels.CSVReader) currently provides group or entity-level labels only, not per-line labels

Use `CSVReader` when the anomaly annotations are stored separately from the raw
logs, for example in a dataset-provided CSV.

### :material-shape-outline: Mine templates

A template is the canonical message pattern behind many concrete log lines.

For example, these two lines:

- `Received block blk_123 from node_7`
- `Received block blk_987 from node_3`

share the same template:

- `Received block <*> from <*>`

Template mining is the step that maps many concrete messages onto those shared
patterns and extracts the variable parts as parameters.

`template_with(...)` chooses the template miner. It takes a class, not an
instance.

[`Drain3Parser`](reference/parsers.md#anomalog.parsers.Drain3Parser) is the
default. [Drain3](https://github.com/logpai/Drain3) is a streaming
implementation built on the original Drain approach to log template mining, and
AnomaLog uses it as the default template parser for turning raw messages into
reusable patterns.

## :material-timeline-text-outline: Sequence construction

Once `.build()` returns a templated dataset, the next question is how a model
should see the log stream. That is what the sequence builders control.

Use entity grouping when the benchmark is defined around entities such as BGL
nodes or HDFS block IDs:

```python
entity_sequences = dataset.group_by_entity().with_train_fraction(0.8)
first_entity_sequence = next(iter(entity_sequences))

print(first_entity_sequence.sole_entity_id)
print(first_entity_sequence.label, first_entity_sequence.split_label)
print(first_entity_sequence.templates[:3])
```

```text
R02-M1-N0-C:J12-U11
0 train
[
    "RAS KERNEL INFO instruction cache parity error corrected",
    "RAS KERNEL INFO data cache parity error corrected",
    "RAS KERNEL INFO data cache parity error corrected",
]
```

Entity-grouped sequences are ordered by each entity's first timestamp before
the train/test cutoff is applied, so train fractions are deterministic
prefixes of the same chronological entity ordering.

Use fixed or time windows when the downstream method expects sliding windows
rather than one sequence per entity:

```python
fixed_sequences = dataset.group_by_fixed_window(window_size=128, step_size=64)
time_sequences = dataset.group_by_time_window(
    time_span_ms=60_000,
    step_span_ms=30_000,
)
```

All three builders yield `TemplateSequence` objects. A `TemplateSequence`
preserves the information that is still useful before choosing a model family:

- ordered templates via `sequence.templates`
- original event ordering and inter-event gaps via `sequence.events`
- sequence label via `sequence.label`
- deterministic split membership via `sequence.split_label`
- contributing entity IDs via `sequence.entity_ids`

Choose the grouping mode based on the research question:

- `group_by_entity()` for entity-level anomaly detection benchmarks
- `group_by_fixed_window(...)` for count- or order-based sliding-window models
- `group_by_time_window(...)` when temporal span matters more than event count

For entity-based grouping, you can also restrict the training split to normal
entities only:

```python
trainable = (
    dataset.group_by_entity()
    .with_train_fraction(0.8)
    .with_train_on_normal_entities_only()
)
```

In that mode, `train_fraction` still applies to the full entity population.
Anomalous entities are forced into test, so some requested overall train
fractions become impossible. In that case AnomaLog raises an error instead of
quietly changing what the percentage means.

This supervised mode is intentionally a separate constraint from the standard
chronological entity prefix, which keeps repeated runs comparable across train
fractions.

That option is intentionally only available for entity grouping. Fixed-window
and time-window configs do not expose it.

See [Sequences](reference/sequences.md) for the full `TemplateSequence` and
builder APIs.

## :material-vector-polyline: Representations

`TemplateSequence` is still model-agnostic. The representation layer turns that
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

first_sequence_sample = next(iter(sequential))
first_count_sample = next(iter(template_counts))
first_phrase_sample = next(iter(template_phrases))
```

```text
first_sequence_sample.data == [
    "RAS KERNEL INFO instruction cache parity error corrected",
    "RAS KERNEL INFO data cache parity error corrected",
    "RAS KERNEL INFO data cache parity error corrected",
]

first_count_sample.data == Counter({
    "RAS KERNEL INFO data cache parity error corrected": 2,
    "RAS KERNEL INFO instruction cache parity error corrected": 1,
})

first_phrase_sample.data == Counter({
    "ras kernel": 3,
    "kernel info": 3,
    "cache parity": 3,
    ...
})
```

The built-in representations cover different model families:

- `SequentialRepresentation` returns the ordered template stream as `list[str]`
- `TemplateCountRepresentation` returns sparse template counts as `Counter[str]`
- `TemplatePhraseRepresentation` returns sparse phrase counts extracted from template text

Those built-ins are template-centric, but the representation interface is not.
A custom representation receives the full `TemplateSequence`, so it can also
use event timing deltas, extracted parameters, entity IDs, or split metadata.

Each represented sample still keeps `entity_ids`, `label`, `split_label`, and
`window_id`, so you can train and evaluate without losing dataset semantics.
Use `iter_labeled_examples()` only when a downstream library specifically wants
plain `(x, y)` pairs.

For example, that helper is convenient for online-learning libraries such as
[`river`](https://riverml.xyz/):

```python
from anomalog.representations import TemplatePhraseRepresentation

river_examples = (
    dataset.group_by_entity()
    .with_train_fraction(0.8)
    .represent_with(
        TemplatePhraseRepresentation(phrase_ngram_min=1, phrase_ngram_max=2),
    )
    .iter_labeled_examples()
)

first_x, first_y = next(river_examples)
```

```text
first_x == Counter({
    "ras kernel": 3,
    "kernel info": 3,
    "cache parity": 3,
    ...
})
first_y == 0
```

If you need model-specific features beyond template text, implement a custom representation over the full `TemplateSequence`:

```python
from dataclasses import dataclass

from anomalog.representations import SequenceRepresentation
from anomalog.sequences import TemplateSequence


@dataclass(frozen=True)
class SequenceSummaryRepresentation(
    SequenceRepresentation[dict[str, int | list[str] | str]]
):
    name = "sequence_summary"

    def represent(
        self,
        sequence: TemplateSequence,
    ) -> dict[str, int | list[str] | str]:
        return {
            "entity_ids": sequence.entity_ids,
            "parameter_count": sum(len(params) for _, params, _ in sequence.events),
            "timed_event_count": sum(
                dt_prev_ms is not None for _, _, dt_prev_ms in sequence.events
            ),
            "split": sequence.split_label.value,
        }


sequence_summaries = builder.represent_with(SequenceSummaryRepresentation())
```

See [Representations](reference/representations.md) for the full reference.

## :material-source-branch: Reproducibility and caching

AnomaLog is designed so that the dataset pipeline itself is the reproducible
artifact.

In practice, that means:

- the dataset definition lives in a `DatasetSpec`
- raw data materialisation is tied to the configured source
- structured data is persisted and reused instead of reparsed every time
- template mining results are cached and reused when the inputs have not changed
- sequence grouping is deterministic for a fixed built dataset and grouping configuration
- train/test splits come from deterministic ordering and grouping rules rather than from a random number generator

There is no seed you need to set just to make the preprocessing split
reproducible. The ordering rules in the sink and sequence builder define the
split behavior.

[Prefect](https://www.prefect.io/) is used internally for task materialisation
and cache reuse, but the public API is intentionally organised around dataset
stages rather than orchestration details.

The sink and cache layers are part of that reproducibility story:

- the sink is the persisted structured representation reused by later stages
- structured writes are tied to the raw-log materialisation they were derived from
- template mining can be reused when the structured inputs and template configuration are unchanged
- local artifact existence is checked defensively after cached task reuse, because a cached completed state alone is not enough to guarantee the expected file still exists on disk

For contributor-facing implementation details, including the Parquet bucket
layout and cache/runtime modules, see [Development](development.md).
