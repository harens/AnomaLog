# Representations

Representations convert a model-agnostic `TemplateSequence` into the concrete
input shape expected by a detector.

A representation receives the full `TemplateSequence`, not just
`sequence.templates`. That means custom representations can use event timing
(`dt_prev_ms`), extracted parameters, entity IDs, or any other sequence
metadata when building model inputs.

```pycon
>>> from anomalog.representations import (
...     SequentialRepresentation,
...     TemplateCountRepresentation,
...     TemplatePhraseRepresentation,
... )
>>> from anomalog.sequences import TemplateSequence
>>> sequence = TemplateSequence(
...     events=[
...         ("Error on node <*>", ["7"], None),
...         ("Error on node <*>", ["8"], 50),
...     ],
...     label=1,
...     entity_ids=["node-7"],
...     window_id=3,
... )
>>> SequentialRepresentation().represent(sequence)
['Error on node <*>', 'Error on node <*>']
>>> TemplateCountRepresentation().represent(sequence)
Counter({'Error on node <*>': 2})
>>> TemplatePhraseRepresentation(phrase_ngram_min=1, phrase_ngram_max=1).represent(sequence)
Counter({'error on node <*>': 2, 'error': 2, 'on': 2, 'node': 2})
```

Use:

- `SequentialRepresentation` for ordered template streams
- `TemplateCountRepresentation` for sparse template-count vectors
- `TemplatePhraseRepresentation` for phrase-count features derived from template text

The built-ins are intentionally template-centric, but that is a choice of those
representations rather than a limit of the interface.

Represented outputs are wrapped in `SequenceSample`, which preserves
`entity_ids`, `label`, `split_label`, and `window_id` alongside the
representation payload.

You can also define your own representation by implementing
`SequenceRepresentation[T]` and passing it to `represent_with(...)`.

```pycon
>>> from dataclasses import dataclass
>>> @dataclass(frozen=True)
... class SequenceSummaryRepresentation:
...     name = "sequence_summary"
...
...     def represent(self, sequence: TemplateSequence) -> dict[str, int | list[str]]:
...         return {
...             "entity_count": len(sequence.entity_ids),
...             "timed_event_count": sum(
...                 dt_prev_ms is not None for _, _, dt_prev_ms in sequence.events
...             ),
...             "entity_ids": sequence.entity_ids,
...         }
>>> SequenceSummaryRepresentation().represent(sequence)
{'entity_count': 1, 'timed_event_count': 1, 'entity_ids': ['node-7']}
```

## `anomalog.representations`

::: anomalog.representations
