# Sequences

Sequences are the point where a templated dataset becomes a modeling dataset.

This page covers the grouping builders, the `TemplateSequence` shape, and the
split semantics that determine how sequences are assigned to train and test.

```pycon
>>> from anomalog.sequences import SplitLabel, TemplateSequence
>>> sequence = TemplateSequence(
...     events=[("template <*>", ["x"], None), ("template <*>", ["y"], 10)],
...     label=0,
...     entity_ids=["node-1"],
...     window_id=7,
...     split_label=SplitLabel.TRAIN,
... )
>>> sequence.sole_entity_id
'node-1'
>>> sequence.templates
['template <*>', 'template <*>']
>>> sequence.split_label.value
'train'
```

## `anomalog.sequences`

::: anomalog.sequences
