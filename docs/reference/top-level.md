# Top-level API

This page covers the smallest public entrypoint surface of AnomaLog.

Start here when you want the library-level names that most examples import
directly, rather than the lower-level building blocks under the submodules.

```pycon
>>> from anomalog import DatasetSpec, SplitLabel
>>> DatasetSpec("demo").dataset_name
'demo'
>>> SplitLabel.TRAIN.value
'train'
```

## `anomalog`

::: anomalog
