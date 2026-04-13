# Presets

Presets are checked-in `DatasetSpec` definitions for common benchmark datasets.

Use them when you want a visible starting configuration that you can inspect,
reuse, and modify for preprocessing ablations.

```pycon
>>> from anomalog.presets import bgl, preset_names
>>> preset_names()
('bgl', 'hdfs_v1')
>>> bgl.dataset_name
'BGL'
>>> bgl.structured_parser.name
'bgl'
```

## `anomalog.presets`

::: anomalog.presets
