# Experiments

This directory holds reproducible model experimentation built on top of
AnomaLog preprocessing.

## Layout

- `configs/datasets/`: dataset variants, usually built from AnomaLog presets like `bgl` and `hdfs_v1`, but also able to define custom sources and parsers.
- `configs/models/`: detector configurations such as template-frequency,
  handwritten Naive Bayes, `river`-backed baselines, and the scoped DeepLog and DeepCASE models.
- `configs/runs/`: experiment run definitions that reference one dataset variant and one model config.
- `runners/`: Python entrypoints for executing experiments.
- `analysis/`: notebooks and one-off visual analysis only.
- `results/`: generated run artifacts. These are not source-controlled.

## Design

Preprocessing stays in the dataset config layer. A dataset variant controls:

- where raw logs come from
- how they are parsed and templated
- how sequences are generated

Model experimentation stays in the model config layer. A run config binds one
dataset variant to one detector config and chooses the results root.

That keeps preprocessing ablations separate from model sweeps while still using the existing `DatasetSpec(...).build()` API as the source of truth.

The checked-in examples use AnomaLog presets rather than tiny test fixtures:

- `bgl_entity` for unsupervised-style entity splits
- `bgl_entity_supervised` for supervised detectors on BGL
- `hdfs_v1_entity_supervised` for supervised detectors that need both labels in train

Custom datasets are still supported through the same config model by setting `source` and `structured_parser` instead of `preset`.

`sequence.train_on_normal_entities_only` is only available for entity-grouped
datasets, matching the core `anomalog` sequence API.

When `sequence.train_on_normal_entities_only = true`, the requested
`train_fraction` still applies to the full entity population. Anomalous
entities are forced into test, so some requested overall train fractions are
impossible under that constraint. Those runs fail fast instead of silently
reinterpreting the percentage. Result manifests still record the eligible
normal-entity count so the constraint remains visible.

## Running

From the repository root:

```bash
uv run python -m experiments.runners.run_experiment \
  --config experiments/configs/runs/bgl_template_frequency.toml
```

Add `--force` to replace the deterministic output directory for the same config fingerprint.

To run `river`-backed or DeepLog/DeepCASE experiments, install the optional
experiment dependencies first:

```bash
uv sync --group experiments
```

## Result Artifacts

Each run writes a deterministic directory under `experiments/results/<run-name>/<fingerprint>/` containing:

- `run_config.json`: normalised run, dataset, and model config
- `dataset_manifest.json`: dataset fingerprint, source summary, raw-log hash, cache roots, sequence settings, and dataset statistics
  It also records `sequence_split_summary`, which makes the effective split
  explicit when training is restricted to normal entities only.
- `metrics.json`: detector metrics
- `predictions.jsonl`: test-sequence outputs, including detector scores and any emitted key phrases
- `environment.json`: Python, platform, package, and git metadata
- `run.log`: run-time logging from dataset build through detector evaluation

Predictions are written as a stream to `predictions.jsonl`, and detector evaluation rereads the sequence builder instead of materialising the full sequence list in memory. Train sequences are still consumed for fitting and run summaries, but they are no longer scored or written to the prediction stream.

## Adding More Experiments

To add a preprocessing ablation, create another file in `configs/datasets/`.

For built-in datasets, prefer `preset = "bgl"` or `preset = "hdfs_v1"`.
For custom datasets, define `source`, `structured_parser`, optional `label_reader`, and sequence settings directly in the dataset config.
Omit `[cache_paths]` to use AnomaLog's default platformdirs-based cache/data locations.

To add a detector sweep, create another file in `configs/models/` and point a run config at it.

To add a new detector implementation, extend `experiments/models/` with a tagged config subclass and detector subclass so the built-in registries pick them up automatically.

The experiment layer intentionally does not have its own `pyproject.toml`. It shares the repo root environment so dataset code, tests, docs, and experiment runners stay locked and validated together.
