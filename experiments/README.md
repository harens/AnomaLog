# Experiments

This directory holds reproducible model experimentation built on top of
AnomaLog preprocessing.

## Layout

- `configs/datasets/`: dataset variants, usually built from AnomaLog presets like `bgl` and `hdfs_v1`, but also able to define custom sources and parsers.
- `configs/models/`: detector configurations such as template-frequency,
  handwritten Naive Bayes, `river`-backed baselines, and the scoped DeepLog and DeepCASE models.
- `configs/sweeps/`: experiment sweep definitions that reference one base dataset variant and one base model config, then optionally override them through fixed overrides and Cartesian-product axes.
- `runners/`: Python entrypoints for executing experiments.
- `analysis/`: notebooks and one-off visual analysis only.
- `results/`: generated run artifacts. These are not source-controlled.

## Design

Preprocessing stays in the dataset config layer. A dataset variant controls:

- where raw logs come from
- how they are parsed and templated
- how sequences are generated

Model experimentation stays in the sweep config layer. A sweep config binds one
base dataset variant to one base detector config, chooses the results root,
and can expand into multiple concrete runs through validated override axes.
Sweep execution defaults `max_workers` to `"auto"`, which uses up to the
concrete run count and local CPU count.
Set an explicit positive integer when a sweep needs a stricter cap.

That keeps preprocessing ablations separate from experiment matrices while
still using the existing `DatasetSpec(...).build()` API as the source of
truth.

The checked-in sweep set is intentionally small:

- `bgl.toml` sweeps DeepCase, DeepLog, template-frequency, and Naive Bayes
  across train fractions `0.01`, `0.1`, and `0.2`
- `hdfs_v1.toml` does the same for HDFS v1

BGL sweeps that need anomalous entities in the training budget now express that
as a sweep override instead of a second near-duplicate dataset file.

Custom datasets are still supported through the same config model by setting `source` and `structured_parser` instead of `preset`.

`sequence.train_on_normal_entities_only` is only available for entity-grouped
datasets, matching the core `anomalog` sequence API.

Entity-grouped sequences are ordered chronologically by each entity's first
timestamp before the split is applied, so train fractions are stable prefixes
of the same entity ordering across repeated runs.

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
  --config experiments/configs/sweeps/bgl.toml
```

Add `--force` to replace the deterministic output directories for the same
concrete sweep variants.
Add `--write-predictions` if you want each run to persist `predictions.jsonl`
alongside the other result artefacts.

## Caching Strategy

AnomaLog caches dataset preprocessing work, not experiment model execution.

- Dataset sourcing, structured parsing, template mining, and other
  preprocessing stages reuse the existing AnomaLog and Prefect-backed caches
  when their inputs and upstream assets have not changed.
- Cold dataset builds are serialised per dataset namespace
  (`dataset_name` plus cache roots), so multi-process sweeps do not race while
  materialising the shared AnomaLog dataset cache for the first time.
- Structured parquet materialisation now writes a tiny entity chronology
  sidecar alongside the parquet partitions, so entity-grouped readers can
  reuse first-seen ordering without rescanning all rows.
- Concrete sweep runs write to deterministic directories under
  `experiments/results/<concrete-run-name>/<fingerprint>/`, where the
  fingerprint comes from the fully resolved sweep, dataset, and model config.
- Re-running the exact same config reuses that deterministic output directory.
  Use `--force` when you want to overwrite it.
- Changing the dataset, sequence settings, or model config produces a new
  fingerprint and therefore a new result directory.
- Detector training and test scoring are intentionally not cached as separate
  reusable stages. If you change an experiment config, the model is retrained
  and the test split is rescored for that new fingerprint.

To run `river`-backed or DeepLog/DeepCASE experiments, install the matching
optional extras first:

```bash
uv sync --extra experiments --extra river
uv sync --extra experiments --extra deeplog
uv sync --extra experiments --extra deepcase
```

Use `uv sync --all-extras` if you want every experiment backend in one
environment.

## Result Artifacts

Each concrete run writes a deterministic directory under `experiments/results/<concrete-run-name>/<fingerprint>/` containing:

- `experiment_config.json`: normalised sweep, concrete override, dataset, and model config
- `dataset_manifest.json`: dataset fingerprint, source summary, raw-log hash, cache roots, sequence settings, and dataset statistics
  It also records `sequence_split_summary`, which makes the effective split
  explicit when training is restricted to normal entities only.
- `metrics.json`: detector metrics
- `predictions.jsonl`: optional test-sequence outputs, including detector
  scores and any emitted key phrases when `--write-predictions` is supplied
- `environment.json`: Python, platform, package, and git metadata
- `run.log`: run-time logging from dataset build through detector evaluation

Predictions are still scored from a streaming replay of the sequence builder
instead of materialising the full sequence list in memory. Train sequences are
still consumed for fitting and run summaries, but they are not scored or
written to the prediction stream unless you explicitly opt in with
`--write-predictions`.

## Adding More Experiments

To add a preprocessing ablation, create another file in `configs/datasets/`.

For built-in datasets, prefer `preset = "bgl"` or `preset = "hdfs_v1"`.
For custom datasets, define `source`, `structured_parser`, optional `label_reader`, and sequence settings directly in the dataset config.
Omit `[cache_paths]` to use AnomaLog's default platformdirs-based cache/data locations.

To add or update an experiment matrix, create another file in `configs/sweeps/`.
Use `[overrides]` for fixed adjustments such as changing
`dataset.sequence.train_on_normal_entities_only`, and `[[axes]]` when you want
Cartesian products across fields such as `sweep.model` or
`dataset.sequence.train_fraction`. Add `max_workers = 2` or another positive
integer only when the default `"auto"` parallelism is too aggressive for a
particular backend or machine.

To add a new detector implementation, extend `experiments/models/` with a tagged config subclass and detector subclass so the built-in registries pick them up automatically.

The experiment layer intentionally does not have its own `pyproject.toml`. It shares the repo root environment so dataset code, tests, docs, and experiment runners stay locked and validated together.
