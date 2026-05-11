# Experiments

This directory holds reproducible model experimentation built on top of
AnomaLog preprocessing.

## Layout

- `configs/models/`: detector configurations such as template-frequency,
  handwritten Naive Bayes, `river`-backed baselines, and the scoped DeepLog and DeepCASE models.
- `configs/datasets/`: dataset-owned experiment manifests that bind one dataset variant to one or more detector configs via `[[models]]` entries, then optionally override them through fixed overrides and Cartesian-product axes.
- `runners/`: Python entrypoints for executing experiments.
- `analysis/`: notebooks and one-off visual analysis only.
- `results/`: generated run artifacts. These are not source-controlled.

## Design

Preprocessing stays in the dataset config layer. A dataset variant controls:

- where raw logs come from
- how they are parsed and templated
- how sequences are generated

Model experimentation stays in the dataset-manifest layer. A dataset manifest
binds one base dataset variant to one or more detector configs, chooses the
results root, and can expand into multiple concrete runs through validated
override axes. The `[[models]]` table array keeps the common case short: the
dataset owns the experiment shape, and the model configs remain shared under
`configs/models/`.
If a manifest mixes heavyweight and lightweight detectors, give the relevant
`[[models]]` entries different `run_group` values. The runner executes one
group at a time, but still uses `max_workers` within each group, so
lightweight models can run in parallel while memory-heavy runs such as
DeepLog and DeepCASE stay isolated.
Manifest execution defaults `max_workers` to `"auto"`, which uses up to the
concrete run count and local CPU count. Set an explicit positive integer when a
manifest needs a stricter cap.

That keeps preprocessing ablations separate from experiment matrices while
still using the existing `DatasetSpec(...).build()` API as the source of
truth.

The checked-in dataset-manifest set is split by dataset family:

- `bgl_entity_chronological.toml` and `hdfs_v1_entity_chronological.toml`
  collect the template-frequency, Naive Bayes, and DeepCASE models for the
  chronological entity baselines.
- `bgl_deeplog_paper_1pct_normal_entry_stream_no_online.toml` and
  `bgl_deeplog_paper_10pct_entry_stream_no_online.toml` keep the BGL DeepLog
  paper-reproduction probes separate because they differ in train fraction and
  split policy.
- `hdfs_v1_deeplog_paper_entry100k_split_partial.toml` and
  `hdfs_v1_deeplog_paper_entry100k_assign_first.toml` each bundle the HDFS
  paper DeepLog variants with both the key-only and full DeepLog model
  configs, so the model choice sits next to the dataset split policy instead of
  living in a separate file.
- `hdfs_wuyifan18_deeplog_preprocessed.toml` keeps the exact `hdfs_train` /
  `hdfs_test` boundary for the wuyifan18 archive and now lists both the
  DeepLog key-only and DeepCASE model configs against the same dataset
  manifest.
- `openstack_deeplog_preprocessed.toml` is the matching OpenStack
  file-boundary reproduction probe. It materialises LogHub's OpenStack archive,
  parses with Spell (`tau=0.5`), builds the DeepLog event-id vocabulary from
  `openstack_normal1.log`, and then evaluates `openstack_normal2.log` and
  `openstack_abnormal.log` on the same exact boundary. The OpenStack parser
  keeps the log level and component in the template input and now preserves
  traceback and terminal error rows that only differ by the absence of an
  address field.

That keeps detector-specific training policy explicit. DeepLog-style runs use
`train_on_normal_entities_only` for the training prefix on entity-grouped
datasets, whereas DeepCASE-style runs leave it disabled and only use the
chronological prefix/suffix split. The wuyifan18 reproduction dataset is
intentionally separate from that policy: it reproduces the preprocessed file
boundary directly, so the full training file and both test files are preserved
exactly. The same dataset manifest can list both detector families, so the
model choice stays next to the dataset contract. If those detectors are both
benchmarked on the same dataset family, keep them in the same manifest or use
fixed overrides rather than letting a shared dataset preset imply the wrong
training contract.

Custom datasets are still supported through the same config model by setting `source` and `structured_parser` instead of `preset`.

`sequence.train_on_normal_entities_only` is only available for entity-grouped
datasets, matching the core `anomalog` sequence API. It is a detector policy,
not part of the shared chronological split contract.

Entity-grouped sequences are ordered chronologically by each entity's first
timestamp before the split is applied. A fixed chronological holdout suffix
defines the test set, and the requested train fraction is applied to the total
population before rounding and capping against the remaining middle band. The
middle portion between the train prefix and fixed test suffix is withheld from
the current run, so performance changes reflect model behaviour rather than a
moving test set. When normal-only training is enabled, the run trains on the
normal subset of that chronological prefix and records the realised train size
separately from the requested fraction.

The same fixed-holdout contract is available for fixed-window and time-window
sequence configs through the `sequence.train_fraction` and
`sequence.test_fraction` pair. The defaults are `0.2` and `0.8`, respectively,
so omitted values still preserve the same fixed suffix behaviour.

When `sequence.train_on_normal_entities_only = true`, the requested
`train_fraction` still applies to the full entity population. Anomalous
entities are forced into test, so some requested overall train fractions are
not realised in full under that constraint. The run no longer fails when the
normal subset is smaller than the requested prefix quota. Result manifests
record the requested fractions, train pool size, realised train size, excluded
prefix count, and eligible normal-entity count so the constraint remains
visible.

## Running

From the repository root:

```bash
uv run python -m experiments.runners.run_experiment \
  --config experiments/configs/datasets/bgl_entity_chronological.toml
```

Add `--force` to replace the deterministic output directories for the same
concrete run variants.
Add `--write-predictions` if you want each run to persist `predictions.jsonl`
alongside the other result artefacts.

The checked-in Slurm wrappers live under `slurm/` and are generated from
`slurm/jobs.toml`. They keep a 1:1 mapping with the dataset manifests and can
be refreshed with `uv run python -m experiments.slurm_scripts`.

To audit dataset/split readiness for DeepLog paper reproduction:

```bash
uv run python -m experiments.runners.audit_deeplog_data
```

The DeepCASE paper-readiness report is checked in at
`experiments/reports/deepcase_reproduction_readiness.md`.

## Caching Strategy

AnomaLog caches dataset preprocessing work, not experiment model execution.

- Dataset sourcing, structured parsing, template mining, and other
  preprocessing stages reuse the existing AnomaLog and Prefect-backed caches
  when their inputs and upstream assets have not changed.
- Local-output materialisation also guards against stale Prefect cache hits
  that point at an incompatible storage base path, so reruns keep working
  after a checkout moves or a local storage root changes.
- Prefect-backed materialisations use a versioned cache namespace under the
  user cache root, with a stable shared result-storage base, so cached dataset
  preprocessing does not inherit run-specific `PREFECT_HOME` paths.
- Cold dataset builds are serialised per dataset namespace
  (`dataset_name` plus cache roots), so multi-process runs do not race while
  materialising the shared AnomaLog dataset cache for the first time.
- Structured parquet materialisation now writes a tiny entity chronology
  sidecar alongside the parquet partitions, so entity-grouped readers can
  reuse first-seen ordering without rescanning all rows.
- Concrete runs write to deterministic directories under
  `experiments/results/<concrete-run-name>/<fingerprint>/`, where the
  fingerprint comes from the fully resolved manifest, dataset, and model config.
- Re-running the exact same config reuses that deterministic output directory.
  Use `--force` when you want to overwrite it.
- Changing the dataset, sequence settings, or model config produces a new
  fingerprint and therefore a new result directory.
- Detector training and test scoring are intentionally not cached as separate
  reusable stages. If you change an experiment config, the model is retrained
  and the test split is rescored for that new fingerprint.
- Entity-grouped experiments record the fixed train-pool, train-prefix,
  ignored, and test counts in `sequence_split_counts`, plus the requested
  fractions, train pool size, realised train size, excluded prefix count, and
  realised fractions in `sequence_split_summary`.

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

- `experiment_config.json`: normalised manifest, concrete override, dataset, and model config
- `dataset_manifest.json`: dataset fingerprint, source summary, raw-log hash, cache roots, sequence settings, and dataset statistics
  It also records `sequence_split_summary`, which makes the effective split
  explicit when training is restricted to normal entities only.
- `metrics.json`: task-aware detector metrics, including the selected
  `primary_metric_scope`, canonical scoped metric blocks, and run-level
  evaluation metadata such as `evaluation_unit`, `prediction_unit`,
  `label_unit`, and split policy details at the top level.
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

For built-in datasets, prefer `preset = "bgl"`, `preset = "hdfs_v1"`, or
`preset = "openstack_deeplog_preprocessed"` depending on whether you want the
LogHub-style raw corpus or the DeepLog-style preprocessed session files with
the exact train/test file boundary. When a run is only used in one place,
consider listing the relevant model refs directly under `[[models]]` in the
same dataset manifest instead of creating a temporary per-model file.
For custom datasets, define `source`, `structured_parser`, optional `label_reader`, and sequence settings directly in the dataset config.
Omit `[cache_paths]` to use AnomaLog's default platformdirs-based cache/data locations.

To add or update an experiment matrix, edit the relevant file in
`configs/datasets/`. Use `[overrides]` for fixed adjustments such as changing
`dataset.sequence.train_on_normal_entities_only`, and `[[axes]]` when you want
Cartesian products across fields such as `dataset.sequence.train_fraction`. For
a one-off reproduction, you can still inline the dataset table next to the
model refs in the same file. For the wuyifan18 DeepLog probe, keep the raw-entry
split fixed and vary only values that should not alter the file boundary if you
want a regression test against accidental split drift. Add `max_workers = 2`
or another positive integer only when the default `"auto"` parallelism is too
aggressive for a particular backend or machine.
Use `run_group = "..."` on individual `[[models]]` entries when a manifest
contains a mixture of heavy and light detectors. The group name is only a
scheduling hint, not part of the detector config itself.

If the manifest needs a Slurm submission wrapper, add a matching entry to
`slurm/jobs.toml` and regenerate the checked-in `.sbatch` files.

To add a new detector implementation, extend `experiments/models/` with a tagged config subclass and detector subclass so the built-in registries pick them up automatically.

The experiment layer intentionally does not have its own `pyproject.toml`. It shares the repo root environment so dataset code, tests, docs, and experiment runners stay locked and validated together.
