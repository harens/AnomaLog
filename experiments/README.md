# Experiments

This directory holds reproducible model experimentation built on top of
AnomaLog preprocessing.

## Layout

- `configs/models/`: detector configurations such as template-frequency,
  a normal-only Markov transition baseline, handwritten Naive Bayes,
  and the scoped DeepLog and DeepCASE models.
- `configs/datasets/`: dataset-owned manifests grouped by family, with
  reusable base manifests for dataset contracts, sequence settings, and shared
  overrides.
- `configs/registry.toml`: named experiment registry that combines dataset
  names with reusable model sets and dataset families into reproducible run
  targets.
- `configs/slurm.toml`: Slurm submission defaults for the optional backend.
- `runners/`: Python entrypoints for executing experiments.
- `execution/`: optional execution backends such as Slurm.
- `analysis/`: notebooks and one-off visual analysis only.
- `results/`: generated run artifacts. These are not source-controlled.

## Design

Preprocessing stays in the dataset config layer. A dataset variant controls:

- where raw logs come from
- how they are parsed and templated
- how sequences are generated

Model experimentation stays in the dataset-manifest layer for dataset-specific
shape and sequencing, while the registry owns the named experiment catalogue.
A dataset manifest binds one base dataset variant to fixed overrides and
validated override axes. The preferred path is to keep the dataset file
focused on the dataset contract and let the registry tie that manifest to
shared model sets under `configs/models/`.
If a manifest mixes heavyweight and lightweight detectors, keep the detector
family in the model set rather than duplicating it in the dataset file. The
runner still executes one concrete run at a time, but the registry expands a
single model-set name into the concrete model configs that should share the
same reporting metadata.
Manifest execution defaults `max_workers` to `"auto"`, which uses up to the
concrete run count and local CPU count. Set an explicit positive integer when a
manifest needs a stricter cap.

That keeps preprocessing ablations separate from experiment matrices while
still using the existing `DatasetSpec(...).build()` API as the source of
truth.

The checked-in dataset-manifest set is split by dataset family:

- `shared/entity_chronological_base.toml`
  holds the shared chronological entity dataset contract.
- `bgl/entity_chronological.toml` and `hdfs/v1_entity_chronological.toml`
  are the concrete chronological entity variants built from that base.
- `bgl/deeplog_paper_1pct_normal_entry_stream_no_online.toml` and
  `bgl/deeplog_paper_10pct_entry_stream_no_online.toml` keep the BGL DeepLog
  paper-reproduction probes separate because they differ in train fraction and
  split policy.
- `hdfs/v1_deeplog_paper_entry100k_split_partial.toml` and
  `hdfs/v1_deeplog_paper_entry100k_assign_first.toml` each bundle the HDFS
  paper DeepLog variant with template-frequency and Markov baselines, so the
  model choice sits next to the dataset split policy instead of living in a
  separate file.
- `hdfs/wuyifan18_deeplog_preprocessed.toml` keeps the exact `hdfs_train` /
  `hdfs_test` boundary for the wuyifan18 archive and now lists the DeepLog
  key-only, template-frequency, and Markov model configs against the same
  dataset manifest.
- `openstack/deeplog_preprocessed.toml` is the matching OpenStack
  file-boundary reproduction probe. It materialises LogHub's OpenStack archive,
  parses with Spell (`tau=0.5`) over a canonicalised message body, builds the
  DeepLog event-id vocabulary from `openstack_normal1.log`, and then evaluates
  `openstack_normal2.log` and `openstack_abnormal.log` on the same exact
  boundary. The OpenStack parser prefixes recovered `instance_id` values with
  the split name so the file boundary survives grouping, and it also strips the
  leading session tag plus volatile UUID, IP, path, hex, and numeric tokens
  before Spell sees the message text. That keeps session identity out of the
  template vocabulary while preserving the file-boundary split contract.
  The manifest also carries template-frequency, Markov, and DeepCASE
  comparators for the same file boundary.
- `ait_ads/<scenario>.toml` covers the AIT Alert Data Set (AIT-ADS) one
  scenario at a time for `fox`, `harrison`, `russellmitchell`, `santos`,
  `shaw`, `wardbeck`, `wheeler`, and `wilson`. Each scenario manifest now
  keeps only the shared dataset contract while the registry defines the named
  model pairings that should run against it. The AIT-ADS source emits stable
  semantic alert keys from AMiner, Wazuh, and Suricata metadata, so the
  manifests intentionally use the identity template parser rather than
  re-mining templates from already-normalised keys.

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

The central registry is intentionally small. It now has four concepts:

- `model_sets` define reusable detector families such as `baselines`,
  `deeplog`, and `deepcase`.
- `experiment_presets` combine model sets and shared overrides into reusable
  experiment profiles.
- `experiments` name one dataset and the preset that should run on it.
- `experiment_sets` name a dataset family and expand one dataset list line into
  the concrete experiment names for that family.

The runner derives reporting groups from the preset and model-set names, so the
TOML does not need `groups` or `run_group` in the normal case. If you need a
small or CI-friendly check, select the explicit experiment name instead of
inventing another registry tag.

The DeepLog reproduction manifests also carry simple sanity baselines on the
same split, so the paper target is always compared against a sequence-statistics
floor rather than only against the neural model.

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

The baseline supervision split is intentional:

- Naive Bayes is supervised at the sequence level.
- Template Frequency is unsupervised apart from optional normal-score calibration.
- Markov is a normal-only sequence-order comparator for DeepLog-style runs.
- DeepCASE is label-aware during fit and falls back to sequence labels when event labels are missing.
- DeepLog is normal-only at fit time, with labels used for eligibility bookkeeping and evaluation rather than class-target learning.

Treat the baseline scores as checks on corpus separability and template
statistics rather than as direct competitors to DeepLog or DeepCASE.

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
  --experiment bgl_entity_chronological
```

To list the curated registry:

```bash
uv run python -m experiments.runners.run_suite --list
```

The listing shows the registry name, dataset config, preset, model-set names,
and the derived reporting groups. Use explicit experiment names when you need
a specific dataset/preset combination.

To run a local group:

```bash
uv run python -m experiments.runners.run_suite \
  --group bgl_deeplog_paper \
  --group hdfs_deeplog_paper \
  --max-parallel 2
```

Add `--force` to replace the deterministic output directories for the same
concrete run variants. Add `--write-predictions` if you want each run to
persist `predictions.jsonl` alongside the other result artefacts.

The same registry also drives the optional Slurm backend:

```bash
uv run python -m experiments.execution.slurm submit \
  --group bgl_deeplog_paper \
  --group hdfs_deeplog_paper
```

Use `--dry-run` with either runner to preview the selected experiments without
executing them. The Slurm backend now submits one Slurm job array for the
selected experiments, leaving scheduling and any cluster-side concurrency
policy to Slurm itself. The selected experiment names are embedded directly
into the wrapped script, so no manifest file is written. Array task logs are
written beneath `experiments/results/slurm-logs/<selection-label>/` with
`%A_%a` in the filenames.

If you do not have Slurm, the local suite runner is the canonical path for
reproducing the paper experiments:

```bash
uv run python -m experiments.runners.run_suite \
  --group bgl_deeplog_paper \
  --group hdfs_deeplog_paper \
  --max-parallel 2
```

To audit dataset/split readiness for DeepLog paper reproduction, including the
embedded dataset tables in the DeepLog experiment matrices:

```bash
uv run python -m experiments.runners.audit_deeplog_data
```

The DeepCASE paper-readiness report is checked in at
`experiments/reports/deepcase_reproduction_readiness.md`.
The baseline sanity report is checked in at
`experiments/reports/baseline_sanity_report.md`.

## Caching Strategy

AnomaLog caches dataset preprocessing work, not experiment model execution.

- Dataset sourcing, structured parsing, template mining, and other
  preprocessing stages reuse the existing AnomaLog and Prefect-backed caches
  when their inputs and upstream assets have not changed.
- Local-output materialisation also guards against stale Prefect cache hits
  that point at an incompatible storage base path, so reruns keep working
  after a checkout moves or a local storage root changes, even when Prefect
  wraps the underlying storage error in a chained exception or
  `ExceptionGroup`.
- Prefect-backed materialisations use a versioned cache namespace under the
  user cache root, with a stable shared result-storage base, so cached dataset
  preprocessing does not inherit run-specific `PREFECT_HOME` paths.
  The generated Slurm wrappers pin that base to `${PREFECT_ROOT}/storage`
  while keeping `PREFECT_HOME` separate for the local Prefect database.
- Cold dataset builds are serialised per dataset namespace
  (`dataset_name` plus cache roots), so multi-process runs do not race while
  materialising the shared AnomaLog dataset cache for the first time.
- Structured parquet materialisation now writes a tiny entity chronology
  sidecar alongside the parquet partitions, so entity-grouped readers can
  reuse first-seen ordering without rescanning all rows.
  If a cached parquet fragment is unreadable, the sink rebuilds that dataset
  namespace under the same dataset-build lock before any model sees it, rather
  than letting one model observe an empty train prefix while another repairs
  the cache.
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

To run DeepLog/DeepCASE experiments, install the matching optional extras
first:

```bash
uv sync --extra experiments --extra deeplog
uv sync --extra experiments --extra deepcase
```

Use `uv sync --all-extras` if you want every experiment backend in one
environment.

## Result Artifacts

Each concrete run writes a deterministic directory under
`experiments/results/<concrete-run-name>/<fingerprint>/` containing:

- `experiment_config.json`: normalised manifest, concrete override, dataset,
  model, and named-experiment metadata
- `dataset_manifest.json`: dataset fingerprint, source summary, raw-log hash,
  sequence settings, compact dataset statistics, and named-experiment
  metadata when the run came from the registry
  It also records `sequence_split_summary`, which makes the effective split
  explicit when training is restricted to normal entities only.
- `metrics.json`: task-aware detector metrics, including the selected
  `primary_metric_scope`, canonical scoped metric blocks, and run-level
  evaluation metadata such as `evaluation_unit`, `prediction_unit`,
  `label_unit`, and split policy details at the top level. By default this
  file keeps the paper-facing summaries only; pass `--debug-reporting` if you
  need the fuller diagnostic payloads during development. DeepLog runs also
  record an exact-rank `top_g_replay` curve using the configured paper
  cut-offs, so you can inspect multiple top-`g` thresholds from one fitted
  model without re-running inference.
- `predictions.jsonl`: optional test-sequence outputs, including detector
  scores and any emitted key phrases when `--write-predictions` is supplied
- `environment.json`: Python, platform, package, git metadata, and the command
  used to launch the run
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
the exact train/test file boundary. For AIT-ADS, prefer the checked-in
`ait_ads/<scenario>.toml` manifests so the chronological stream split and
canonical alert-key contract stay consistent across detector families. Use the
registry to choose which model config should run for each named experiment.
For custom datasets, define `source`, `structured_parser`, optional `label_reader`, and sequence settings directly in the dataset config.
Use `[dataset.cache_paths] namespace = "..."` when you want dataset-scoped
cache and data roots without spelling out both paths separately. Omit
`[cache_paths]` entirely to use AnomaLog's default platformdirs-based
cache/data locations.

To add or update a dataset manifest, edit the relevant file in
`configs/datasets/`. Use `[overrides]` for fixed adjustments such as changing
`dataset.sequence.train_on_normal_entities_only`, and `[[axes]]` when you want
Cartesian products across fields such as `dataset.sequence.train_fraction`.
For a one-off reproduction, keep the dataset file focused on the dataset
contract and use the registry for the model pairing. Add `max_workers = 2` or
another positive integer only when the default `"auto"` parallelism is too
aggressive for a particular backend or machine.

When several manifests share the same dataset shape, use `extends = "base.toml"`
at the top of the child manifest. Nested tables merge recursively, so the base
file can hold the shared boilerplate while each scenario keeps only its
specific overrides.

To add a named experiment to the canonical registry, add a table to
`configs/registry.toml` that points at one dataset manifest and the preset that
should run on it. You do not need to spell out `groups` or `run_group` for the
normal case.

Example:

```toml
[model_sets.baselines]
models = ["template_frequency_default", "naive_bayes_default", "markov_default"]

[experiment_presets.entity_with_deepcase]
models = ["baselines", "deepcase"]

[experiments.bgl_entity_chronological]
dataset = "bgl/entity_chronological"
preset = "entity_with_deepcase"
```

For a dataset family such as AIT-ADS, use `experiment_sets` so the dataset list
stays to one line:

```toml
[experiment_presets.ait_ads]
models = ["baselines", "deeplog", "deepcase"]

[experiment_sets.ait_ads]
preset = "ait_ads"
dataset_prefix = "ait_ads"
datasets = ["fox", "harrison", "russellmitchell"]
```

To add a new detector implementation, extend `experiments/models/` with a tagged config subclass and detector subclass so the built-in registries pick them up automatically.

The experiment layer intentionally does not have its own `pyproject.toml`. It shares the repo root environment so dataset code, tests, docs, and experiment runners stay locked and validated together.
