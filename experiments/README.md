# Experiments

This directory contains the AnomaLog experiment layer: reproducible detector runs
built on top of the reusable AnomaLog preprocessing pipeline.

The aim of this layer is to make each experiment explicit and auditable. Dataset
manifests define the realised corpus and sequence construction. Model configs
define detector behaviour. The registry binds those pieces into named runs.

## Directory layout

| Path | Purpose |
| --- | --- |
| `configs/datasets/` | Dataset manifests grouped by dataset family. These define corpus variants, sequence settings, overrides, and sweep axes. |
| `configs/models/` | Detector configs for DeepLog, DeepCASE, template frequency, Markov, and Naive Bayes. |
| `configs/registry.toml` | Canonical experiment registry. It maps named experiments or experiment sets to dataset manifests and model configs. |
| `configs/slurm.toml` | Defaults for the optional Slurm backend. |
| `runners/` | Local experiment and suite entrypoints. |
| `execution/` | Optional execution backends, including Slurm submission. |
| `analysis/` | One-off notebooks and exploratory analysis. Not part of the reproducible run contract. |
| `reports/` | Audit and readiness notes used to interpret experiment results. |
| `results/` | Generated run artefacts. These are not source-controlled. |

## Core design

The experiment layer separates three concerns.

1. **Dataset contracts** live in `configs/datasets/`. A dataset manifest fixes
the corpus object: source artefact, parser, template or event-key policy,
sequence construction, split semantics, and any fixed dataset-level overrides.

2. **Model behaviour** lives in `configs/models/`. A model config fixes detector
settings such as DeepLog history size, top-`g` values, DeepCASE clustering
settings, or baseline configuration.

3. **Named experiment selection** lives in `configs/registry.toml`. The registry
keeps the canonical experiment catalogue small and explicit by pairing one
dataset manifest with one or more model configs or model sets.

This means preprocessing ablations should normally become new dataset manifests,
whereas detector changes should normally become new model configs. The registry
should only name the combinations worth running or reporting.

## Important conventions

- Keep dataset files focused on the dataset contract. Use the registry to pair
  those datasets with model configs.
- Prefer explicit registry experiment names when reproducing a specific result.
- Use `experiment_sets` for dataset-family groups, not for hiding protocol
  differences.
- Use `model_sets` for repeated detector bundles, such as shared baseline stacks.
- Treat baselines as sanity checks on corpus separability or transition signal,
  not as direct DeepLog or DeepCASE competitors unless the metric unit matches.
- Use `sequence.train_on_normal_entities_only` only for entity-grouped datasets.
  This is a detector-facing training policy, not part of the shared chronological
  split contract.
- When normal-only entity training is enabled, anomalous entities are forced out
  of the realised training set. The requested train fraction may therefore not
  be realised exactly; the result artefacts record the realised counts.

## Main dataset manifests

The checked-in manifests are grouped by dataset family.

| Manifest | Role |
| --- | --- |
| `shared/entity_chronological_base.toml` | Shared chronological entity-grouped base contract. |
| `bgl/entity_chronological.toml` | BGL entity-chronological variant. |
| `bgl/bgl_deeplog_ccs2017_paper_1pct_normal_entry_stream_no_online.toml` | BGL DeepLog static paper-facing 1% normal-entry stream reconstruction. |
| `bgl/bgl_deeplog_ccs2017_paper_10pct_entry_stream_no_online.toml` | BGL DeepLog static paper-facing 10% early-entry reconstruction. |
| `hdfs/v1_entity_chronological.toml` | HDFS raw-style entity-chronological variant. |
| `hdfs/v1_deeplog_paper_entry100k_split_partial.toml` | HDFS raw-prefix DeepLog reconstruction with split-partial boundary handling. |
| `hdfs/v1_deeplog_paper_entry100k_assign_first.toml` | HDFS raw-prefix DeepLog reconstruction with assign-first boundary handling. |
| `hdfs/wuyifan18_deeplog_preprocessed.toml` | Wuyifan18 preprocessed HDFS archive with inherited `hdfs_train` / `hdfs_test` boundaries. |
| `openstack/deeplog_preprocessed.toml` | OpenStack file-boundary reconstruction probe using recovered VM instance grouping and canonicalised message text. |
| `thunderbird.toml` | Thunderbird fixed-window benchmark slice. Main preset uses the corrected raw-position 100-log window contract. |
| `ait_ads/base.toml` | Combined AIT-ADS chronological alert stream. This is the preferred paper-compatible AIT-ADS contract. |
| `ait_ads/entity_chronological.toml` | AIT-ADS entity-local DeepCASE extension, split globally before entity grouping. |

Dataset-specific caveats and count audits belong in `experiments/reports/`, not
in this README. In particular, see:

- `experiments/reports/deeplog_reproduction_readiness.md`
- `experiments/reports/deepcase_reproduction_readiness.md`
- `experiments/reports/thunderbird_reproduction_readiness.md`
- `experiments/reports/thunderbird_slice_count_audit.md`
- `experiments/reports/baseline_sanity_report.md`

## Installation

From the repository root, install the experiment extras you need:

```bash
uv sync --extra experiments --extra deeplog
uv sync --extra experiments --extra deepcase
```

Or install all optional experiment dependencies:

```bash
uv sync --all-extras
```

## Running experiments

### Run one named registry experiment

```bash
uv run python -m experiments.runners.run_experiment \
  --experiment bgl_entity_chronological
```

### Run one model against one dataset manifest

```bash
uv run python -m experiments.runners.run_experiment \
  --config experiments/configs/datasets/hdfs/v1_deeplog_paper_entry100k_assign_first.toml \
  --model deeplog_default
```

### Run one resolved dataset/model variant from a registry-expanded run

```bash
uv run python -m experiments.runners.run_experiment \
  --experiment hdfs_wuyifan18_deepcase_table_iv_compat \
  --variant deeplog_short_session_padding_fidelity \
  --dataset hdfs/v1_deeplog_paper_entry100k_assign_first
```

### List the registry

```bash
uv run python -m experiments.runners.run_suite --list
```

The listing shows registry names, dataset configs, model names, and derived
reporting groups.

### Run a concrete registry experiment

```bash
uv run python -m experiments.runners.run_suite \
  --experiment bgl_deeplog_ccs2017_paper_10pct_entry_stream_no_online
```

### Run a family group

```bash
uv run python -m experiments.runners.run_suite \
  --group bgl_deeplog_ccs2017_paper \
  --group hdfs_deeplog_paper \
  --max-parallel 2
```

### Run Thunderbird entries

```bash
uv run python -m experiments.runners.run_suite --experiment thunderbird
```

The entity-grouped Thunderbird DeepCASE extension is separate:

```bash
uv run python -m experiments.runners.run_suite \
  --experiment thunderbird_entity_chronological
```

The smaller `thunderbird_smoke` preset is intended for local parser/template
checks and is not exposed as a public registry experiment.

### Preview or check missing runs

Use `--dry-run` to preview selected experiments without executing them:

```bash
uv run python -m experiments.runners.run_suite \
  --group bgl_deeplog_ccs2017_paper \
  --dry-run
```

Check which concrete registry runs still lack completed results:

```bash
uv run python -m experiments.runners.run_suite --check-missing
```

## Slurm backend

The Slurm backend uses the same registry names as the local runner.

Submit one experiment:

```bash
uv run python -m experiments.execution.slurm submit \
  --experiment bgl_deeplog_ccs2017_paper_10pct_entry_stream_no_online
```

Submit groups:

```bash
uv run python -m experiments.execution.slurm submit \
  --group bgl_deeplog_ccs2017_paper \
  --group hdfs_deeplog_paper
```

Use `--data-root` and `--cache-root` to place materialised datasets and caches
outside the repository tree. For example:

```bash
uv run python -m experiments.execution.slurm submit \
  --group hdfs_deeplog_paper \
  --data-root /data/hs1822 \
  --cache-root /data/hs1822/.cache
```

The generated Slurm wrapper disables Prefect telemetry by default with
`PREFECT_SERVER_ANALYTICS_ENABLED=false` and `DO_NOT_TRACK=1`, unless those
variables are already overridden in the submission environment.

Slurm array logs are written under:

```text
experiments/results/slurm-logs/<selection-label>/
```

## Useful runner flags

| Flag | Meaning |
| --- | --- |
| `--force` | Replace a completed deterministic output directory. |
| `--rerun` | Preserve earlier attempts by writing a fresh `attempt-XXXX/` under the same fingerprint root. |
| `--write-predictions` | Persist `predictions.jsonl` for scored test outputs. |
| `--debug-reporting` | Write fuller diagnostic metric payloads. Default reports keep paper-facing summaries lean. |
| `--dry-run` | Resolve and print the selected runs without executing them. |
| `--check-missing` | Report concrete registry runs without completed result artefacts. |
| `--max-parallel N` | Limit local suite concurrency. |

If a result directory exists but `metrics.json` is missing, the runner treats it
as stale and replaces it without requiring `--force`.

## Caching strategy

AnomaLog caches dataset preprocessing, not detector execution.

Cached stages include source materialisation, structured parsing, template
mining, derived raw-log construction, and dataset sidecars. Detector fitting and
test scoring are intentionally rerun for each resolved experiment fingerprint.

Concrete runs write deterministic output directories:

```text
experiments/results/<concrete-run-name>/<fingerprint>/
```

The fingerprint is derived from the resolved manifest, dataset contract, and
model config. Changing the dataset, sequence settings, or model config produces
a new fingerprint and therefore a new result directory.

Entity-grouped datasets also write split summaries and sidecars so repeated runs
can reuse chronology, entity counts, sparse inline labels, and other derived
preprocessing state without rescanning the full cache.

## Result artefacts

Each completed run writes:

| File | Contents |
| --- | --- |
| `experiment_config.json` | Normalised manifest, concrete override, dataset, model, and registry metadata. |
| `dataset_manifest.json` | Dataset fingerprint, source summary, raw-log hash, sequence settings, dataset statistics, and split summaries. |
| `metrics.json` | Task-aware metrics, primary metric scope, evaluation unit, split policy details, and scoped metric blocks. |
| `predictions.jsonl` | Optional scored test outputs when `--write-predictions` is supplied. |
| `environment.json` | Python, platform, package, git, and launch-command metadata. |
| `run.log` | Runtime logging from dataset build through detector evaluation. |
| `figure9_parameter_ci.json` | Publication-facing OpenStack parameter-branch summary, where applicable. |
| `figure9_parameter_ci_debug.json` | Verbose parameter trace, only with `--debug-reporting`. |

With `--rerun`, attempts are written to:

```text
experiments/results/<concrete-run-name>/<fingerprint>/attempt-XXXX/
```

## Audits and diagnostics

Run the DeepLog data-readiness audit with:

```bash
uv run python -m experiments.runners.audit_deeplog_data
```

Use the reports in `experiments/reports/` to interpret result lineages,
reproduction gaps, Thunderbird window contracts, DeepCASE workload definitions,
and baseline sanity checks.

## Adding experiments

### Add a preprocessing or dataset ablation

Create a new manifest under `configs/datasets/`. For built-in datasets, prefer
existing presets such as:

- `preset = "bgl"`
- `preset = "hdfs_v1"`
- `preset = "openstack_deeplog_preprocessed"`
- `ait_ads/base.toml` for the combined chronological AIT-ADS contract

For custom datasets, define `source`, `structured_parser`, optional
`label_reader`, and sequence settings directly in the manifest.

Use `[dataset.cache_paths] namespace = "..."` when a dataset needs portable,
dataset-scoped cache and data roots. Omit it to use the default platformdirs
cache/data locations.

### Share common manifest structure

Use:

```toml
extends = "base.toml"
```

Nested tables merge recursively, so a base manifest can hold shared boilerplate
while child manifests keep only scenario-specific overrides.

Use `[overrides]` for fixed adjustments and `[[axes]]` for Cartesian products
over fields such as `dataset.sequence.train_fraction`.

### Add a registry experiment

Add a table to `configs/registry.toml`:

```toml
[experiments.bgl_entity_chronological]
dataset = "bgl/entity_chronological"
model_sets = ["baselines_with_nb"]
models = [
  "deepcase",
  "deeplog_default",
]
```

For dataset families, use an experiment set:

```toml
[experiment_sets.ait_ads]
model_sets = ["baselines_no_nb"]
models = ["deeplog_default"]
datasets = ["ait_ads/base"]
```

### Add a detector

Add a tagged config subclass and detector subclass under `experiments/models/`
so the registries can resolve it like existing detectors.

## Environment note

The experiment layer intentionally shares the repository root environment rather
than defining its own `pyproject.toml`. This keeps dataset code, tests, docs,
and experiment runners locked and validated together.
