# Experiments

The `experiments/` layer sits after AnomaLog preprocessing. Use it when the
core library has already given you the sequence view you want and you now want
reproducible detector runs, model sweeps, or preprocessing ablations recorded
as artifacts on disk.

If you want the module-level API reference for the separate `experiments`
package, see [Reference](reference/experiments.md).

## :material-sign-direction: When to use it

Use the core `anomalog` API when you are:

- defining or modifying dataset preprocessing
- inspecting parsed records, templates, sequences, and representations
- prototyping sequence construction in a notebook or script

Use `experiments/` when you are:

- comparing multiple detector configurations on the same sequence setup
- comparing multiple preprocessing variants with the same detector
- writing reproducible run artifacts for later analysis

This layer is intentionally separate from the public `anomalog` API.
`anomalog` is the reusable library surface for building datasets, sequences, and
representations. `experiments/` is a repository-local workflow for
configuration-driven runs, checked-in baselines, and artifact recording.

If you want to inspect the implementation directly, see the
[`experiments/` directory on GitHub](https://github.com/harens/AnomaLog/tree/main/experiments).

## :material-layers-outline: Mental model

The experiment layer separates three concerns:

1. A dataset config defines preprocessing and sequence generation.
2. A model config defines the detector family and its hyperparameters.
3. A run config binds one dataset config to one model config.

That keeps preprocessing decisions explicit and reusable while making model
experimentation easy to repeat.

## :material-file-document-outline: Dataset configs

Dataset configs answer the questions a researcher normally asks before running
a detector:

- which dataset or preset am I using?
- how are the logs parsed and templated?
- how are the sequences grouped?
- what train/test split semantics am I using?

For built-in benchmarks, a dataset config is usually just a thin wrapper around
a preset plus a sequence policy:

```toml
name = "bgl_entity_supervised"
dataset_name = "BGL"
preset = "bgl"

[sequence]
grouping = "entity"
train_fraction = 0.8
train_on_normal_entities_only = false
```

## :material-tune-variant: Model configs

Model configs describe how a detector consumes the sequence data produced by
the dataset config.

For example, a phrase-based Naive Bayes model config records both the detector
hyperparameters and the phrase representation settings:

```toml
detector = "naive_bayes"
name = "naive_bayes_default"
smoothing = 1.0
phrase_ngram_min = 1
phrase_ngram_max = 2
top_k_phrases = 5
anomalous_posterior_threshold = 0.5
```

## :material-play-circle-outline: Run configs

A run config combines one dataset config with one model config:

```toml
name = "bgl_naive_bayes"
dataset = "bgl_entity_supervised"
model = "naive_bayes_default"
results_root = "experiments/results"
```

Run it from the repository root with:

```bash
uv run python -m experiments.runners.run_experiment \
  --config experiments/configs/runs/bgl_naive_bayes.toml
```

Install the optional experiment dependencies first when you want to run the
checked-in baselines:

```bash
uv sync --extra experiments --extra river
uv sync --extra experiments --extra deeplog
uv sync --extra experiments --extra deepcase
```

Use `uv sync --all-extras` if you want every detector family available at once.

## :material-database-clock-outline: Caching and reruns

The experiment layer reuses AnomaLog's dataset-side caches, but it does not
cache detector training or detector scoring as separate reusable stages.

- Dataset sourcing, structured parsing, template mining, and other
  preprocessing work can be reused through the existing AnomaLog and
  Prefect-backed caches when their inputs have not changed.
- Each experiment run writes to a deterministic result directory under
  `experiments/results/<run-name>/<fingerprint>/`.
- The fingerprint is derived from the fully resolved run, dataset, and model
  config, so changing any of those inputs produces a new output directory.
- Re-running the exact same config targets the same result directory. Use
  `--force` when you want to replace that existing output.
- If you change the experiment config, the detector is intentionally retrained
  and the test split is rescored for the new fingerprint.

## :material-file-chart-outline: Artifacts

Each run writes a deterministic result directory containing:

- `run_config.json` with the normalsed dataset, model, and run config
- `dataset_manifest.json` with preprocessing provenance and sequence metadata
- `metrics.json` with detector metrics
- `predictions.jsonl` with per-sequence outputs
- `environment.json` with Python, platform, and git metadata
- `run.log` with run-time logging

The important point is not the file names themselves. It is that the
preprocessing choices, sequence settings, model settings, and outputs are all
recorded together, so later analysis does not depend on memory or notebook
state.

## :material-flask-outline: Current built-in detectors

The checked-in examples currently cover:

- `template_frequency` for a simple template-frequency baseline
- `naive_bayes` for a phrase-based classifier
- `river` for online-learning style baselines backed by [`river`](https://riverml.xyz/)
- `deeplog` for the scoped DeepLog implementation with stacked-LSTM
  next-log-key prediction and per-template parameter-value models
- `deepcase` for the official DeepCase Context Builder and Interpreter workflow
  adapted to entity-grouped AnomaLog sequences

Those models consume AnomaLog sequence representations rather than defining
their own preprocessing path.
