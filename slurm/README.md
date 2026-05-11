# Slurm Wrappers

These `.sbatch` files are generated from [`jobs.toml`](./jobs.toml) so the
Slurm wrapper names stay in lock-step with the dataset manifests they run.

Regenerate them from the repository root with:

```bash
uv run python -m experiments.slurm_scripts
```

Queue the whole set with:

```bash
bash slurm/submit_all.sh
```

Each generated wrapper uses the dataset-manifest stem as both the Slurm job
name and the `RUN_NAME`, then launches `experiments.runners.run_experiment`
against the matching manifest file.

When a dataset manifest contains a mix of heavyweight and lightweight models,
the runner batches entries by their `run_group` value so the expensive models
do not share a worker pool with the lighter baselines.

The wrappers resolve the repository root from `SLURM_SUBMIT_DIR` when the
job is launched through `sbatch`, and they point `uv` at `SLURM_TMPDIR` if it
is available. That avoids writing caches into the Slurm spool directory on
clusters where the generated script itself is staged elsewhere.
