# Slurm Wrappers

These `.sbatch` files are generated from [`jobs.toml`](./jobs.toml) so the
Slurm wrapper names stay in lock-step with the sweep configs they run.

Regenerate them from the repository root with:

```bash
uv run python -m experiments.slurm_scripts
```

Queue the whole set with:

```bash
bash slurm/submit_all.sh
```

Each generated wrapper uses the sweep config stem as both the Slurm job name
and the `RUN_NAME`, then launches `experiments.runners.run_experiment` against
the matching sweep file.

The wrappers resolve the repository root from `SLURM_SUBMIT_DIR` when the
job is launched through `sbatch`, and they point `uv` at `SLURM_TMPDIR` if it
is available. That avoids writing caches into the Slurm spool directory on
clusters where the generated script itself is staged elsewhere.
