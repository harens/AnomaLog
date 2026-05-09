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
