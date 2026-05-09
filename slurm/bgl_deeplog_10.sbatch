#!/bin/bash
#SBATCH --job-name=bgl-deeplog-10
#SBATCH --output=slurm-bgl-deeplog-10-%j.out
#SBATCH --error=slurm-bgl-deeplog-10-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

export PATH="/homes/hs1822/.local/bin:$PATH"
export RUN_NAME="bgl-deeplog-10"

export PREFECT_ROOT="/vol/gpudata/hs1822-anomalog/prefect"
export PREFECT_HOME="${PREFECT_ROOT}/${RUN_NAME}"
export PREFECT_LOCAL_STORAGE_PATH="${PREFECT_HOME}/storage"
export PREFECT_API_DATABASE_CONNECTION_URL="sqlite+aiosqlite:///${PREFECT_HOME}/prefect.db"
#export PREFECT_TASKS_REFRESH_CACHE=true

mkdir -p "$PREFECT_HOME" "$PREFECT_LOCAL_STORAGE_PATH"

cd /vol/gpudata/hs1822-anomalog/AnomaLog

uv run python -m experiments.runners.run_experiment \
  --config experiments/configs/sweeps/bgl_deeplog_paper_10pct_entry_stream_no_online.toml --force

