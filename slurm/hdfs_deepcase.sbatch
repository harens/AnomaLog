#!/bin/bash
#SBATCH --job-name=hdfs-deepcase
#SBATCH --output=slurm-hdfs-deepcase--%j.out
#SBATCH --error=slurm-hdfs-deepcase-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

export PATH="/homes/hs1822/.local/bin:$PATH"
export RUN_NAME="hdfs-deepcase"

export PREFECT_ROOT="/vol/gpudata/hs1822-anomalog/prefect"
export PREFECT_HOME="${PREFECT_ROOT}/${RUN_NAME}"
export PREFECT_LOCAL_STORAGE_PATH="${PREFECT_HOME}/storage"
export PREFECT_API_DATABASE_CONNECTION_URL="sqlite+aiosqlite:///${PREFECT_HOME}/prefect.db"
#export PREFECT_TASKS_REFRESH_CACHE=true

mkdir -p "$PREFECT_HOME" "$PREFECT_LOCAL_STORAGE_PATH"

cd /vol/gpudata/hs1822-anomalog/AnomaLog

uv run python -m experiments.runners.run_experiment \
  --config experiments/configs/sweeps/hdfs_v1_deepcase.toml --force

