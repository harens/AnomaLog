#!/bin/bash
# Generated from slurm/jobs.toml; edit that manifest and regenerate.
#SBATCH --job-name=hdfs_v1_deeplog_paper_entry100k_assign_first
#SBATCH --output=slurm-hdfs_v1_deeplog_paper_entry100k_assign_first-%j.out
#SBATCH --error=slurm-hdfs_v1_deeplog_paper_entry100k_assign_first-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

export PATH="${HOME}/.local/bin:${PATH}"
export RUN_NAME="hdfs_v1_deeplog_paper_entry100k_assign_first"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PREFECT_ROOT="${PREFECT_ROOT:-${REPO_ROOT}/prefect}"
export PREFECT_HOME="${PREFECT_ROOT}/${RUN_NAME}"
export PREFECT_LOCAL_STORAGE_PATH="${PREFECT_HOME}/storage"
export PREFECT_API_DATABASE_CONNECTION_URL="sqlite+aiosqlite:///${PREFECT_HOME}/prefect.db"
# export PREFECT_TASKS_REFRESH_CACHE=true

mkdir -p "$PREFECT_HOME" "$PREFECT_LOCAL_STORAGE_PATH"

cd "$REPO_ROOT"

uv run python -m experiments.runners.run_experiment \
  --config "experiments/configs/sweeps/hdfs_v1_deeplog_paper_entry100k_assign_first.toml" \
  --force
