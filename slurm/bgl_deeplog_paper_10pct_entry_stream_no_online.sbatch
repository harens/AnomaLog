#!/bin/bash
# Generated from slurm/jobs.toml; edit that manifest and regenerate.
#SBATCH --job-name=bgl_deeplog_paper_10pct_entry_stream_no_online
#SBATCH --output=slurm-bgl_deeplog_paper_10pct_entry_stream_no_online-%j.out
#SBATCH --error=slurm-bgl_deeplog_paper_10pct_entry_stream_no_online-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

export PATH="${HOME}/.local/bin:${PATH}"
export RUN_NAME="bgl_deeplog_paper_10pct_entry_stream_no_online"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

export PREFECT_ROOT="${PREFECT_ROOT:-${REPO_ROOT}/prefect}"
export PREFECT_HOME="${PREFECT_ROOT}/${RUN_NAME}"
export PREFECT_LOCAL_STORAGE_PATH="${PREFECT_HOME}/storage"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-${REPO_ROOT}/.cache/uv}}"
export PREFECT_API_DATABASE_CONNECTION_URL="sqlite+aiosqlite:///${PREFECT_HOME}/prefect.db"
# export PREFECT_TASKS_REFRESH_CACHE=true

mkdir -p "$PREFECT_HOME" "$PREFECT_LOCAL_STORAGE_PATH" "$UV_CACHE_DIR"

cd "$REPO_ROOT"

uv run python -m experiments.runners.run_experiment \
  --config "experiments/configs/sweeps/bgl_deeplog_paper_10pct_entry_stream_no_online.toml" \
  --force
