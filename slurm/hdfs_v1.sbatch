#!/bin/bash
#SBATCH --job-name=hdfs_v1
#SBATCH --output=slurm-hdfs-%j.out
#SBATCH --error=slurm-hdfs-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

export PATH="/homes/hs1822/.local/bin:$PATH"

cd /vol/gpudata/hs1822-anomalog/AnomaLog

uv run python -m experiments.runners.run_experiment \
  --config experiments/configs/sweeps/hdfs_v1.toml
