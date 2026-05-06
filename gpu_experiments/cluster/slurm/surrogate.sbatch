#!/usr/bin/env bash
#SBATCH --job-name=ct_surrogate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=gpu_experiments/results/slurm/surrogate-%j.out
#SBATCH --error=gpu_experiments/results/slurm/surrogate-%j.err
#
# CPU-only surrogate. ~2 minutes on a recent CPU. No GPU requested so it
# starts immediately.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p gpu_experiments/results/slurm
source .venv/bin/activate

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID"
python src/run_surrogate.py --config configs/ct_surrogate.yaml
echo "[$(date '+%F %T')] surrogate done"
