#!/usr/bin/env bash
#SBATCH --job-name=ct_sweep_64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=lme49,lme171
#SBATCH --output=results/slurm/sweep_64-%j.out
#SBATCH --error=results/slurm/sweep_64-%j.err
#
# 64x64 sample-efficiency sweep — gives us the KO-vs-N curve we don't
# have yet (we already have FC ridge at 64x64).
# Grid: 6 N-values x 3 seeds x 2 models = 36 cells. Per-cell at 64x64 is
# fast (~1-2 min), so the whole sweep finishes in < 1 hour.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm "results/sample_efficiency_64"
source .venv/bin/activate

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/sample_efficiency_gpu.py \
    --config configs/ct_sample_efficiency_64.yaml
echo "[$(date '+%F %T')] sweep done"
