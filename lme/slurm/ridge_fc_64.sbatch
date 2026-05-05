#!/usr/bin/env bash
#SBATCH --job-name=ridge_fc_64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --exclude=lme49,lme171
#SBATCH --output=results/slurm/ridge_fc_64-%j.out
#SBATCH --error=results/slurm/ridge_fc_64-%j.err
#
# Closed-form ridge regression FC at 64x64. p_FC = 7.86M, fits anywhere.
# Sweeps N=4,16,64,256,1024,2048 with λ=1.0. Trains its own KO baseline
# (no cache exists for 64²) and persists it for downstream reuse.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm results/sample_efficiency_64
source .venv/bin/activate

KO_CKPT="results/checkpoints/cache/ko_ct_sample_efficiency_64_n2048_i5000.pt"

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/ridge_fc.py \
    --config configs/ct_sample_efficiency_64.yaml \
    --train-sizes 4,16,64,256,1024,2048 \
    --lambdas 1e-4,1e-2,1.0,1e2,1e4 \
    --seeds 1,2,3 \
    --num-samples 2 \
    --num-test-stats 50 \
    --ko-checkpoint "$KO_CKPT" \
    --out results/sample_efficiency_64/ridge_fc.png

echo "[$(date '+%F %T')] done"
