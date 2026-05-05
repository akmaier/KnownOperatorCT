#!/usr/bin/env bash
#SBATCH --job-name=ridge_fc_128
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodelist=lme221
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171,lme222,lme223
#SBATCH --output=results/slurm/ridge_fc_128-%j.out
#SBATCH --error=results/slurm/ridge_fc_128-%j.err
#
# Closed-form ridge regression FC at 128x128.
# Sweeps N=4,16,64,256,1024,2048 with λ=1.0 (matches surrogate's central
# lambda from its grid). Reuses the cached KO baseline.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm results/sample_efficiency_128
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

KO_CKPT="results/checkpoints/cache/ko_ct_sample_efficiency_128_n2048_i5000.pt"

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/ridge_fc.py \
    --config configs/ct_sample_efficiency_128.yaml \
    --train-sizes 4,16,64,256,1024,2048 \
    --lambda 1.0 \
    --num-samples 2 \
    --ko-checkpoint "$KO_CKPT" \
    --out results/sample_efficiency_128/ridge_fc.png

echo "[$(date '+%F %T')] done"
