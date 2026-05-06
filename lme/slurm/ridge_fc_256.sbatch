#!/usr/bin/env bash
#SBATCH --job-name=ridge_fc_256
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# Bulk of the work lives on GPU; host RAM mostly holds Python + the
# .npz archive buffers (max ~150 MB). 8 GB is plenty.
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
# Need 48 GB GPU. Observed peak with linalg.solve workspace was 19.4 GB
# allocated + 5.6 GB more in the solve = 25 GB, OOMs on lme221's 24 GB.
# Pin to lme170 (RTX 8000 48 GB).
#SBATCH --nodelist=lme170
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme171,lme221,lme222,lme223
#SBATCH --output=results/slurm/ridge_fc_256-%j.out
#SBATCH --error=results/slurm/ridge_fc_256-%j.err
#
# Closed-form ridge regression FC at 256x256.
# Sweeps N=4,16,64,256,1024,2048 with λ=1.0. Trains and caches its own
# 256² KO baseline (no cache exists yet).
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm results/sample_efficiency_256
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

KO_CKPT="results/checkpoints/cache/ko_ct_sample_efficiency_256_n2048_i5000.pt"

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/ridge_fc.py \
    --config configs/ct_sample_efficiency_256.yaml \
    --train-sizes 4,16,64,256,1024,2048 \
    --lambdas 1e-4,1e-2,1.0,1e2,1e4 \
    --seeds 1,2,3 \
    --num-samples 2 \
    --num-test-stats 50 \
    --ko-checkpoint "$KO_CKPT" \
    --out results/sample_efficiency_256/ridge_fc.png

echo "[$(date '+%F %T')] done"
