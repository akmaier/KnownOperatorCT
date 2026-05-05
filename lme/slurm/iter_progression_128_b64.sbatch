#!/usr/bin/env bash
#SBATCH --job-name=ct_iter_b64
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --nodelist=lme221
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171,lme222,lme223
#SBATCH --output=results/slurm/iter_prog_b64-%j.out
#SBATCH --error=results/slurm/iter_prog_b64-%j.err
#
# 128² FC progression with batch_size=64 to test whether averaging the
# gradient over more fresh phantoms per step (a) yields a visibly smoother
# loss curve and (b) helps FC escape the dead-ReLU plateau by reducing
# per-step gradient noise. Runs on lme221 (24 GB) — plenty for 128².
#
# Per-iter cost ~64x batch=4 cost (~1.9 s/iter), so 10k iters ≈ 5h30.
# Reuses the cached KO checkpoint from job 760170.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm results/sample_efficiency_128
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CHECKPOINTS="100,1000,10000"
KO_CKPT="results/checkpoints/cache/ko_ct_sample_efficiency_128_n2048_i5000.pt"
OUT_FILE="results/sample_efficiency_128/iter_progression_b64.png"

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID checkpoints=$CHECKPOINTS batch=64"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/render_iter_progression.py \
    --config configs/ct_sample_efficiency_128.yaml \
    --checkpoints "$CHECKPOINTS" \
    --ko-checkpoint "$KO_CKPT" \
    --batch-size 64 \
    --out "$OUT_FILE"

echo "[$(date '+%F %T')] done"
