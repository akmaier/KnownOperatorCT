#!/usr/bin/env bash
#SBATCH --job-name=ct_iter_prog
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodelist=lme170
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme171,lme221,lme222,lme223
#SBATCH --output=results/slurm/iter_prog_128-%j.out
#SBATCH --error=results/slurm/iter_prog_128-%j.err
#
# 128² FC training progression: snapshots reconstructions at 100, 1k, 10k, 50k
# iters (online fresh data), then renders side-by-side figure + loss curve.
# KO is loaded from cache when available.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm results/sample_efficiency_128
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CHECKPOINTS="${CHECKPOINTS:-100,1000,10000,50000}"
KO_CKPT="${KO_CKPT:-results/checkpoints/cache/ko_ct_sample_efficiency_128_n2048_i5000.pt}"
OUT_FILE="${OUT_FILE:-results/sample_efficiency_128/iter_progression.png}"

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID checkpoints=$CHECKPOINTS out=$OUT_FILE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/render_iter_progression.py \
    --config configs/ct_sample_efficiency_128.yaml \
    --checkpoints "$CHECKPOINTS" \
    --ko-checkpoint "$KO_CKPT" \
    --out "$OUT_FILE"

echo "[$(date '+%F %T')] done"
