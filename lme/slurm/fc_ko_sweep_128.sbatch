#!/usr/bin/env bash
#SBATCH --job-name=fc_ko_128
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --nodelist=lme221
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme170,lme171,lme222,lme223
#SBATCH --output=results/slurm/fc_ko_128-%j.out
#SBATCH --error=results/slurm/fc_ko_128-%j.err
#
# Unified KO + FC ridge sweep at 128x128 on 24 GB GPU (lme221).
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm "results/sample_efficiency_128"
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/fc_ko_sample_efficiency.py \
    --config configs/ct_sample_efficiency_128.yaml \
    --train-sizes 4,16,64,256,1024,2048 \
    --seeds 1,2,3 \
    --lambdas 1e-4,1e-2,1.0,1e2,1e4 \
    --ko-num-iterations 5000 \
    --num-test-stats 50 \
    --num-samples 2 \
    --num-save-recons 8 \
    --out results/sample_efficiency_128/fc_ko_sweep.png

echo "[$(date '+%F %T')] done"
