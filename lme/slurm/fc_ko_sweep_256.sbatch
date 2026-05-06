#!/usr/bin/env bash
#SBATCH --job-name=fc_ko_256
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --nodelist=lme170
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme171,lme221,lme222,lme223
#SBATCH --output=results/slurm/fc_ko_256-%j.out
#SBATCH --error=results/slurm/fc_ko_256-%j.err
#
# Unified KO + FC ridge sweep at 256x256 on 48 GB GPU (lme170).
# Heavy: 18 cells x (KO 5000 iter + FC 5 lambdas). KO ~5 min/cell on
# RTX 8000, FC ridge solve ~5 sec/cell. Pool gen at N=2048 ~90s.
# Total per-cell ~6-8 min, 18 cells ~2-2.5 h.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm "results/sample_efficiency_256"
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python src/fc_ko_sample_efficiency.py \
    --config configs/ct_sample_efficiency_256.yaml \
    --train-sizes 4,16,64,256,1024,2048 \
    --seeds 1,2,3 \
    --lambdas 1e-4,1e-2,1.0,1e2,1e4 \
    --ko-num-iterations 5000 \
    --num-test-stats 50 \
    --num-samples 2 \
    --num-save-recons 8 \
    --out results/sample_efficiency_256/fc_ko_sweep.png

echo "[$(date '+%F %T')] done"
