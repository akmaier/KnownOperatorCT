#!/usr/bin/env bash
#SBATCH --job-name=ct_recons
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
# Need >=24 GB GPU because the 256x256 FC backward peaks at ~17 GB.
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme171,lme222,lme223
#SBATCH --output=results/slurm/recons-%j.out
#SBATCH --error=results/slurm/recons-%j.err
#
# Render two sample reconstructions for each sample-efficiency
# configuration. Trains KO and FC fresh at the largest N from each sweep
# config, plots phantom | KO recon | KO error | FC recon | FC error.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm results/sample_efficiency_128 results/sample_efficiency_256
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

echo
echo "=== 128x128 reconstructions (N=2048, 5000 iters) ==="
python src/render_reconstructions.py \
    --config configs/ct_sample_efficiency_128.yaml \
    --train-size 2048 --num-iterations 5000 --num-samples 2 \
    --out results/sample_efficiency_128/reconstructions.png

echo
echo "=== 256x256 reconstructions (N=1024, 1000 iters) ==="
python src/render_reconstructions.py \
    --config configs/ct_sample_efficiency_256.yaml \
    --train-size 1024 --num-iterations 1000 --num-samples 2 \
    --out results/sample_efficiency_256/reconstructions.png

echo "[$(date '+%F %T')] done"
