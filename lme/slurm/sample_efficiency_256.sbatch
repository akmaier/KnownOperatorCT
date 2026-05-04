#!/usr/bin/env bash
#SBATCH --job-name=ct_sweep_256
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
# Need a >=24 GB GPU because single-GPU FC at 256x256 wants ~17 GB
# (5.6 weights + 5.6 Adagrad + 5.6 grads peak during backward).
# That restricts to lme170/lme171 (RTX 8000 48 GB) and lme221 (RTX 6000
# 24 GB). lme49 and lme171 are broken.
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme171,lme222,lme223
#SBATCH --output=results/slurm/sweep_256-%j.out
#SBATCH --error=results/slurm/sweep_256-%j.err
#
# 256x256 sample-efficiency sweep, 1000 iterations per cell.
# Grid: 5 N-values x 2 seeds x 2 models = 20 cells.
#
# Submit from the repo root:  sbatch lme/slurm/sample_efficiency_256.sbatch
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm "results/sample_efficiency_256"
source .venv/bin/activate

# Defragment the caching allocator so the backward gradient buffer fits
# alongside the weights + optimizer state on a 24 GB card.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(0)}')"

python src/sample_efficiency_gpu.py \
    --config configs/ct_sample_efficiency_256.yaml
echo "[$(date '+%F %T')] sweep done"
