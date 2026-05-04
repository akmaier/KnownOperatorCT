#!/usr/bin/env bash
#SBATCH --job-name=ct_sweep_128
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --exclude=lme49,lme171
#SBATCH --output=results/slurm/sweep_128-%j.out
#SBATCH --error=results/slurm/sweep_128-%j.err
#
# 128x128 sample-efficiency sweep, 5000 iterations per cell.
# Grid: 6 N-values x 3 seeds x 2 models = 36 cells.
#
# Submit from the repo root:  sbatch lme/slurm/sample_efficiency_128.sbatch
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm "results/sample_efficiency_128"
source .venv/bin/activate

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(0)}')"

python src/sample_efficiency_gpu.py \
    --config configs/ct_sample_efficiency_128.yaml
echo "[$(date '+%F %T')] sweep done"
