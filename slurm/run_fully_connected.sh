#!/bin/bash -l
#
# SLURM job: Train and evaluate the Fully Connected model on Alex (A40).
# The FC model at 512x512 needs ~90 GB weights — too large even for A100 80GB.
# This script uses the reduced-resolution config (128x128) which fits on A40.
# Submit from the repo root:  sbatch slurm/run_fully_connected.sh
#
#SBATCH --job-name=fc_ct_train
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm_fc_%j.out
#SBATCH --error=results/slurm_fc_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
mkdir -p results

echo "=== Hardware ==="
nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"

echo "=== Training Fully Connected ==="
python src/ct_train.py --config configs/ct_fc_reduced.yaml --model fully_connected

echo "=== Evaluating Fully Connected ==="
python src/ct_eval.py --config configs/ct_fc_reduced.yaml --model fully_connected

echo "=== Done ==="
