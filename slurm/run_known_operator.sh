#!/bin/bash -l
#
# SLURM job: Train and evaluate the Known Operator model on Alex (A40).
# Submit from the repo root:  sbatch slurm/run_known_operator.sh
#
#SBATCH --job-name=ko_ct_train
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm_ko_%j.out
#SBATCH --error=results/slurm_ko_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
mkdir -p results

echo "=== Hardware ==="
nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"

echo "=== Training Known Operator ==="
python src/ct_train.py --config configs/ct_full_resolution.yaml --model known_operator

echo "=== Evaluating Known Operator ==="
python src/ct_eval.py --config configs/ct_full_resolution.yaml --model known_operator

echo "=== Done ==="
