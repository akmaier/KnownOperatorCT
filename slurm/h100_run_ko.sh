#!/bin/bash -l
#
# SLURM job: Train and evaluate the Known Operator model on a 1x H100 node.
# Submit from the repo root:  sbatch slurm/h100_run_ko.sh
#
#SBATCH --job-name=ko_ct_train
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=results/h100_ko_%j.out
#SBATCH --error=results/h100_ko_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
mkdir -p results

echo "=== Hardware ==="
nvidia-smi
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "=== Training Known Operator (512x512, 180 views) ==="
python src/ct_train.py --config configs/ct_full_resolution.yaml --model known_operator

echo ""
echo "=== Evaluating Known Operator ==="
python src/ct_eval.py --config configs/ct_full_resolution.yaml --model known_operator

echo ""
echo "=== Done ==="
