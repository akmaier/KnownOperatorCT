#!/bin/bash -l
#
# SLURM job: Full pipeline on Alex (A100 80GB recommended).
# Runs surrogate (CPU), KO train+eval (GPU), FC train+eval (reduced res, GPU),
# then harvests all results.
# Submit from the repo root:  sbatch slurm/run_all_alex.sh
#
#SBATCH --job-name=ct_full_pipeline
#SBATCH --gres=gpu:a100:1
#SBATCH -C a100_80
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=results/slurm_all_%j.out
#SBATCH --error=results/slurm_all_%j.err
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
echo "=== Step 1: CPU Surrogate ==="
python src/run_surrogate.py --config configs/ct_surrogate.yaml

echo ""
echo "=== Step 2: KO Train ==="
python src/ct_train.py --config configs/ct_full_resolution.yaml --model known_operator

echo ""
echo "=== Step 3: KO Eval ==="
python src/ct_eval.py --config configs/ct_full_resolution.yaml --model known_operator

echo ""
echo "=== Step 4: FC Train (reduced resolution) ==="
python src/ct_train.py --config configs/ct_fc_reduced.yaml --model fully_connected

echo ""
echo "=== Step 5: FC Eval (reduced resolution) ==="
python src/ct_eval.py --config configs/ct_fc_reduced.yaml --model fully_connected

echo ""
echo "=== Step 6: Harvest ==="
python src/harvest_results.py --output results/RESULTS.md

echo ""
echo "=== Pipeline complete. Results in results/RESULTS.md ==="
