#!/bin/bash -l
#
# SLURM job: Full reproduction pipeline on a 4x H100 node.
# Runs: surrogate (CPU) -> KO train+eval (1 GPU) -> FC train (4-GPU FSDP)
#     -> FC eval -> harvest results.
#
# Submit from the repo root:  sbatch slurm/h100_run_all.sh
#
#SBATCH --job-name=ct_full_pipeline
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=700G
#SBATCH --time=24:00:00
#SBATCH --output=results/h100_all_%j.out
#SBATCH --error=results/h100_all_%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

module load python

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
mkdir -p results

echo "=== Hardware ==="
nvidia-smi
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory / 1024**3:.0f} GB)')
"

echo ""
echo "=== Step 1: CPU Surrogate ==="
python src/run_surrogate.py --config configs/ct_surrogate.yaml
echo "Surrogate done."

echo ""
echo "=== Step 2: KO Train (single GPU) ==="
CUDA_VISIBLE_DEVICES=0 python src/ct_train.py \
    --config configs/ct_full_resolution.yaml --model known_operator
echo "KO training done."

echo ""
echo "=== Step 3: KO Eval ==="
CUDA_VISIBLE_DEVICES=0 python src/ct_eval.py \
    --config configs/ct_full_resolution.yaml --model known_operator
echo "KO eval done."

echo ""
echo "=== Step 4: FC Train (FSDP, 4x H100) ==="
torchrun --nproc_per_node=4 --master_port=29500 \
    src/ct_train_distributed.py \
    --config configs/ct_full_resolution_fc.yaml --model fully_connected
echo "FC training done."

echo ""
echo "=== Step 5: FC Eval ==="
CUDA_VISIBLE_DEVICES=0 python src/ct_eval.py \
    --config configs/ct_full_resolution_fc.yaml --model fully_connected
echo "FC eval done."

echo ""
echo "=== Step 6: Harvest ==="
python src/harvest_results.py --output results/RESULTS.md
echo "Harvest done."

echo ""
echo "=== Pipeline complete. Results in results/RESULTS.md ==="
