#!/bin/bash -l
#
# SLURM job: Train the Fully Connected model at FULL resolution (512x512)
# on Helma using 4x H100 with FSDP + CPU offload.
#
# FC model: 24 billion params, ~90 GB weights, ~360 GB Adam state.
# FSDP shards across 4 GPUs; optimizer state offloaded to CPU.
#
# Submit from the repo root:  sbatch slurm/helma_run_fc.sh
#
#SBATCH --job-name=fc_ct_fsdp
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=700G
#SBATCH --time=24:00:00
#SBATCH --output=results/helma_fc_%j.out
#SBATCH --error=results/helma_fc_%j.err
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
echo "=== Training FC model (512x512, FSDP, 4x H100) ==="
torchrun --nproc_per_node=4 --master_port=29500 \
    src/ct_train_distributed.py \
    --config configs/ct_full_resolution_fc.yaml \
    --model fully_connected

echo ""
echo "=== Evaluating FC model ==="
python src/ct_eval.py --config configs/ct_full_resolution_fc.yaml --model fully_connected

echo ""
echo "=== Done ==="
