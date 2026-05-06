#!/usr/bin/env bash
# Full reproduction pipeline on lab GPU machines.
# Runs surrogate (CPU), KO (512x512 GPU), FC (256x256 GPU), harvest.
# Auto-detects GPU memory and uses FSDP for FC if needed.
#
# Usage:
#   bash cluster/run_all.sh
#   CUDA_VISIBLE_DEVICES=0 bash cluster/run_all.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate
mkdir -p results

echo "=== GPUs ==="
python -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory/1024**3:.1f} GB)')
"

GPU_MEM_GB=$(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}')")
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")

echo ""
echo "=== Step 1: CPU Surrogate ==="
python src/run_surrogate.py --config configs/ct_surrogate.yaml
echo "Surrogate done."

echo ""
echo "=== Step 2: KO Train (512x512) ==="
CUDA_VISIBLE_DEVICES=0 python src/ct_train.py \
    --config configs/ct_full_resolution.yaml --model known_operator
echo "KO training done."

echo ""
echo "=== Step 3: KO Eval ==="
CUDA_VISIBLE_DEVICES=0 python src/ct_eval.py \
    --config configs/ct_full_resolution.yaml --model known_operator
echo "KO eval done."

echo ""
if [ "$GPU_MEM_GB" -ge 24 ]; then
    echo "=== Step 4: FC Train (256x256, single GPU — ${GPU_MEM_GB} GB available) ==="
    CUDA_VISIBLE_DEVICES=0 python src/ct_train.py \
        --config configs/ct_fc_lab.yaml --model fully_connected
else
    echo "=== Step 4: FC Train (256x256, FSDP on ${NGPU} GPUs — ${GPU_MEM_GB} GB/GPU) ==="
    torchrun --nproc_per_node="$NGPU" --master_port=29500 \
        src/ct_train_distributed.py \
        --config configs/ct_fc_lab.yaml --model fully_connected
fi
echo "FC training done."

echo ""
echo "=== Step 5: FC Eval ==="
CUDA_VISIBLE_DEVICES=0 python src/ct_eval.py \
    --config configs/ct_fc_lab.yaml --model fully_connected
echo "FC eval done."

echo ""
echo "=== Step 6: Harvest ==="
python src/harvest_results.py --output results/RESULTS.md
echo "Harvest done."

echo ""
echo "=== Pipeline complete. Results in results/RESULTS.md ==="
