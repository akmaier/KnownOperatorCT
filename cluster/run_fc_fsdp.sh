#!/usr/bin/env bash
# Train the FC model at 256x256 using FSDP across all available GPUs.
# Use this on machines with < 24 GB per GPU (1080 Ti, V100, RTX 5000).
#
# Usage:
#   bash cluster/run_fc_fsdp.sh              # All GPUs
#   CUDA_VISIBLE_DEVICES=0,1 bash cluster/run_fc_fsdp.sh  # Specific GPUs
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
source .venv/bin/activate

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "=== Using $NGPU GPUs with FSDP ==="
python -c "
import torch
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name} ({props.total_memory/1024**3:.1f} GB)')
"

echo ""
echo "=== Training FC (256x256, FSDP, ${NGPU} GPUs) ==="
torchrun --nproc_per_node="$NGPU" --master_port=29500 \
    src/ct_train_distributed.py \
    --config configs/ct_fc_lab.yaml --model fully_connected

echo ""
echo "=== Evaluating FC ==="
CUDA_VISIBLE_DEVICES=0 python src/ct_eval.py \
    --config configs/ct_fc_lab.yaml --model fully_connected

echo ""
echo "=== Done. Results in results/ ==="
