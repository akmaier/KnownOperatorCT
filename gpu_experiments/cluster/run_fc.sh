#!/usr/bin/env bash
# Train and evaluate the Fully Connected model at 256x256 resolution.
# Fits on any lab GPU machine with a GPU >= 8 GB.
#
# Usage:
#   bash cluster/run_fc.sh                        # GPU 0
#   CUDA_VISIBLE_DEVICES=1 bash cluster/run_fc.sh # GPU 1
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate

echo "=== GPU ==="
python -c "import torch; print(f'{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)')"

echo ""
echo "=== Training Fully Connected (256x256, 90 views) ==="
python src/ct_train.py --config configs/ct_fc_lab.yaml --model fully_connected

echo ""
echo "=== Evaluating Fully Connected ==="
python src/ct_eval.py --config configs/ct_fc_lab.yaml --model fully_connected

echo ""
echo "=== Done. Results in results/ ==="
