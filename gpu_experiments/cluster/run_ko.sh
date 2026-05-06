#!/usr/bin/env bash
# Train and evaluate the Known Operator model at full resolution.
# Works on any lab GPU machine with a GPU >= 11 GB.
#
# Usage:
#   bash cluster/run_ko.sh                        # GPU 0
#   CUDA_VISIBLE_DEVICES=2 bash cluster/run_ko.sh # GPU 2
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source .venv/bin/activate

echo "=== GPU ==="
python -c "import torch; print(f'{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)')"

echo ""
echo "=== Training Known Operator (512x512, 180 views) ==="
python src/ct_train.py --config configs/ct_full_resolution.yaml --model known_operator

echo ""
echo "=== Evaluating Known Operator ==="
python src/ct_eval.py --config configs/ct_full_resolution.yaml --model known_operator

echo ""
echo "=== Done. Results in results/ ==="
