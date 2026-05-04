#!/bin/bash -l
#
# One-time environment setup on an H100 cluster.
# Run interactively:
#   salloc --gres=gpu:h100:1 --partition=h100 --time=0:30:00
#   bash slurm/h100_setup.sh
#
set -euo pipefail

module load python

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT_DIR="$(pwd)"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Environment ready at $ROOT_DIR/.venv"
echo "Verify GPU:"
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory / 1024**3:.0f} GB)')
"
