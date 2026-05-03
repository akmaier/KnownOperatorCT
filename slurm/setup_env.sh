#!/bin/bash -l
#
# One-time environment setup on RRZE Alex.
# Run this interactively (not via sbatch):
#   salloc --gres=gpu:a40:1 --time=0:30:00
#   bash slurm/setup_env.sh
#
set -euo pipefail

module load python

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT_DIR="$(pwd)"

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment ready at $ROOT_DIR/.venv"
echo "Verify GPU:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else 'No GPU')"
