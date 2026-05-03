#!/usr/bin/env bash
#SBATCH --job-name=ct_train_fc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --signal=B:USR1@180
# Need a single GPU with >=24 GB for the FC model at 256x256.
# That restricts us to lme170/lme171 (RTX 8000, 48 GB) and lme221 (RTX 6000, 24 GB).
#SBATCH --exclude=lme49,lme50,lme51,lme52,lme53,lme171,lme222,lme223
#SBATCH --output=results/slurm/train_fc-%j.out
#SBATCH --error=results/slurm/train_fc-%j.err
#
# Train the Fully Connected network at 256x256 / 90 views on a single big GPU.
# For nodes with smaller GPUs (V100, 1080 Ti, RTX 5000), use train_fc_fsdp.sbatch
# instead — it shards across all 4 GPUs of the node.
#
# Submit from the repo root:  sbatch lme/slurm/train_fc.sbatch
set -uo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm results/checkpoints
source .venv/bin/activate
source lme/slurm/_train_lib.sh

CHAIN_IDX="${LME_CHAIN_IDX:-1}"
DONE_MARKER="results/checkpoints/fully_connected.done"

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID chain=$CHAIN_IDX"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(0)}')"

python src/ct_train.py \
    --config configs/ct_fc_lme.yaml \
    --model fully_connected &
PYTHON_PID=$!
wait_with_signal_forwarding "$PYTHON_PID"
PYTHON_RC=$?

echo "[$(date '+%F %T')] python exit=$PYTHON_RC"
maybe_resubmit "$DONE_MARKER" "$PYTHON_RC"
