#!/usr/bin/env bash
#SBATCH --job-name=ct_train_fc_fsdp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:00
#SBATCH --signal=B:USR1@180
# Want 4 same-type GPUs on one node. lme49 is broken; lme50 mixes Titan + 1080 Ti.
# lme170/lme171 only have 2 GPUs and are filtered out by --gres=gpu:4 anyway.
#SBATCH --exclude=lme49,lme50
#SBATCH --output=results/slurm/train_fc_fsdp-%j.out
#SBATCH --error=results/slurm/train_fc_fsdp-%j.err
#
# Train the Fully Connected network at 256x256 / 90 views with FSDP across
# 4 GPUs of one node. Use this on machines whose individual GPUs are < 24 GB
# (V100, 1080 Ti, RTX 5000).
#
# Submit from the repo root:  sbatch lme/slurm/train_fc_fsdp.sbatch
set -uo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm results/checkpoints
source .venv/bin/activate
source lme/slurm/_train_lib.sh

CHAIN_IDX="${LME_CHAIN_IDX:-1}"
DONE_MARKER="results/checkpoints/fully_connected.done"

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID chain=$CHAIN_IDX ngpu=$NGPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

torchrun --nproc_per_node="$NGPU" --master_port=29500 \
    src/ct_train_distributed.py \
    --config configs/ct_fc_lme.yaml \
    --model fully_connected &
PYTHON_PID=$!
wait_with_signal_forwarding "$PYTHON_PID"
PYTHON_RC=$?

echo "[$(date '+%F %T')] torchrun exit=$PYTHON_RC"
maybe_resubmit "$DONE_MARKER" "$PYTHON_RC"
