#!/usr/bin/env bash
#SBATCH --job-name=ct_train_fc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --signal=B:USR1@180
# Need a single GPU with ~32 GB free for the FC model at 256x256: 5.6 GB
# weights + 5.6 GB Adagrad state + 5.6 GB gradients + activations. The 24 GB RTX 6000
# (24 GB RTX 6000) OOMs in the backward pass — observed in job 760105.
# So restrict to the 48 GB nodes only:
#   RTX 8000 and A6000 nodes.
# some lab nodes may be broken (driver/library mismatches, unknown
# state). In practice the FC single-GPU path lands on a 48 GB node only —
# usually busy. Use train_fc_fsdp.sbatch instead unless you specifically
# want a single-GPU run.
# #SBATCH --exclude=node1,node2,...   # uncomment + add your cluster's broken or small-GPU nodes
#SBATCH --output=results/slurm/train_fc-%j.out
#SBATCH --error=results/slurm/train_fc-%j.err
#
# Train the Fully Connected network at 256x256 / 90 views on a single big GPU.
# For nodes with smaller GPUs (V100, 1080 Ti, RTX 5000), use train_fc_fsdp.sbatch
# instead — it shards across all 4 GPUs of the node.
#
# Submit from the repo root:  sbatch cluster/slurm/train_fc.sbatch
set -uo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm results/checkpoints
source .venv/bin/activate
source cluster/slurm/_train_lib.sh

# Help the caching allocator avoid fragmentation (matches the FSDP wrapper).
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CHAIN_IDX="${CHAIN_IDX:-1}"
DONE_MARKER="results/checkpoints/fully_connected.done"

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID chain=$CHAIN_IDX"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(0)}')"

python src/ct_train.py \
    --config configs/ct_fc_lab.yaml \
    --model fully_connected &
PYTHON_PID=$!
wait_with_signal_forwarding "$PYTHON_PID"
PYTHON_RC=$?

echo "[$(date '+%F %T')] python exit=$PYTHON_RC"
maybe_resubmit "$DONE_MARKER" "$PYTHON_RC"
