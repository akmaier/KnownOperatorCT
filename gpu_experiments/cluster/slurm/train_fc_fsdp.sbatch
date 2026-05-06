#!/usr/bin/env bash
#SBATCH --job-name=ct_train_fc_fsdp
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:00
#SBATCH --signal=B:USR1@180
# Need 4 same-type GPUs on one node, each with >=24 GB. The FC FSDP path
# all-gathers 5.6 GB of weights for forward AND another 5.6 GB for backward;
# 16 GB nodes (V100, RTX 5000, 1080 Ti) don't fit even with CPU offload —
# observed in jobs 760108, 760111, 760114 on the configured nodes.
# The viable target is a 4-GPU 24 GB node (e.g. RTX 6000).
# some nodes are broken; some mix GPU types; 2-GPU nodes have <4 GPUs (filtered
# out by --gres=gpu:4 anyway) and are also broken.
# #SBATCH --exclude=node1,node2,...   # uncomment + add your cluster's broken or small-GPU nodes
#SBATCH --output=gpu_experiments/results/slurm/train_fc_fsdp-%j.out
#SBATCH --error=gpu_experiments/results/slurm/train_fc_fsdp-%j.err
#
# Train the Fully Connected network at 256x256 / 90 views with FSDP across
# 4 GPUs of one node. Use this on machines whose individual GPUs are < 24 GB
# (V100, 1080 Ti, RTX 5000).
#
# Submit from the repo root:  sbatch gpu_experiments/cluster/slurm/train_fc_fsdp.sbatch
set -uo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p gpu_experiments/results/slurm gpu_experiments/results/checkpoints
source .venv/bin/activate
source gpu_experiments/cluster/slurm/_train_lib.sh

# FSDP all-gathers the full 5.6 GB weight matrix on every rank during forward
# and again during backward. On 16 GB GPUs (V100, RTX 5000) PyTorch's caching
# allocator fragments and the second all-gather OOMs even though 4 GB is
# reserved-but-unallocated. expandable_segments=True lets the allocator reuse
# that fragmented space. Observed in job 760108 (4x RTX 5000).
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Force CPU offload of FSDP parameter shards even on 24 GB GPUs. Without
# offload, FSDP keeps the unsharded forward all-gather (5.6 GB) plus the
# backward all-gather (5.6 GB) plus per-rank shards in GPU memory at the
# same time — 20 GB allocated before the next op tries to grab another
# 5.6 GB and OOMs (observed in job 760117 on a node, 4x RTX 6000 24 GB).
# With offload the parameter shard lives on host and only the unsharded
# all-gather lives on GPU during fwd/bwd. Trades H2D bandwidth for headroom.
export FSDP_CPU_OFFLOAD=1

CHAIN_IDX="${CHAIN_IDX:-1}"
DONE_MARKER="gpu_experiments/results/checkpoints/fully_connected.done"

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID chain=$CHAIN_IDX ngpu=$NGPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

torchrun --nproc_per_node="$NGPU" --master_port=29500 \
    src/ct_train_distributed.py \
    --config configs/ct_fc_lab.yaml \
    --model fully_connected &
PYTHON_PID=$!
wait_with_signal_forwarding "$PYTHON_PID"
PYTHON_RC=$?

echo "[$(date '+%F %T')] torchrun exit=$PYTHON_RC"
maybe_resubmit "$DONE_MARKER" "$PYTHON_RC"
