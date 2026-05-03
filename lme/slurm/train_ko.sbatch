#!/usr/bin/env bash
#SBATCH --job-name=ct_train_ko
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=23:59:00
#SBATCH --signal=B:USR1@180
#SBATCH --exclude=lme49,lme171
#SBATCH --output=results/slurm/train_ko-%j.out
#SBATCH --error=results/slurm/train_ko-%j.err
#
# Train the Known Operator network at 512x512 / 180 views.
#
# Wall-time and checkpointing:
#   * --time=23:59:00   stays under the documented 24h soft limit.
#   * --signal=B:USR1@180  Slurm sends SIGUSR1 to this script 3 minutes before
#                          the wall-time limit. We forward it to python, which
#                          flushes a resume snapshot and exits cleanly.
#   * if training did not finish (no <model>.done marker), this script
#                          resubmits itself with afterok:<this job> until done
#                          (capped by LME_MAX_CHAIN, default 10).
#
# Submit from the repo root:  sbatch lme/slurm/train_ko.sbatch
set -uo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p results/slurm results/checkpoints
source .venv/bin/activate
source lme/slurm/_train_lib.sh

CHAIN_IDX="${LME_CHAIN_IDX:-1}"
DONE_MARKER="results/checkpoints/known_operator.done"

echo "[$(date '+%F %T')] node=$(hostname) job=$SLURM_JOB_ID chain=$CHAIN_IDX"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU {torch.cuda.get_device_name(0)}')"

python src/ct_train.py \
    --config configs/ct_full_resolution.yaml \
    --model known_operator &
PYTHON_PID=$!
wait_with_signal_forwarding "$PYTHON_PID"
PYTHON_RC=$?

echo "[$(date '+%F %T')] python exit=$PYTHON_RC"
maybe_resubmit "$DONE_MARKER" "$PYTHON_RC"
