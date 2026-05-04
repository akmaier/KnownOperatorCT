#!/usr/bin/env bash
# One-time environment setup on a lab Slurm cluster.
#
# Run from the submit node interactively after cloning the repo to
# /cluster/$(whoami)/known_operator_ct_release. This script grabs an
# interactive GPU allocation, builds a venv on /cluster (so it's visible to
# every compute node), installs the project requirements, and verifies torch
# can see a GPU.
#
# Usage (from the cluster's submit node):
#   cd /cluster/$(whoami)/known_operator_ct_release
#   bash lme/slurm/setup.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT_DIR="$(pwd)"

case "$ROOT_DIR" in
    /cluster/*) ;;
    *)
        echo "WARNING: repo is not on /cluster ($ROOT_DIR)."
        echo "         Compute nodes mount /cluster but not /home for big I/O."
        echo "         Move the repo to /cluster/$(whoami)/ before running jobs."
        ;;
esac

# Pre-create the directories Slurm needs to write into. Slurm opens
# --output / --error paths *before* our sbatch script gets to run, so any
# in-script `mkdir -p results/slurm` is too late on the first run.
mkdir -p results/slurm results/checkpoints

echo "Building venv at $ROOT_DIR/.venv ..."
echo "(this needs network — only the submit node has it; running here.)"

# `python3 -m venv` may be broken on the submit node if the system python
# ships without ensurepip. Prefer virtualenv (often at /usr/bin/virtualenv);
# fall back to `venv --without-pip` + get-pip.py bootstrap if virtualenv is
# ever uninstalled.
#
# Trigger on the absence of activate (not the .venv dir) so a previous
# half-built env doesn't trick us into skipping creation.
if [ ! -e .venv/bin/activate ]; then
    if [ -d .venv ]; then
        echo "Removing incomplete .venv from a previous attempt ..."
        rm -rf .venv
    fi
    if command -v virtualenv >/dev/null 2>&1; then
        virtualenv --python=python3 .venv
    elif python3 -c "import ensurepip" >/dev/null 2>&1; then
        python3 -m venv .venv
    else
        echo "Bootstrapping pip into a --without-pip venv ..."
        python3 -m venv --without-pip .venv
        # shellcheck disable=SC1091
        source .venv/bin/activate
        curl -sS https://bootstrap.pypa.io/get-pip.py | python
        deactivate
    fi
fi

if [ ! -e .venv/bin/activate ]; then
    echo "ERROR: venv creation failed; .venv/bin/activate not present" >&2
    ls -la .venv 2>/dev/null || true
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel

# The default PyPI torch wheel is built against CUDA 13, which needs an NVIDIA
# driver >= 545. The LME compute nodes are on driver 535.x (CUDA 12.x), so we
# install the cu121 wheels explicitly. If a cu130 (or other non-cu121) torch
# is already in the venv from an earlier setup run, uninstall it first so pip
# can downgrade — `pip install --upgrade` alone won't move backwards in
# version, and unpinned --upgrade won't replace the installed wheel.
INSTALLED_TORCH=$(pip show torch 2>/dev/null | awk '/^Version:/ {print $2}')
case "$INSTALLED_TORCH" in
    *+cu121) echo "torch $INSTALLED_TORCH already cu121; skipping reinstall." ;;
    "")      echo "torch not installed yet." ;;
    *)
        echo "Uninstalling torch $INSTALLED_TORCH (need cu121 build for driver 535.x)..."
        pip uninstall -y torch torchvision || true
        # Drop the cu13 helper packages dragged in by the cu130 wheel — they're
        # unused once torch is gone and just waste space on the shared FS.
        pip uninstall -y \
            cuda-toolkit cuda-bindings cuda-pathfinder \
            nvidia-cudnn-cu13 nvidia-cusparselt-cu13 nvidia-nccl-cu13 \
            nvidia-nvshmem-cu13 nvidia-cufft nvidia-cublas nvidia-cusparse \
            nvidia-nvtx nvidia-cufile nvidia-nvjitlink nvidia-cuda-cupti \
            nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cusolver nvidia-curand \
            triton 2>/dev/null || true
        ;;
esac

if [ "${INSTALLED_TORCH}" != "${INSTALLED_TORCH%+cu121}" ] && [ -n "$INSTALLED_TORCH" ]; then
    : # cu121 already
else
    pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision
fi

# Remaining deps from PyPI; pip leaves cu121 torch in place because it already
# satisfies `torch>=2.2`.
pip install -r requirements.txt

echo
echo "Verifying torch + CUDA on a compute node (1 GPU, 5 min)..."
srun --gres=gpu:1 --time=5 --exclude=lme49,lme171 \
    bash -c "source $ROOT_DIR/.venv/bin/activate && python -c '
import torch
print(f\"PyTorch {torch.__version__}\")
print(f\"CUDA available: {torch.cuda.is_available()}\")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f\"  GPU {i}: {p.name} ({p.total_memory/1024**3:.1f} GB)\")
'"

echo
echo "Setup complete. Submit jobs with:"
echo "  bash lme/slurm/submit_chain.sh"
