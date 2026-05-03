#!/usr/bin/env bash
# One-time environment setup on the LME cluster.
#
# Run from the submit node (lme242) interactively after cloning the repo to
# /cluster/$(whoami)/known_operator_ct_release. This script grabs an
# interactive GPU allocation, builds a venv on /cluster (so it's visible to
# every compute node), installs the project requirements, and verifies torch
# can see a GPU.
#
# Usage (from cluster.i5.informatik.uni-erlangen.de):
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

echo "Building venv at $ROOT_DIR/.venv ..."
echo "(this needs network — only the submit node has it; running here.)"

# `python3 -m venv` is broken on lme242 because the system python3.10 ships
# without ensurepip. Prefer virtualenv (installed at /usr/bin/virtualenv);
# fall back to `venv --without-pip` + get-pip.py bootstrap if virtualenv is
# ever uninstalled.
if [ ! -d .venv ]; then
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
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt

echo
echo "Verifying torch + CUDA on a compute node (1 GPU, 5 min)..."
srun --gres=gpu:1 --time=5 --exclude=lme49 \
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
