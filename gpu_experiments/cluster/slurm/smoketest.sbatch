#!/usr/bin/env bash
#SBATCH --job-name=ct_smoketest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
# #SBATCH --exclude=node1,node2,...   # uncomment + add your cluster's broken or small-GPU nodes
#SBATCH --output=gpu_experiments/results/slurm/smoketest-%j.out
#SBATCH --error=gpu_experiments/results/slurm/smoketest-%j.err
#
# 30-second sanity check: GPU visible, torch+CUDA happy, a matmul runs.
# Submit from the repo root:  sbatch gpu_experiments/cluster/slurm/smoketest.sbatch
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p gpu_experiments/results/slurm
source .venv/bin/activate

echo "=== node ==="
hostname
echo
echo "=== nvidia-smi ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || echo "(nvidia-smi unavailable)"
echo
echo "=== torch ==="
python - <<'PY'
import time, torch
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("no CUDA, smoketest failed")

dev = torch.device("cuda")
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name} ({p.total_memory/1024**3:.1f} GB, cc {p.major}.{p.minor})")

x = torch.randn(2048, 2048, device=dev)
torch.cuda.synchronize()
t0 = time.perf_counter()
y = x @ x
torch.cuda.synchronize()
dt = (time.perf_counter() - t0) * 1000
print(f"matmul 2048x2048: {dt:.1f} ms, |y|_F={y.norm().item():.2f}")

# Touch the project imports so we know src/ deps resolve too.
import sys
sys.path.insert(0, "src")
from ct_dataset import FanBeamGeometry
geom = FanBeamGeometry(image_size=64, num_views=30, detector_bins=64,
                       angular_range_degrees=180.0, source_to_iso_mm=1000.0,
                       source_to_detector_mm=1500.0, detector_pixel_mm=1.0)
print(f"FanBeamGeometry import ok: {geom}")
print("OK")
PY
