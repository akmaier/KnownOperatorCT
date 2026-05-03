# Running on LME Lab Machines

## Machine → Experiment Matrix

The FC model at 256×256 needs ~17 GB total (5.6 GB weights + 5.6 GB Adagrad + 5.6 GB grads).
Machines with < 24 GB GPUs use FSDP to shard across multiple GPUs.

| Machine | GPUs | KO (512×512) | FC (256×256) | FC method |
|---------|------|:------------:|:------------:|-----------|
| **lme170/171** | 2× RTX 8000 (48 GB) | ✅ 1 GPU | ✅ 1 GPU | `run_fc.sh` |
| **lme49** | 1× A6000 (48 GB) | ✅ 1 GPU | ✅ 1 GPU | `run_fc.sh` |
| **lme221** | 4× RTX 6000 (24 GB) | ✅ 1 GPU | ✅ 1 GPU | `run_fc.sh` |
| **lme53** | 4× V100 (16 GB) | ✅ 1 GPU | ✅ 4-GPU FSDP | `run_fc_fsdp.sh` |
| **lme222/223** | 4× RTX 5000 (16 GB) | ✅ 1 GPU | ✅ 4-GPU FSDP | `run_fc_fsdp.sh` |
| **lme51/52** | 4× 1080 Ti (11 GB) | ✅ 1 GPU | ✅ 4-GPU FSDP | `run_fc_fsdp.sh` |
| **lme50** | 1× Titan XP + 3× 1080 Ti | ✅ 1 GPU | ⚠️ Titan only | `CUDA_VISIBLE_DEVICES=0 run_fc.sh` (Titan 12 GB, tight) |

For the full-resolution FC (512×512, 24B params), use Helma (see `slurm/` directory).

## Quick Start

```bash
# 1. SSH into a machine
ssh lme170

# 2. Clone and set up
cd /path/to/repo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run (pick a script)
bash lme/run_ko.sh                            # KO at 512×512, GPU 0
bash lme/run_fc.sh                            # FC at 256×256, single GPU (>= 24 GB)
bash lme/run_fc_fsdp.sh                       # FC at 256×256, multi-GPU FSDP (< 24 GB)
bash lme/run_all.sh                           # Full pipeline (auto-detects)
CUDA_VISIBLE_DEVICES=1 bash lme/run_ko.sh     # Use a different GPU
```

## Estimated Runtimes

| Experiment | 1080 Ti | V100 | RTX 6000/8000 | A6000 |
|-----------|---------|------|---------------|-------|
| Surrogate (CPU) | ~2 min | ~2 min | ~2 min | ~2 min |
| KO train (512×512, 10k iter) | ~60 min | ~30 min | ~30 min | ~20 min |
| FC train (256×256, 10k iter) | ~30 min | ~15 min | ~15 min | ~10 min |
