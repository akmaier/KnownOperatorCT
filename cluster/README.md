# Running on Lab GPU Machines and a Slurm Cluster

There are two ways to run on the lab hardware:

1. **Direct SSH to a lab machine** (`run_*.sh` here) — quick iteration, no
   queue. Use during development.
2. **Slurm batch jobs** (`slurm/*.sbatch` plus `slurm/submit_chain.sh`) —
   the right way to run the full reproduction: it queues, schedules, and
   (importantly) auto-resumes from checkpoints if a job hits the 24-hour
   wall-time limit. See [`slurm/`](slurm/) for the wrappers.

## GPU memory → Experiment matrix

The FC model at 256×256 needs ~17 GB total (5.6 GB weights + 5.6 GB Adagrad + 5.6 GB grads).
Machines with < 24 GB GPUs use FSDP to shard across multiple GPUs.

| GPU class | KO (512×512) | FC (256×256) | FC method |
|---|:---:|:---:|---|
| 48 GB (RTX 8000 / A6000) | ✅ 1 GPU | ✅ 1 GPU | `run_fc.sh` |
| 24 GB (RTX 6000) | ✅ 1 GPU | ❌ OOM in backward | use FSDP instead |
| 4 × 16 GB (V100 / RTX 5000) | ✅ 1 GPU | ✅ 4-GPU FSDP | `run_fc_fsdp.sh` |
| 4 × 11 GB (1080 Ti) | ✅ 1 GPU | ✅ 4-GPU FSDP | `run_fc_fsdp.sh` |

> **Note on 24 GB cards.** The single-GPU FC at 256×256 fits the weights
> and Adagrad state but PyTorch needs another ~5.6 GB for gradients and
> ~3 GB for backward-pass activations on top of that — the total
> exceeds 24 GB and OOMs in practice. On 24 GB hardware, run FC via
> FSDP instead (`run_fc_fsdp.sh` or `cluster/slurm/train_fc_fsdp.sbatch`).

For the full-resolution FC (512×512, 24B params), use the H100 cluster scripts in `slurm/`.

## Quick Start

```bash
# 1. SSH into a lab machine
ssh <hostname>

# 2. Clone and set up
cd /path/to/repo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Run (pick a script)
bash cluster/run_ko.sh                            # KO at 512×512, GPU 0
bash cluster/run_fc.sh                            # FC at 256×256, single GPU (>= 24 GB)
bash cluster/run_fc_fsdp.sh                       # FC at 256×256, multi-GPU FSDP (< 24 GB)
bash cluster/run_all.sh                           # Full pipeline (auto-detects)
CUDA_VISIBLE_DEVICES=1 bash cluster/run_ko.sh     # Use a different GPU
```

## Estimated Runtimes

| Experiment | 1080 Ti | V100 | RTX 6000/8000 | A6000 |
|-----------|---------|------|---------------|-------|
| Surrogate (CPU) | ~2 min | ~2 min | ~2 min | ~2 min |
| KO train (512×512, 10k iter) | ~60 min | ~30 min | ~30 min | ~20 min |
| FC train (256×256, 10k iter) | ~30 min | ~15 min | ~15 min | ~10 min |

## Running on the lab Slurm cluster

The cluster has a soft 24-hour wall-time limit per job. Our trainers now
checkpoint every `training.checkpoint_every` iterations (see configs) and
catch SIGTERM / SIGUSR1 cleanly, so a job killed at the time limit leaves a
resume snapshot at `results/checkpoints/<model>.resume.pt`. The sbatch
wrappers under [`slurm/`](slurm/) auto-resubmit with `--dependency=afterok`
until each model writes its `<model>.done` sentinel.

### One-time setup

```bash
# from your laptop
ssh <user>@<submit-node>

# on the submit node:
mkdir -p /cluster/$(whoami) && cd /cluster/$(whoami)
git clone <repo-url> known_operator_ct_release
cd known_operator_ct_release
bash cluster/slurm/setup.sh        # builds .venv on /cluster, verifies torch+CUDA on a compute node
```

`setup.sh` insists the repo lives on `/cluster/<user>` because compute nodes
mount `/cluster` (and `/scratch` for transient data) but heavy I/O against
`/home` slows every user down.

### Submit the full pipeline

```bash
cd /cluster/$(whoami)/known_operator_ct_release

# default: FC trains via 4-GPU FSDP (any 4-GPU node with >= 11 GB per GPU)
bash cluster/slurm/submit_chain.sh

# opt-in: FC trains on a single 48 GB GPU. Usually pends on any 48 GB
# node unless one is free.
bash cluster/slurm/submit_chain.sh --single
```

The chain is:
`surrogate → {ko_train → ko_eval, fc_train → fc_eval} → harvest`. Each
edge is an `afterok` dependency, so a real failure halts downstream work
instead of running on stale state.

### Watching jobs

```bash
squeue -u $(whoami)                    # your jobs
sinfo -h -o "%n %T %G"                 # node + GPU types per node
tail -f results/slurm/train_ko-*.out   # live log of the latest KO step
```

### What gets written

* `results/checkpoints/<model>.resume.pt` — periodic snapshot, removed on completion
* `results/checkpoints/<model>.pt`        — final weights (eval reads this)
* `results/checkpoints/<model>.done`      — empty sentinel signalling completion
* `results/ct_<model>_metrics.json`       — full training metrics
* `results/ct_<model>_eval.json`          — test-set metrics
* `results/RESULTS.md`                    — final harvest report
* `results/slurm/<job>-<id>.out|err`      — per-job stdout / stderr

### Individual jobs

If you'd rather submit pieces by hand:

```bash
sbatch cluster/slurm/surrogate.sbatch
sbatch cluster/slurm/train_ko.sbatch
sbatch --export=ALL,MODEL=known_operator,CONFIG=configs/ct_full_resolution.yaml cluster/slurm/eval.sbatch
sbatch cluster/slurm/train_fc.sbatch          # or: cluster/slurm/train_fc_fsdp.sbatch
sbatch --export=ALL,MODEL=fully_connected,CONFIG=configs/ct_fc_lab.yaml cluster/slurm/eval.sbatch
sbatch cluster/slurm/harvest.sbatch
```

Override the resubmission cap with `RESUBMIT_MAX`:

```bash
sbatch --export=ALL,RESUBMIT_MAX=20 cluster/slurm/train_ko.sbatch
```
