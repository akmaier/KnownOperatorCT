# Known Operator CT — Reproduction Bundle

This bundle reproduces the experiments of the submission *A Deep Risk
Estimator for Known Operator Learning*. It is organised around two
reproduction paths:

* **CPU path (`cpu_experiments/`).** The surrogate sample-efficiency
  study at $H \in \{8, 16, 32\}$ that produces the paper's Figure 3.
  Pure NumPy/SciPy/Matplotlib, no GPU, no PyTorch. Finishes in well
  under a minute on a recent laptop. Self-contained: own `README.md`,
  `requirements.txt`, scripts, and `results/`.
* **GPU path (`src/`, `gpu_experiments/`, `run_all.sh`).** The
  full-resolution sample-efficiency sweeps at $H \in \{128, 256\}$ and
  the legacy single-host KO training/eval pipeline. PyTorch on one
  $\ge 24$ GB or $\ge 48$ GB NVIDIA GPU; Slurm submission scripts and
  multi-GPU FSDP variants are also bundled.

Output filenames in `cpu_experiments/` are all `cpu_*`-prefixed, so the
two paths can coexist in a single checkout without colliding. The
"What is reproduced" table further down maps each paper artifact to the
exact entry point and output file.

## Directory layout

```
known_operator_ct_release/
├── README.md                       this file
├── AGENTS.md                       protocol for automated agents
├── LICENSE                         MIT
├── requirements.txt                pip dependencies for the GPU path
├── environment.yml                 optional conda environment for the GPU path
├── run_all.sh                      single-host GPU pipeline entry point
├── configs/                        YAML configs for the GPU pipeline
├── src/                            GPU pipeline source (KO and FC models)
│   ├── ct_dataset.py                  slice-wise CT phantom generator
│   ├── ct_models.py                   known-operator and fully-connected models
│   ├── ct_train.py                    single-GPU training entry point
│   ├── ct_train_distributed.py        FSDP entry point for the FC counterfactual
│   ├── ct_eval.py                     evaluation and metrics
│   ├── fc_ko_sample_efficiency.py     unified KO+FC sample-efficiency sweep
│   ├── run_surrogate.py               legacy CPU surrogate (single H)
│   └── harvest_results.py             collects all metrics into RESULTS.md
├── docs/
│   ├── AGENT_INSTRUCTIONS.md       step-by-step protocol for a cloud agent
│   └── HARDWARE_NOTES.md           expected memory and runtime requirements
├── cpu_experiments/                CPU surrogate sweep at H ∈ {8, 16, 32}
│   ├── README.md                      reproduction instructions
│   ├── requirements.txt               numpy / scipy / matplotlib only
│   ├── cpu_surrogate_sweep.py         runs the three-H sweep
│   ├── cpu_combined_figure.py         regenerates the 1×3 paper figure
│   └── results/                       per-H JSON / CSV / PNG outputs
├── gpu_experiments/                full-resolution sample-efficiency sweeps
│   ├── cluster/                       portable Slurm wrapper scripts
│   ├── slurm/                         H100 batch scripts
│   └── results/                       sample_efficiency_{128,256}/
└── results/                        outputs of run_all.sh on a single host
```

## Setup

Two execution paths, two dependency sets. Use the one that matches what
you want to reproduce. They can coexist in separate virtual
environments without conflict.

### CPU path (lightweight, no GPU required)

Reproduces the Figure 3 sample-efficiency sweeps at $H \in \{8, 16, 32\}$.

```bash
cd cpu_experiments
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python3 cpu_surrogate_sweep.py     # ~17 s on a recent x86 laptop
python3 cpu_combined_figure.py     # writes results/cpu_sample_efficiency_combined.png
```

Total wall time well under one minute. See
[`cpu_experiments/README.md`](cpu_experiments/README.md) for the
parameter table and a description of the closed-form ridge fits.

### GPU path — environment

```bash
# from the repo root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Or, equivalently, with conda (pins `pytorch-cuda=12.1`):

```bash
conda env create -f environment.yml
conda activate known_operator_ct
```

Verify GPU access before launching anything heavy:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

### GPU path — sample-efficiency sweeps at $H \in \{128, 256\}$ (paper's main result)

The unified KO+FC sweeps fit both models on **identical** training
pools per $(N, \text{seed})$ cell using the script
`src/fc_ko_sample_efficiency.py`. They run equally well from a plain
shell on any single-GPU host or as Slurm batch jobs.

**Minimum hardware:**

| Sweep | GPU memory | Wall time (single GPU) | Pre-rendered artifacts |
|---|---|---|---|
| 128×128, $V=60$ | $\ge 24$ GB (e.g. RTX 6000) | $\sim 3$–$4$ h | `gpu_experiments/results/sample_efficiency_128/fc_ko_sweep.{png,json,npz}` |
| 256×256, $V=90$ | $\ge 48$ GB (e.g. RTX 8000 / A6000 / A100 / H100) | $\sim 2$–$2.5$ h | `gpu_experiments/results/sample_efficiency_256/fc_ko_sweep.{png,json,npz}` |

**Direct invocation (no Slurm), from the repo root:**

```bash
# 128x128 sweep
python src/fc_ko_sample_efficiency.py \
    --config configs/ct_sample_efficiency_128.yaml \
    --train-sizes 4,16,64,256,1024,2048 \
    --seeds 1,2,3 \
    --lambdas 1e-4,1e-2,1.0,1e2,1e4 \
    --ko-num-iterations 5000 \
    --num-test-stats 50 \
    --num-samples 2 \
    --num-save-recons 8 \
    --out gpu_experiments/results/sample_efficiency_128/fc_ko_sweep.png

# 256x256 sweep — same flags, swap config + output dir
python src/fc_ko_sample_efficiency.py \
    --config configs/ct_sample_efficiency_256.yaml \
    --train-sizes 4,16,64,256,1024,2048 \
    --seeds 1,2,3 \
    --lambdas 1e-4,1e-2,1.0,1e2,1e4 \
    --ko-num-iterations 5000 \
    --num-test-stats 50 \
    --num-samples 2 \
    --num-save-recons 8 \
    --out gpu_experiments/results/sample_efficiency_256/fc_ko_sweep.png
```

Each sweep writes a side-by-side reconstruction figure (`*.png`), the
full per-cell metrics in JSON, an NPZ archive that lets you
re-render figures without retraining, and a sample-efficiency curve.
Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the
256² run to avoid allocator fragmentation in the FC backward pass.

**Slurm submission (recommended for cluster users):**

```bash
# from the repo root, on a Slurm submit node
sbatch gpu_experiments/cluster/slurm/fc_ko_sweep_128.sbatch
sbatch gpu_experiments/cluster/slurm/fc_ko_sweep_256.sbatch
```

The scripts are written for a generic Slurm cluster; the
`#SBATCH --exclude=...` line is commented out and should be filled in
with any locally broken or undersized GPU nodes. Reference paths inside
the scripts assume the repo root as the working directory. See
[`gpu_experiments/cluster/README.md`](gpu_experiments/cluster/README.md)
for the full submission protocol (dependency-chained jobs, auto-resume
on 24-hour wall-time pre-emption, individual-job submission).

### GPU path — full-resolution single-host pipeline (`run_all.sh`)

`run_all.sh` is a six-step sequence with concrete commands you can also
run individually:

```bash
bash run_all.sh
```

is equivalent to:

```bash
python src/run_surrogate.py --config configs/ct_surrogate.yaml                              # 1. CPU surrogate (legacy single-H, ~2 min)
python src/ct_train.py     --config configs/ct_full_resolution.yaml --model known_operator  # 2. KO train at 512x512, 180 views (~30-90 min on A100)
python src/ct_eval.py      --config configs/ct_full_resolution.yaml --model known_operator  # 3. KO eval
python src/ct_train.py     --config configs/ct_full_resolution.yaml --model fully_connected # 4. FC train at 512x512 — see note below
python src/ct_eval.py      --config configs/ct_full_resolution.yaml --model fully_connected # 5. FC eval
python src/harvest_results.py --output results/RESULTS.md                                   # 6. aggregate everything into RESULTS.md
```

Steps run in their own subshells; a failure of one does not stop the
others. Each step's wall time is logged to `results/run_all.log` and
`results/run_all_steps.csv`.

> **Note on step 4.** The FC counterfactual at the full $512 \times 512$
> resolution is a $\approx 2.42 \cdot 10^{10}$-parameter dense matrix
> ($\sim 90$ GB FP32 weights, $\sim 360$ GB Adam state) and will OOM on
> any single GPU. The step is intentionally allowed to fail; the
> harvester records the failure in `RESULTS.md` and reports the
> bound-inspired prediction for that case from the configuration. To
> actually train the full-resolution FC, use the FSDP pipeline in
> [`gpu_experiments/cluster/`](gpu_experiments/cluster/) on $\ge 4 \times$
> H100 with CPU offload — `bash gpu_experiments/cluster/run_fc_fsdp.sh`
> on a lab node, or `sbatch gpu_experiments/cluster/slurm/train_fc_fsdp.sbatch`
> on Slurm.

The end product is `results/RESULTS.md`. See
[`docs/AGENT_INSTRUCTIONS.md`](docs/AGENT_INSTRUCTIONS.md) for the
cloud-agent contract describing the exact section structure of
`RESULTS.md`.

## What is reproduced

| Paper artifact | Experiment | Hardware | Entry point | Output |
|---|---|---|---|---|
| Figure 3 (CPU surrogate sweeps) | $H \in \{8, 16, 32\}$, closed-form ridge, 5 seeds | any laptop CPU | `cpu_experiments/cpu_surrogate_sweep.py` then `cpu_experiments/cpu_combined_figure.py` | `cpu_experiments/results/cpu_sample_efficiency_combined.png` |
| Sample-efficiency curves at 128×128 | unified KO+FC sweep on identical pools | $\ge 24$ GB GPU | `src/fc_ko_sample_efficiency.py --config configs/ct_sample_efficiency_128.yaml …` | `gpu_experiments/results/sample_efficiency_128/fc_ko_sweep.{png,json,npz}` |
| Sample-efficiency curves at 256×256 | unified KO+FC sweep on identical pools | $\ge 48$ GB GPU | `src/fc_ko_sample_efficiency.py --config configs/ct_sample_efficiency_256.yaml …` | `gpu_experiments/results/sample_efficiency_256/fc_ko_sweep.{png,json,npz}` |
| Full-resolution KO training/eval | $512 \times 512$, $V = 180$ | $\ge 16$ GB GPU | `bash run_all.sh` | `results/ct_known_operator_metrics.json`, `results/ct_known_operator_eval.json`, `results/RESULTS.md` |
| Bound-inspired Table 1 estimate | sample-complexity proxy, parameter / runtime predictions | none (deterministic) | `python src/harvest_results.py --output results/RESULTS.md` | section "Bound-inspired full-scale estimate" in `results/RESULTS.md` |
| Full-resolution FC counterfactual | $\approx 2.42 \cdot 10^{10}$ dense parameters | $\ge 4 \times$ H100 with CPU offload | `bash gpu_experiments/cluster/run_fc_fsdp.sh` | optional; bound-inspired prediction is reported either way |

A more discursive description follows.

1. **CPU-scale surrogate sweeps (Figure 3).** Test MSE versus training
   set size for the known-operator and fully-connected models at three
   image scales $H \in \{8, 16, 32\}$. Closed-form ridge fits, five
   seeds per cell, validation-selected $\lambda$. Lives entirely in
   `cpu_experiments/`.
2. **Full-resolution sample-efficiency sweeps.** Slice-wise fan-beam
   reconstruction at $H \in \{128, 256\}$, $V = 1.25\,H$ views. Both
   models are fit on **identical** training pools per $(N, \text{seed})$
   cell. KO is trained with Adagrad SGD and FC with closed-form ridge.
   Reports test rRMSE per fold with mean ± std. Implementation lives in
   `src/fc_ko_sample_efficiency.py`; reference outputs and Slurm
   wrappers live in `gpu_experiments/`.
3. **Bound-inspired full-scale estimate.** Sample-complexity proxy and
   parameter / runtime predictions from the deep risk estimator
   (Theorem 1 of the paper). Computed deterministically from the
   configuration; emitted by `src/harvest_results.py`.

## Verifying that reproduction succeeded

* **CPU surrogate.** `cpu_experiments/results/cpu_multiH_summary.json`
  reports a per-$H$ ratio `sigma_FC / sigma_KO`. Reproduction is
  consistent with the paper if this ratio grows monotonically with $H$.
* **128×128 / 256×256 sweeps.** Open
  `gpu_experiments/results/sample_efficiency_{128,256}/fc_ko_sweep.png`.
  At 128² the KO and FC ridge curves cross around $N \in [1024, 2048]$;
  at 256² the FC curve never crosses the KO curve within the budget.
  Quantitative comparison against the pre-rendered JSON is the
  authoritative check.
* **Full-resolution KO.** Open `results/RESULTS.md`. The
  "Full-resolution CT evaluation (operator-aware)" section reports test
  rRMSE per fold; reproduction is consistent if it improves over the
  analytic baseline rRMSE printed alongside it.

The fully-connected counterfactual collapses the entire reconstruction
pipeline into a single learned dense matrix $M$ that maps projections
directly to image pixels, followed by a fixed ReLU:
$\hat{y}_{\text{FC}} = \text{ReLU}(M\,x)$. At the full
$512 \times 512$ resolution it is **not trained** by default because
its memory footprint exceeds typical single-GPU budgets
($p_{\text{FC}} = N_{\text{pixels}} \cdot N_{\text{measurements}}
\approx 2.42 \cdot 10^{10}$ parameters, $\sim 90$ GB FP32 weights
$+ \sim 360$ GB Adam state). The bundle reports the bound-inspired
prediction for that case and runs the dense baseline only at the
sweep scales described above.

## License of new assets

All code in `src/`, `cpu_experiments/`, and `gpu_experiments/`, and all
configuration files are released under the MIT license; see `LICENSE`.
Any third-party dependencies retain their respective licenses.

## Anonymity

This bundle is prepared for double-blind peer review. It contains no
author names, institution names, hostnames, absolute filesystem paths,
or proprietary cluster identifiers. Synthetic phantoms only; no patient
data are referenced.
