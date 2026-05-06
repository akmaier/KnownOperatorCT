# Known Operator CT — Reproduction Bundle

This bundle reproduces the experiments of the submission *A Deep Risk
Estimator for Known Operator Learning*. It is organised around two
self-contained reproduction paths:

* **`cpu_experiments/`** — the surrogate sample-efficiency study at
  $H \in \{8, 16, 32\}$ that produces the paper's Figure 3. Pure
  NumPy/SciPy/Matplotlib, no GPU, no PyTorch. Finishes in well under a
  minute on a recent laptop.
* **`gpu_experiments/`** — the full-resolution sample-efficiency sweeps
  at $H \in \{128, 256\}$ that produce the paper's main quantitative
  results. Single-GPU PyTorch (A100/H100 recommended) or the bundled
  Slurm submission scripts on a multi-node cluster.

Each subfolder is a fully self-contained reproduction unit with its own
`README.md`, dependencies, scripts, and `results/` directory. They share
no input or output filenames.

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

### GPU path (single host, full-resolution)

Reproduces the full-resolution training and evaluation on a single
NVIDIA A100/H100.

```bash
# from the repo root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

bash run_all.sh
```

`run_all.sh` runs the legacy single-H surrogate, the full-resolution
known-operator training, evaluation, and result harvesting. The end
product is `results/RESULTS.md`. See
[`docs/AGENT_INSTRUCTIONS.md`](docs/AGENT_INSTRUCTIONS.md) for the
cloud-agent protocol.

If you prefer conda over venv, use the equivalent
`environment.yml` (which pins `pytorch-cuda=12.1`):

```bash
conda env create -f environment.yml
conda activate known_operator_ct
bash run_all.sh
```

### GPU path (Slurm cluster, sample-efficiency sweeps)

Reproduces the unified KO+FC sample-efficiency sweeps at $H \in \{128, 256\}$
on a multi-GPU cluster. The submission scripts live under
[`gpu_experiments/`](gpu_experiments/):

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
for the full submission protocol, dependency-chained jobs, and resume
behaviour.

## What is reproduced

1. **CPU-scale surrogate sweeps (Figure 3).** Test MSE versus training
   set size for the known-operator and fully-connected models at three
   image scales $H \in \{8, 16, 32\}$. Closed-form ridge fits, five
   seeds per cell, validation-selected $\lambda$. Lives entirely in
   `cpu_experiments/`.
2. **Full-resolution sample-efficiency sweeps.** Slice-wise fan-beam
   reconstruction at $H \in \{128, 256\}$, $V = 1.25\,H$ views. Both
   models are fit on **identical** training pools per $(N, \text{seed})$
   cell. KO is trained with Adagrad SGD and FC with closed-form ridge.
   Reports test rRMSE per fold with mean ± std. Lives in
   `gpu_experiments/`.
3. **Bound-inspired full-scale estimate.** Sample-complexity proxy and
   parameter / runtime predictions from the deep risk estimator
   (Theorem 1 of the paper). Computed deterministically from the
   configuration; emitted by `src/harvest_results.py`.

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
