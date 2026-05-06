# CPU experiments — *A Deep Risk Estimator for Known Operator Learning*

This folder contains the CPU-only release of the surrogate experiments
reported in the paper. It is designed to be merged with a separately-
released GPU bundle: every Python file and every output filename in this
folder is prefixed with `cpu_` so that no collision occurs when the two
folders are concatenated under a common parent.

## Contents

```
cpu_experiments/
├── README.md                       (this file)
├── requirements.txt                pip dependencies
├── cpu_surrogate_sweep.py          runs H ∈ {8, 16, 32} surrogate sweeps
├── cpu_combined_figure.py          regenerates Figure 3 from the JSON results
└── results/
    ├── cpu_results_H8.json         per-seed test MSE + aggregate at H = 8
    ├── cpu_results_H16.json        same at H = 16
    ├── cpu_results_H32.json        same at H = 32
    ├── cpu_ablation_H{H}.csv       CSV form of each aggregate
    ├── cpu_multiH_summary.json     cross-H calibration summary
    └── cpu_sample_efficiency_combined.png
                                    1×3 panel paper figure (Figure 3)
```

The `results/` directory is empty until the sweep is run.

## Reproducing the experiments

The experiments are pure-CPU NumPy / SciPy and finish in well under a
minute on a modern laptop.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python3 cpu_surrogate_sweep.py     # ~17 s on a recent x86 laptop
python3 cpu_combined_figure.py     # produces results/cpu_sample_efficiency_combined.png
```

## What the experiments cover

Three image scales:

| `H` | `V` | `B` | trainable parameters (KO) | trainable parameters (FC) |
|----:|----:|----:|--------------------------:|--------------------------:|
| 8   | 10  | 8   | 80          | 5,120          |
| 16  | 20  | 16  | 320         | 81,920         |
| 32  | 40  | 32  | 1,280       | 1,310,720      |

Training-set sizes `N ∈ {4, 8, 16, 32, 64}` (the `N = 128` point used in
older drafts is intentionally dropped here because the closed-form ridge
solution overfits the fully-connected baseline at that point and
pollutes the bound calibration). Five random seeds per `(H, N)` cell.

Each cell is fit in closed form:

* **Known-operator (KO):** `y_hat = B · diag(w) · x`, where `B = (Aᵀ A
  + 0.1 I)⁻¹ Aᵀ` is a fixed Tikhonov-regularised analytic inverse and
  `w` is the trainable diagonal projection-domain weighting.
* **Fully connected (FC):** `y_hat = M · x` for a single dense matrix
  `M`.

Both fits are ridge-regularised closed-form least-squares with the
regularisation coefficient chosen on a held-out validation set from
`{1e-6, 1e-4, 1e-2, 1, 1e2}`.

## Naming convention (for merging with the GPU bundle)

Every file produced by this folder is prefixed with `cpu_`. The GPU
bundle is expected to use names without that prefix (e.g. `gpu_*` or
plain `ct_*`), so concatenating the two release folders under a common
parent never overwrites anything.

Specifically:

* Source: `cpu_surrogate_sweep.py`, `cpu_combined_figure.py`.
* Outputs: `cpu_results_H{H}.json`, `cpu_ablation_H{H}.csv`,
  `cpu_multiH_summary.json`, `cpu_sample_efficiency_combined.png`.

If the GPU bundle uses the same prefix, rename one of the two before
merging.

## Anonymity

The bundle contains no author names, no institutional names, no
hostnames or paths. Synthetic phantoms only; no patient data are
referenced.
