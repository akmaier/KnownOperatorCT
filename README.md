# Known Operator CT — Reproduction Bundle

This bundle reproduces the experiments of the NeurIPS submission *A Deep Risk Estimator for Known Operator Learning*. It contains a full-resolution GPU pipeline for the limited-angle CT reconstruction setting and a CPU fallback that runs the surrogate study from the paper.

The bundle is intended to be run **once** by an automated cloud agent on a Linux x86_64 host with an NVIDIA GPU. The agent should follow the instructions in [`docs/AGENT_INSTRUCTIONS.md`](docs/AGENT_INSTRUCTIONS.md) and write all results into a single Markdown file at `results/RESULTS.md`. That file is the authoritative artifact this bundle produces.

## Directory layout

```
known_operator_ct_release/
├── README.md                  this file
├── requirements.txt           pip dependencies for the CPU and GPU paths
├── environment.yml            optional conda environment
├── run_all.sh                 single entry point invoked by the agent
├── configs/
│   ├── ct_full_resolution.yaml   default 512x512 fan-beam config
│   └── ct_surrogate.yaml         CPU surrogate config for sanity checks
├── src/
│   ├── ct_dataset.py             slice-wise CT phantom generator
│   ├── ct_models.py              known operator and fully connected models
│   ├── ct_train.py               GPU training entry point
│   ├── ct_eval.py                evaluation and metrics
│   ├── run_surrogate.py          CPU surrogate (paper Figure 3)
│   └── harvest_results.py        collects all metrics into RESULTS.md
├── docs/
│   ├── AGENT_INSTRUCTIONS.md     step-by-step protocol for the cloud agent
│   └── HARDWARE_NOTES.md         expected memory and runtime requirements
└── results/                    written to by the agent (initially empty)
```

## Quick start for a human operator

If you want to run this bundle yourself rather than via the cloud agent:

```bash
# 1. set up the environment
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. run the full reproduction
bash run_all.sh
```

`run_all.sh` runs the surrogate, the full-resolution training, evaluation, and result harvesting. The end product is `results/RESULTS.md`, which contains every number reported in the paper plus runtime and hardware metadata.

## What is reproduced

1. **CPU-scale CT surrogate.** Test MSE versus training set size for the known operator and fully connected models. This corresponds to Figure 3 and Table 2 of the paper.
2. **Full-resolution GPU CT training.** Slice-wise $512 \times 512$ fan-beam reconstruction with $180$ views, using the operator-aware filtered backprojection network. Reports test rRMSE and per-fold timings.
3. **Bound-inspired full-scale estimate.** Sample-complexity proxy and parameter and runtime predictions from the deep risk estimator (Theorem 1 of the paper). Computed deterministically from the configuration.

The fully connected counterfactual at full resolution is **not trained** by default because its memory footprint exceeds typical single-GPU budgets ($\sim 90$ GB FP32 weights + $\sim 360$ GB Adam state). The bundle reports the bound-inspired prediction for that case and runs the dense baseline only at the surrogate scale.

## License of new assets

All code in `src/` and all configuration files are released under the MIT license; see `LICENSE`. Any third-party dependencies retain their respective licenses.
