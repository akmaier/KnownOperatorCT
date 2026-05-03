# Cloud Agent Instructions

This document is the contract between the human reviewer and the automated cloud agent that executes this reproduction bundle. The agent **must** follow these steps in order and **must** write all results into a single Markdown file at `results/RESULTS.md`. The reviewer will harvest only that file.

## 0. Hardware and operating system requirements

- Linux x86_64 (Ubuntu 22.04 LTS or later recommended)
- One NVIDIA GPU with at least 16 GB of memory; A100 40 GB or H100 preferred
- CUDA 12.x with matching NVIDIA driver
- At least 64 GB of system RAM
- At least 200 GB of free disk space

## 1. Bring up the environment

```bash
# from the unzipped bundle root
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If a conda environment is preferred, use `environment.yml` instead. Verify GPU access with `python -c "import torch; print(torch.cuda.is_available())"`. If this prints `False`, **stop** and report the failure in `results/RESULTS.md` under the section `## Environment failures`.

## 2. Run the full reproduction

The single entry point is `run_all.sh`. Invoke it with:

```bash
bash run_all.sh
```

This script in turn runs:

1. `python src/run_surrogate.py --config configs/ct_surrogate.yaml`
   CPU-scale surrogate. Writes `results/surrogate_results.json`, `results/surrogate_ablation.csv`, and `results/sample_efficiency.png`. Wall time: under 5 minutes on any modern CPU.

2. `python src/ct_train.py --config configs/ct_full_resolution.yaml --model known_operator`
   Full-resolution operator-aware CT training over a synthetic phantom population at $512 \times 512$ resolution with $180$ views. Writes `results/ct_known_operator_metrics.json` and per-fold checkpoints into `results/checkpoints/`. Wall time: $\sim 30$ to $\sim 90$ minutes on a single A100 depending on configuration.

3. `python src/ct_eval.py --config configs/ct_full_resolution.yaml --model known_operator`
   Evaluation on the held-out test split. Writes `results/ct_known_operator_eval.json`.

4. `python src/harvest_results.py --output results/RESULTS.md`
   Collects every number from the JSON and CSV artifacts above and writes the single Markdown report. **This is the file the reviewer reads.**

If any individual step fails, `run_all.sh` continues with the remaining steps so that partial results are still harvested. Failed steps are logged with a stderr trace to `results/run_all.log` and noted in `results/RESULTS.md`.

## 3. What `RESULTS.md` must contain

The harvest script populates the following sections automatically; the agent only needs to make sure the script runs.

1. `## Hardware`
   - GPU model, driver, CUDA version, PyTorch version, CPU model, RAM, disk free
2. `## CPU surrogate (paper Figure 3, Table 2)`
   - Per-training-size mean and standard deviation of test MSE for both models
   - Path to `results/sample_efficiency.png`
3. `## Full-resolution CT training (operator-aware)`
   - Configuration summary (image size, views, detector bins, training set size)
   - Per-fold training time, GPU peak memory, final training loss
4. `## Full-resolution CT evaluation (operator-aware)`
   - Test rRMSE per fold, mean, standard deviation
   - Comparison against the analytic baseline rRMSE reported by the script
5. `## Bound-inspired full-scale estimate (paper Table 1)`
   - Trainable parameter counts, relative estimation proxy, FP32 weight memory, Adam state memory, lower-bound runtimes
6. `## Run timings`
   - Wall time per stage of `run_all.sh`
7. `## Environment failures` (only if any)
   - One line per failure with the failing command and a pointer to the relevant log

## 4. Things the agent must not do

- Do not modify any file under `src/` or `configs/`. If the configuration is wrong for the available hardware, the agent should add a new file under `configs/` and pass it via `--config`.
- Do not delete `results/` after the run. The reviewer will retrieve the directory.
- Do not upload checkpoints or intermediate tensors to any third party. Only `results/RESULTS.md` is intended to be returned.
- Do not attempt to run the fully connected counterfactual at full resolution. The full-resolution FC counterfactual is a single learned dense matrix of shape $(N_{\text{pixels}}, N_{\text{measurements}})$ followed by a fixed ReLU, with $p_{\text{FC}} = N_{\text{pixels}} \cdot N_{\text{measurements}} \approx 2.42 \cdot 10^{10}$ parameters. Its memory footprint is documented as $\sim 90$ GB FP32 weights and $\sim 360$ GB Adam state and exceeds single-GPU budgets. The harvest script reports the deep risk estimator prediction for that case from the configuration; an actual run is not required.

## 5. Reporting back

When `run_all.sh` finishes, send the contents of `results/RESULTS.md` back to the reviewer. If `RESULTS.md` is missing, send `results/run_all.log`. Do not paraphrase, summarize, or filter the contents.
