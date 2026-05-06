# AGENTS.md

## Cursor Cloud specific instructions

This is a single-service Python scientific computing project (no web servers, databases, or external services). The bundle is organised into two self-contained reproduction paths:

* `cpu_experiments/` — pure-CPU NumPy/SciPy surrogate sweeps (Figure 3, $H \in \{8, 16, 32\}$). Self-contained; its own `README.md` and `requirements.txt`.
* `gpu_experiments/` and the top-level `src/` — the full-resolution PyTorch pipeline and the multi-GPU sample-efficiency sweeps at $H \in \{128, 256\}$.

See `README.md` and `docs/AGENT_INSTRUCTIONS.md` for the full protocol.

### Environment

The two paths use disjoint dependency sets and should live in disjoint virtual environments.

- **CPU path:** lightweight venv inside `cpu_experiments/` using `cpu_experiments/requirements.txt` (numpy, scipy, matplotlib only — no torch).

  ```bash
  cd cpu_experiments
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

- **GPU path:** top-level venv at `.venv/` using the top-level `requirements.txt` (numpy, scipy, matplotlib, pyyaml, torch, torchvision, tqdm).

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  ```

  No GPU is available in the Cloud Agent VM. The CPU surrogate experiment runs fine without one; the full-resolution CT training/eval can run on CPU with a small config (see `configs/ct_test_small.yaml`).

### Running the project

#### CPU experiments (paper Figure 3)

```bash
cd cpu_experiments
source venv/bin/activate
python3 cpu_surrogate_sweep.py     # ~17 s; writes results/cpu_results_H{8,16,32}.json + CSVs + cpu_multiH_summary.json
python3 cpu_combined_figure.py     # writes results/cpu_sample_efficiency_combined.png
```

Total wall time well under one minute.

#### GPU experiments (full-resolution training and eval)

- **Smoke test (CPU surrogate, top-level legacy path):** `python src/run_surrogate.py --config configs/ct_surrogate.yaml` — takes ~1-2 minutes, produces `results/surrogate_results.json`, `results/surrogate_ablation.csv`, `results/sample_efficiency.png`.
- **Small-scale GPU code validation on CPU:** Use `configs/ct_test_small.yaml` (32x32, 20 views) with `python src/ct_train.py --config configs/ct_test_small.yaml --model known_operator` and `--model fully_connected`. Takes seconds on CPU.
- **Harvest results:** `python src/harvest_results.py --output results/RESULTS.md` — aggregates all artifacts into a single Markdown report. Supports both KO and FC model results.
- **Full pipeline:** `bash run_all.sh` — runs surrogate, KO train/eval, FC train/eval, harvest. GPU steps will fail without CUDA but the script continues and the harvester handles partial results.

### Models

- **Known Operator (KO):** `--model known_operator` — fan-beam FBP architecture with trainable diagonal weights (Parker x cosine initialization), ramp filter, and distance-weighted backprojection. Few parameters. Single-GPU training via `ct_train.py`.
- **Fully Connected (FC):** `--model fully_connected` — dense learned sinogram-to-image map. At full resolution (512x512, 180 views) the weight matrix is ~90 GB FP32 (~360 GB Adam state). Requires multi-GPU FSDP training via `ct_train_distributed.py` with `torchrun --nproc_per_node=4`.

### Slurm cluster (`gpu_experiments/`)

Two layers of Slurm scripts live under `gpu_experiments/`:

- `gpu_experiments/cluster/slurm/` — portable, generically-named submission scripts for any Slurm cluster. The recommended entry point for the unified KO+FC sample-efficiency sweeps:

  ```bash
  sbatch gpu_experiments/cluster/slurm/fc_ko_sweep_128.sbatch
  sbatch gpu_experiments/cluster/slurm/fc_ko_sweep_256.sbatch
  ```

  Other scripts in this directory cover the surrogate, KO/FC training, eval, FSDP, harvest, and a smoke test. See `gpu_experiments/cluster/README.md` for the full submission protocol, dependency-chained jobs, and resume behaviour.

- `gpu_experiments/slurm/` — older H100-specific batch scripts retained for reference:
  - `sbatch gpu_experiments/slurm/h100_run_ko.sh` — KO train+eval on 1x H100 (~2h)
  - `sbatch gpu_experiments/slurm/h100_run_fc.sh` — FC train+eval on 4x H100 via FSDP (~24h)
  - `sbatch gpu_experiments/slurm/h100_run_all.sh` — Full pipeline: surrogate + KO + FC + harvest (~24h)

The FC FSDP job requests `--mem=700G` for CPU-offloaded optimizer state. The cluster partition must be specified explicitly (otherwise jobs may land in a preempt queue with a short guarantee). The `#SBATCH --exclude=...` line is commented out by default and should be filled in with the local cluster's broken or undersized GPU nodes if needed.

### Linting / Testing

- No test suite or linter configuration is included in the repository.
- Cheapest end-to-end validation: run `cpu_experiments/cpu_surrogate_sweep.py`. It reports per-H calibration ratios and finishes in seconds.
- For the GPU pipeline, use `configs/ct_test_small.yaml` and run both models through train + eval. Check that the KO model's rRMSE improves over the analytic baseline.

### Caveats

- The `harvest_results.py` script emits a `DeprecationWarning` about `datetime.utcnow()`. This is harmless and does not affect output.
- Fan-beam geometry uses `source_to_iso_mm`, `source_to_detector_mm`, `detector_pixel_mm` from the config. The FOV radius is derived as `0.5 * bins * pixel_mm * D_si / D_sd`.
- The `cpu_experiments/` and `gpu_experiments/` paths produce non-overlapping output filenames by design (CPU outputs are all `cpu_*`-prefixed). They can coexist in the same checkout safely.
