# AGENTS.md

## Cursor Cloud specific instructions

This is a single-service Python scientific computing project (no web servers, databases, or external services). See `README.md` and `docs/AGENT_INSTRUCTIONS.md` for the full protocol.

### Environment

- Python venv at `.venv/` — activate with `source .venv/bin/activate`.
- Dependencies are in `requirements.txt` (pip).
- No GPU is available in the Cloud Agent VM. The CPU surrogate experiment runs fine without one; the full-resolution CT training/eval can run on CPU with a small config (see `configs/ct_test_small.yaml`).

### Running the project

- **Smoke test (CPU surrogate):** `python src/run_surrogate.py --config configs/ct_surrogate.yaml` — takes ~1-2 minutes, produces `results/surrogate_results.json`, `results/surrogate_ablation.csv`, `results/sample_efficiency.png`.
- **Small-scale GPU code validation on CPU:** Use `configs/ct_test_small.yaml` (32x32, 20 views) with `python src/ct_train.py --config configs/ct_test_small.yaml --model known_operator` and `--model fully_connected`. Takes seconds on CPU.
- **Harvest results:** `python src/harvest_results.py --output results/RESULTS.md` — aggregates all artifacts into a single Markdown report. Supports both KO and FC model results.
- **Full pipeline:** `bash run_all.sh` — runs surrogate, KO train/eval, FC train/eval, harvest. GPU steps will fail without CUDA but the script continues and the harvester handles partial results.

### Models

- **Known Operator (KO):** `--model known_operator` — fan-beam FBP architecture with trainable diagonal weights (Parker x cosine initialization), ramp filter, and distance-weighted backprojection. Few parameters. Single-GPU training via `ct_train.py`.
- **Fully Connected (FC):** `--model fully_connected` — dense learned sinogram-to-image map. At full resolution (512x512, 180 views) the weight matrix is ~90 GB FP32 (~360 GB Adam state). Requires multi-GPU FSDP training via `ct_train_distributed.py` with `torchrun --nproc_per_node=4`.

### H100 Slurm cluster (4x H100 94 GB per node, 768 GB RAM)

SLURM job scripts are in `slurm/`:
- `sbatch slurm/h100_run_ko.sh` — KO train+eval on 1x H100 (~2h)
- `sbatch slurm/h100_run_fc.sh` — FC train+eval on 4x H100 via FSDP (~24h)
- `sbatch slurm/h100_run_all.sh` — Full pipeline: surrogate + KO + FC + harvest (~24h)

The FC FSDP job requests `--mem=700G` for CPU-offloaded optimizer state. The `h100` partition must be specified explicitly (otherwise jobs may land in a preempt queue with a 2h guarantee).

### Linting / Testing

- No test suite or linter configuration is included in the repository.
- To verify correctness, use `configs/ct_test_small.yaml` and run both models through train + eval. Check that the KO model's rRMSE improves over the analytic baseline.

### Caveats

- The `harvest_results.py` script emits a `DeprecationWarning` about `datetime.utcnow()`. This is harmless and does not affect output.
- Fan-beam geometry uses `source_to_iso_mm`, `source_to_detector_mm`, `detector_pixel_mm` from the config. The FOV radius is derived as `0.5 * bins * pixel_mm * D_si / D_sd`.
