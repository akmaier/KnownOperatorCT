# AGENTS.md

## Cursor Cloud specific instructions

This is a single-service Python scientific computing project (no web servers, databases, or external services). See `README.md` and `docs/AGENT_INSTRUCTIONS.md` for the full protocol.

### Environment

- Python venv at `.venv/` — activate with `source .venv/bin/activate`.
- Dependencies are in `requirements.txt` (pip).
- No GPU is available in the Cloud Agent VM. The CPU surrogate experiment runs fine without one; the full-resolution CT training (`ct_train.py`, `ct_eval.py`) requires an NVIDIA GPU and will not work here.

### Running the project

- **Smoke test (CPU surrogate):** `python src/run_surrogate.py --config configs/ct_surrogate.yaml` — takes ~1-2 minutes, produces `results/surrogate_results.json`, `results/surrogate_ablation.csv`, `results/sample_efficiency.png`.
- **Harvest results:** `python src/harvest_results.py --output results/RESULTS.md` — aggregates all artifacts into a single Markdown report.
- **Full pipeline:** `bash run_all.sh` — runs all 4 steps (surrogate, train, eval, harvest). Steps 2-3 will fail without a GPU but the script continues and the harvester handles partial results.

### Linting / Testing

- No test suite or linter configuration is included in the repository. There are no `pytest`, `flake8`, `ruff`, or `mypy` configs.
- To verify correctness, run the CPU surrogate and check that `results/surrogate_results.json` is valid JSON with expected keys (`geometry`, `parameter_counts`, `aggregate`, `raw_results`).

### Caveats

- `docs/AGENT_INSTRUCTIONS.md` says "do not modify any file under `src/` or `configs/`". Respect this unless the user explicitly overrides it.
- The `harvest_results.py` script emits a `DeprecationWarning` about `datetime.utcnow()`. This is harmless and does not affect output.
