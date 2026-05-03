"""Harvest every artifact produced by run_all.sh into a single Markdown file.

The output of this script is the only file the reviewer reads. It must
therefore be self-contained: hardware metadata, paper-aligned tables, training
metrics, evaluation numbers, runtime information, and any failure traces.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import platform
import shutil
import subprocess
import textwrap
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/RESULTS.md")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--config", default="configs/ct_full_resolution.yaml")
    parser.add_argument("--surrogate-config", default="configs/ct_surrogate.yaml")
    return parser.parse_args()


def safe_load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def load_yaml(path: Path) -> Any:
    if not path.exists():
        return None
    return yaml.safe_load(path.read_text())


def collect_hardware() -> dict[str, str]:
    info: dict[str, str] = {}
    info["timestamp"] = dt.datetime.utcnow().isoformat() + "Z"
    info["python_version"] = platform.python_version()
    info["platform"] = platform.platform()
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_available"] = "True"
            info["cuda_version"] = torch.version.cuda or "unknown"
            info["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_total_memory_gb"] = f"{props.total_memory / (1024 ** 3):.2f}"
            info["gpu_multi_processor_count"] = str(props.multi_processor_count)
        else:
            info["cuda_available"] = "False"
    except Exception as exc:
        info["torch_import_error"] = repr(exc)
    info["cpu_model"] = platform.processor() or "unknown"
    info["cpu_count"] = str(shutil.os.cpu_count() or 0)
    try:
        df = subprocess.check_output(["df", "-h", "."], text=True)
        info["disk_free"] = df.strip().splitlines()[-1]
    except Exception:
        info["disk_free"] = "unavailable"
    return info


def fmt_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    results_dir = Path(args.results_dir)
    cfg = load_yaml(Path(args.config)) or {}
    surrogate_cfg = load_yaml(Path(args.surrogate_config)) or {}

    surrogate_json = safe_load_json(results_dir / "surrogate_results.json")
    train_metrics = safe_load_json(results_dir / "ct_known_operator_metrics.json")
    eval_metrics = safe_load_json(results_dir / "ct_known_operator_eval.json")
    fc_train_metrics = safe_load_json(results_dir / "ct_fully_connected_metrics.json")
    fc_eval_metrics = safe_load_json(results_dir / "ct_fully_connected_eval.json")
    steps_csv = results_dir / "run_all_steps.csv"

    sections: list[str] = []
    sections.append("# Known Operator CT Reproduction Results\n")
    sections.append("This file is produced automatically by `harvest_results.py`. It contains every number the reviewer needs.\n")

    # Hardware
    hw = collect_hardware()
    sections.append("## Hardware")
    sections.append(fmt_table(["Field", "Value"], [[k, v] for k, v in hw.items()]))

    # Surrogate
    sections.append("\n## CPU surrogate (paper Figure 3, Table 2)")
    if surrogate_json is None:
        sections.append("_No surrogate results found. The surrogate stage may have failed; see `run_all.log`._")
    else:
        agg = sorted(surrogate_json["aggregate"], key=lambda r: (r["model"], r["train_size"]))
        rows = []
        for r in agg:
            rows.append([
                r["model"],
                str(r["train_size"]),
                f"{r['mse_mean']:.3e} ± {r['mse_std']:.3e}",
                f"{r['train_time_mean_s']:.4f}",
                f"{r['infer_time_mean_ms']:.4f}",
                f"{r['bound_proxy']:.3e}",
            ])
        sections.append(fmt_table(
            ["Model", "N", "Test MSE (mean ± std)", "Train time s", "Infer time ms", "p log N / N"],
            rows,
        ))
        sections.append(f"\nFigure: `{(results_dir / 'sample_efficiency.png').as_posix()}`")
        p = surrogate_json["parameter_counts"]
        ratio = p["fully_connected"] / max(1, p["known_operator"])
        sections.append(f"\nParameter ratio (fully connected / known operator) at surrogate scale: **{ratio:,.0f}**")

    # Full-resolution training
    sections.append("\n## Full-resolution CT training (operator-aware)")
    if train_metrics is None:
        sections.append("_No full-resolution training metrics found. The training stage may have failed; see `run_all.log`._")
    else:
        g = train_metrics["geometry"]
        rows = [
            ["image size", f"{g['image_size']} × {g['image_size']}"],
            ["views", str(g["num_views"])],
            ["detector bins", str(g["detector_bins"])],
            ["training slices", str(train_metrics.get("num_training_slices", "?"))],
            ["batch size", str(train_metrics["training"]["batch_size"])],
            ["iterations", str(train_metrics["training"]["num_iterations"])],
            ["wall time s", f"{train_metrics.get('wall_time_seconds', float('nan')):.2f}"],
            ["peak GPU memory MB", f"{train_metrics.get('peak_gpu_memory_bytes', 0) / (1024 ** 2):.1f}"],
            ["device", train_metrics.get("device", "unknown")],
        ]
        sections.append(fmt_table(["Field", "Value"], rows))
        if train_metrics["iterations"]:
            last = train_metrics["iterations"][-1]
            sections.append(f"\nLast logged training loss at iter {last['iter']}: **{last['loss']:.6f}**")

    # Full-resolution evaluation
    sections.append("\n## Full-resolution CT evaluation (operator-aware)")
    if eval_metrics is None:
        sections.append("_No evaluation metrics found. The evaluation stage may have failed; see `run_all.log`._")
    else:
        rows = [
            ["test slices", str(eval_metrics["num_test_slices"])],
            ["wall time s", f"{eval_metrics['wall_time_seconds']:.2f}"],
            ["mean inference time ms", f"{eval_metrics['inference_time_ms']['mean']:.2f}"],
            ["rRMSE trained (mean)", f"{eval_metrics['rrmse_trained']['mean']:.4e}"],
            ["rRMSE analytic baseline (mean)", f"{eval_metrics['rrmse_analytic_baseline']['mean']:.4e}"],
        ]
        sections.append(fmt_table(["Metric", "Value"], rows))

    # Full-resolution FC training
    sections.append("\n## Full-resolution CT training (fully connected)")
    if fc_train_metrics is None:
        sections.append("_No FC training metrics found. The FC training stage may have been skipped or failed; see `run_all.log`._")
    else:
        g = fc_train_metrics["geometry"]
        rows = [
            ["image size", f"{g['image_size']} × {g['image_size']}"],
            ["views", str(g["num_views"])],
            ["detector bins", str(g["detector_bins"])],
            ["training slices", str(fc_train_metrics.get("num_training_slices", "?"))],
            ["batch size", str(fc_train_metrics["training"]["batch_size"])],
            ["iterations", str(fc_train_metrics["training"]["num_iterations"])],
            ["wall time s", f"{fc_train_metrics.get('wall_time_seconds', float('nan')):.2f}"],
            ["peak GPU memory MB", f"{fc_train_metrics.get('peak_gpu_memory_bytes', 0) / (1024 ** 2):.1f}"],
            ["device", fc_train_metrics.get("device", "unknown")],
        ]
        sections.append(fmt_table(["Field", "Value"], rows))
        if fc_train_metrics["iterations"]:
            last = fc_train_metrics["iterations"][-1]
            sections.append(f"\nLast logged training loss at iter {last['iter']}: **{last['loss']:.6f}**")

    # Full-resolution FC evaluation
    sections.append("\n## Full-resolution CT evaluation (fully connected)")
    if fc_eval_metrics is None:
        sections.append("_No FC evaluation metrics found. The FC evaluation stage may have been skipped or failed; see `run_all.log`._")
    else:
        rows = [
            ["test slices", str(fc_eval_metrics["num_test_slices"])],
            ["wall time s", f"{fc_eval_metrics['wall_time_seconds']:.2f}"],
            ["mean inference time ms", f"{fc_eval_metrics['inference_time_ms']['mean']:.2f}"],
            ["rRMSE trained (mean)", f"{fc_eval_metrics['rrmse_trained']['mean']:.4e}"],
            ["rRMSE analytic baseline (mean)", f"{fc_eval_metrics['rrmse_analytic_baseline']['mean']:.4e}"],
        ]
        sections.append(fmt_table(["Metric", "Value"], rows))

    # Bound-inspired estimate
    sections.append("\n## Bound-inspired full-scale estimate (paper Table 1)")
    geom = (cfg or {}).get("geometry", {})
    image_size = geom.get("image_size", 512)
    num_views = geom.get("num_views", 180)
    detector_bins = geom.get("detector_bins", 512)
    p_ko = num_views * detector_bins
    p_fc = (image_size ** 2) * p_ko
    weight_bytes_fc = p_fc * 4.0
    adam_bytes_fc = p_fc * 16.0
    bandwidth = 1.0e12
    fwd = weight_bytes_fc / bandwidth
    step = adam_bytes_fc / bandwidth
    rows = [
        ["Trainable parameters (KO)", f"{p_ko:,}"],
        ["Trainable parameters (FC)", f"{p_fc:,}"],
        ["Parameter ratio FC / KO", f"{p_fc / max(1, p_ko):,.0f}"],
        ["FC FP32 weight memory", f"{weight_bytes_fc / (1024 ** 3):.2f} GB"],
        ["FC Adam state memory", f"{adam_bytes_fc / (1024 ** 3):.2f} GB"],
        ["FC forward time at 1 TB/s", f"{fwd:.3e} s"],
        ["FC train step at 1 TB/s", f"{step:.3e} s"],
        ["FC 10k-step lower-bound runtime", f"{10000.0 * step / 60.0:.2f} min"],
    ]
    sections.append(fmt_table(["Quantity", "Value"], rows))

    n_train = (cfg or {}).get("dataset", {}).get("num_train_slices", 2140)
    estimation_proxy = math.log(n_train) / n_train
    n_match_lo, n_match_hi = 10.0, 1e12
    for _ in range(200):
        mid = 0.5 * (n_match_lo + n_match_hi)
        rhs = p_fc * math.log(mid) / mid
        target = p_ko * estimation_proxy
        if rhs > target:
            n_match_lo = mid
        else:
            n_match_hi = mid
    sections.append(
        f"\nUnder a matched estimation budget at $N = {n_train}$, the dense substitute would require approximately"
        f" **{n_match_hi:,.0f} training slices** to match the operator-aware proxy."
    )

    # Run timings
    sections.append("\n## Run timings")
    if steps_csv.exists():
        rows = []
        for line in steps_csv.read_text().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) == 3:
                rows.append(parts)
        if rows:
            sections.append(fmt_table(["Step", "Status", "Elapsed s"], rows))
    else:
        sections.append("_No per-step timings recorded._")

    # Failure trace
    log_path = results_dir / "run_all.log"
    if log_path.exists() and any(line.startswith("fail:") for line in log_path.read_text().splitlines()):
        sections.append("\n## Environment failures")
        sections.append("```")
        sections.append("\n".join(line for line in log_path.read_text().splitlines() if line.startswith("fail:")))
        sections.append("```")

    out_path.write_text("\n\n".join(sections) + "\n")


if __name__ == "__main__":
    main()
