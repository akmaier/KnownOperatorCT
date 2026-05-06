"""Per-scale FC/KO sample-efficiency plots, one figure per resolution.

For each scale (64, 128, 256) we produce a 2-panel figure mirroring the
surrogate's results/sample_efficiency.png:

  Left  — Test rRMSE vs N for KO (operator-aware) and FC ridge curves.
  Right — Estimation proxy p log N / N for both models.

Inputs (read-only, no retraining):
  results/sample_efficiency_<S>/sample_efficiency_gpu_results.json
      → KO rRMSE per N (varying), FC SGD per N (used for parameter counts).
  results/sample_efficiency_<S>/ridge_fc_log.json
      → FC ridge rRMSE per N (best-lambda across seeds).

Outputs:
  results/sample_efficiency_<S>/fc_ko_sample_efficiency.png   # the figure
  results/sample_efficiency_<S>/fc_ko_sample_efficiency.csv   # numbers
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scales", default="64,128,256",
                   help="Comma-separated list of scales to plot.")
    p.add_argument("--results-root", default="results")
    return p.parse_args()


def mse_to_rrmse(mse: float, phantom_max: float = 1.5) -> float:
    """Approximate rRMSE = sqrt(MSE) / phantom.max() — convention used in
    the rest of this codebase. Phantom values lie in [0, 1.5] by design."""
    return float(math.sqrt(max(mse, 0.0)) / phantom_max)


def main() -> None:
    args = parse_args()
    scales = [int(s) for s in args.scales.split(",")]

    for scale in scales:
        ko_color = "#1b7f5a"
        fc_color = "#b54b32"
        scale_dir = Path(args.results_root) / f"sample_efficiency_{scale}"
        sgd_path = scale_dir / "sample_efficiency_gpu_results.json"
        ridge_path = scale_dir / "ridge_fc_log.json"

        if not sgd_path.exists():
            print(f"[plot] missing {sgd_path} — skipping scale {scale}")
            continue
        if not ridge_path.exists():
            print(f"[plot] missing {ridge_path} — skipping scale {scale}")
            continue

        sgd = json.loads(sgd_path.read_text())
        ridge = json.loads(ridge_path.read_text())

        # KO curve: from sample_efficiency_gpu_results.json
        ko_rows = [r for r in sgd["aggregate"] if r["model"] == "known_operator"]
        ko_rows.sort(key=lambda r: r["train_size"])
        ko_ns = [r["train_size"] for r in ko_rows]
        ko_mse = [r["mse_mean"] for r in ko_rows]
        ko_mse_std = [r["mse_std"] for r in ko_rows]
        ko_rrmse = [mse_to_rrmse(m) for m in ko_mse]
        # First-order error propagation: rRMSE = sqrt(MSE)/c, σ_rRMSE ≈ σ_MSE / (2 c sqrt(MSE)).
        ko_rrmse_std = [
            (s / (2 * 1.5 * math.sqrt(max(m, 1e-12)))) if m > 0 else 0.0
            for m, s in zip(ko_mse, ko_mse_std)
        ]

        # FC ridge curve: from ridge_fc_log.json (already in rRMSE)
        fc_per_n = ridge["fc_per_n"]
        fc_ns = sorted(int(n) for n in fc_per_n.keys())
        fc_rrmse = [fc_per_n[str(n)]["rrmse_mean_across_seeds"] for n in fc_ns]
        fc_rrmse_std = [fc_per_n[str(n)]["rrmse_std_across_seeds"] for n in fc_ns]

        p_ko = sgd["parameter_counts"]["known_operator"]
        p_fc = sgd["parameter_counts"]["fully_connected"]
        n_views = sgd["geometry"]["num_views"]
        det_bins = sgd["geometry"]["detector_bins"]

        # ---------- figure ----------
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

        axes[0].errorbar(
            ko_ns, ko_rrmse, yerr=ko_rrmse_std,
            marker="o", linewidth=2, capsize=3, color=ko_color,
            label=f"Known operator ($p={p_ko:,}$)",
        )
        axes[0].errorbar(
            fc_ns, fc_rrmse, yerr=fc_rrmse_std,
            marker="o", linewidth=2, capsize=3, color=fc_color,
            label=f"Fully connected (ridge, $p={p_fc:,}$)",
        )
        axes[0].set_xlabel("Training samples $N$")
        axes[0].set_ylabel("Test rRMSE (mean ± std)")
        axes[0].set_xscale("log", base=2)
        axes[0].set_yscale("log")
        axes[0].grid(True, alpha=0.25, which="both")
        axes[0].legend(frameon=False, loc="upper right")
        axes[0].set_title(f"Test rRMSE vs. training size @ {scale}×{scale}")

        # Right panel: bound proxy p log N / N
        for ns, p, color, name in [
            (ko_ns, p_ko, ko_color, "Known operator"),
            (fc_ns, p_fc, fc_color, "Fully connected"),
        ]:
            ns_arr = np.array(ns, dtype=float)
            proxy = p * np.log(np.clip(ns_arr, 2, None)) / ns_arr
            axes[1].plot(
                ns_arr, proxy, marker="o", linewidth=2, color=color,
                label=f"{name} ($p={p:,}$)",
            )
        axes[1].set_xlabel("Training samples $N$")
        axes[1].set_ylabel(r"$p\,\log N \,/\, N$")
        axes[1].set_xscale("log", base=2)
        axes[1].set_yscale("log")
        axes[1].grid(True, alpha=0.25, which="both")
        axes[1].legend(frameon=False, loc="upper right")
        axes[1].set_title(r"Estimation proxy $p\,\log N \,/\, N$")

        fig.suptitle(
            f"Sample efficiency @ {scale}×{scale} fan-beam, "
            f"{n_views} views, {det_bins} detector bins",
            fontsize=12,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))

        out_png = scale_dir / "fc_ko_sample_efficiency.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

        # ---------- CSV ----------
        out_csv = scale_dir / "fc_ko_sample_efficiency.csv"
        with out_csv.open("w", newline="") as h:
            w = csv.writer(h)
            w.writerow(["scale", "model", "N", "rrmse_mean", "rrmse_std",
                        "p_param"])
            for n, m, s in zip(ko_ns, ko_rrmse, ko_rrmse_std):
                w.writerow([scale, "known_operator", n,
                            f"{m:.6f}", f"{s:.6f}", p_ko])
            for n, m, s in zip(fc_ns, fc_rrmse, fc_rrmse_std):
                w.writerow([scale, "fully_connected_ridge", n,
                            f"{m:.6f}", f"{s:.6f}", p_fc])

        print(f"[plot] wrote {out_png} and {out_csv}")


if __name__ == "__main__":
    main()
