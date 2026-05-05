"""Aggregate the ridge_fc.py JSON logs from each scale into one figure.

Reads:
  results/sample_efficiency_64/ridge_fc_log.json
  results/sample_efficiency_128/ridge_fc_log.json
  results/sample_efficiency_256/ridge_fc_log.json

Writes:
  results/ridge_fc_sample_efficiency.png  — Test rRMSE vs. N for FC ridge
                                            at each scale, with KO baselines
                                            as horizontal lines.
  results/ridge_fc_sample_efficiency.csv  — flat table of the same numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SCALES = [
    ("64x64",   "results/sample_efficiency_64/ridge_fc_log.json",   "#1b7f5a"),
    ("128x128", "results/sample_efficiency_128/ridge_fc_log.json",  "#b54b32"),
    ("256x256", "results/sample_efficiency_256/ridge_fc_log.json",  "#3b5b9a"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/ridge_fc_sample_efficiency.png")
    p.add_argument("--csv", default="results/ridge_fc_sample_efficiency.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    csv_path = Path(args.csv)

    # Collect data
    series = []
    for label, path, color in SCALES:
        p = Path(path)
        if not p.exists():
            print(f"[plot] missing {path} — skipping {label}")
            continue
        d = json.loads(p.read_text())
        # New schema (post-seed-averaging) uses fc_per_n with rrmse_mean_across_seeds.
        # Old schema used fc_rrmse_per_n with mean/std over test slices.
        if "fc_per_n" in d:
            ns = sorted(int(n) for n in d["fc_per_n"].keys())
            means = [d["fc_per_n"][str(n)]["rrmse_mean_across_seeds"] for n in ns]
            stds = [d["fc_per_n"][str(n)]["rrmse_std_across_seeds"] for n in ns]
            lam_desc = f"λ ∈ {d.get('lambdas', '?')}, seeds {d.get('seeds', '?')}"
        elif "fc_rrmse_per_n" in d:
            ns = sorted(int(n) for n in d["fc_rrmse_per_n"].keys())
            means = [d["fc_rrmse_per_n"][str(n)]["mean"] for n in ns]
            stds = [d["fc_rrmse_per_n"][str(n)]["std"] for n in ns]
            lam_desc = f"λ={d.get('lambda', '?')} (single seed)"
        else:
            print(f"[plot] {path} has no per-N stats — skipping {label}")
            continue
        ko_mean = d["ko_rrmse"]["mean"]
        ko_std = d["ko_rrmse"]["std"]
        p_fc = d["geometry"]["image_size"] ** 2 * d["geometry"]["num_views"] * d["geometry"]["detector_bins"]
        p_ko = d["geometry"]["num_views"] * d["geometry"]["detector_bins"]
        series.append({
            "label": label, "color": color, "path": path,
            "ns": ns, "means": means, "stds": stds,
            "ko_mean": ko_mean, "ko_std": ko_std,
            "p_fc": p_fc, "p_ko": p_ko,
            "lam": lam_desc,
            "num_test": d.get("num_test_stats", "?"),
        })
    if not series:
        print("[plot] no data found — exiting")
        return

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for s in series:
        # Left: rRMSE vs N
        axes[0].errorbar(
            s["ns"], s["means"], yerr=s["stds"],
            marker="o", linewidth=2, capsize=3, color=s["color"],
            label=f"FC ridge {s['label']} (p={s['p_fc']:,})",
        )
        axes[0].axhline(
            s["ko_mean"], color=s["color"], linestyle="--", linewidth=1.2,
            alpha=0.65,
            label=f"KO {s['label']} = {s['ko_mean']:.3f}",
        )

        # Right: bound proxy p log N / N
        ns_arr = np.array(s["ns"], dtype=float)
        proxy = s["p_fc"] * np.log(np.clip(ns_arr, 2, None)) / ns_arr
        axes[1].plot(ns_arr, proxy, marker="o", linewidth=2, color=s["color"],
                     label=f"FC {s['label']} ($p_{{FC}}={s['p_fc']:,}$)")

    axes[0].set_xlabel("Training samples $N$")
    axes[0].set_ylabel("Test rRMSE (mean ± std)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_title("FC ridge regression sample efficiency")
    axes[0].grid(True, alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    axes[1].set_xlabel("Training samples $N$")
    axes[1].set_ylabel(r"$p_{FC}\,\log N \,/\, N$")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_title("Estimation proxy")
    axes[1].grid(True, alpha=0.25, which="both")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")

    fig.suptitle(
        "Ridge regression FC at increasing scales — closed-form, "
        "λ chosen from a grid per (N, seed); error bars across seeds",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # CSV dump
    with csv_path.open("w", newline="") as h:
        w = csv.writer(h)
        w.writerow(["scale", "p_FC", "p_KO", "lambda", "num_test_slices",
                    "model", "N",
                    "rrmse_mean", "rrmse_std"])
        for s in series:
            w.writerow([s["label"], s["p_fc"], s["p_ko"], s["lam"],
                        s["num_test"], "ko", "—",
                        f"{s['ko_mean']:.6f}", f"{s['ko_std']:.6f}"])
            for n, m, st in zip(s["ns"], s["means"], s["stds"]):
                w.writerow([s["label"], s["p_fc"], s["p_ko"], s["lam"],
                            s["num_test"], "fc_ridge", n,
                            f"{m:.6f}", f"{st:.6f}"])
    print(f"[plot] wrote {out_path} and {csv_path}")


if __name__ == "__main__":
    main()
