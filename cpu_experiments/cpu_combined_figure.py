#!/usr/bin/env python3
"""Reproduce Figure 3 (Sample-efficiency sweeps) from the CPU surrogate runs.

Loads the per-H result JSONs written by ``cpu_surrogate_sweep.py``,
fits the calibrated bound floor + sigma * log N / N per architecture,
and writes a 1x3 figure with N=128 already excluded by upstream.

Output: ``results/cpu_sample_efficiency_combined.png``.

This script is self-contained; filenames are CPU-specific so the file
will not collide with a separately-released GPU bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

DISPLAY = {"known_operator": "Known operator", "fully_connected": "Fully connected"}
COLORS = {"known_operator": "#1b7f5a", "fully_connected": "#b54b32"}


def load(H: int) -> dict:
    return json.loads((RESULTS_DIR / f"cpu_results_H{H}.json").read_text())


def per_seed_array(raw, train_sizes, seeds):
    out = np.zeros((len(seeds), len(train_sizes)), dtype=np.float64)
    for i, seed in enumerate(seeds):
        for j, ts in enumerate(train_sizes):
            row = next(r for r in raw if r["seed"] == seed and r["train_size"] == ts)
            out[i, j] = row["test_mse"]
    return out


def fit_calibration(N_arr, mean_arr):
    floor = float(mean_arr.min())
    excess = np.maximum(mean_arr - floor, 0.0)
    proxy = np.log(N_arr) / N_arr
    if (proxy ** 2).sum() > 0:
        sigma = float((proxy * excess).sum() / (proxy ** 2).sum())
    else:
        sigma = 0.0
    return floor, sigma


def plot_one_H(H: int, ax) -> None:
    d = load(H)
    seeds = d["seeds"]
    train_sizes = d["train_sizes"]
    N_arr = np.array(train_sizes, dtype=np.float64)

    for model in ["known_operator", "fully_connected"]:
        per_seed = per_seed_array(d["raw_results"][model], train_sizes, seeds)
        mean = per_seed.mean(axis=0)
        std = per_seed.std(axis=0, ddof=1)
        floor, sigma = fit_calibration(N_arr, mean)

        ax.errorbar(
            train_sizes, mean, yerr=std,
            marker="o", linewidth=2, capsize=3,
            color=COLORS[model], label=f"{DISPLAY[model]}",
        )
        N_dense = np.geomspace(min(N_arr), max(N_arr), 200)
        bound = floor + sigma * np.log(N_dense) / N_dense
        ax.plot(
            N_dense, bound,
            linestyle="--", linewidth=1.5,
            color=COLORS[model], alpha=0.85,
            label=f"calibrated bound (floor={floor:.2e}, $\\sigma$={sigma:.2e})",
        )

    ax.set_title(f"$H = {H}$, $V = {d['num_angles']}$")
    ax.set_xlabel("Training samples $N$")
    ax.set_ylabel("Test MSE")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def make() -> None:
    Hs = [8, 16, 32]
    fig, axes = plt.subplots(1, len(Hs), figsize=(5.2 * len(Hs), 4.6))
    for k, H in enumerate(Hs):
        plot_one_H(H, axes[k])
    fig.tight_layout()
    out = RESULTS_DIR / "cpu_sample_efficiency_combined.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    make()
