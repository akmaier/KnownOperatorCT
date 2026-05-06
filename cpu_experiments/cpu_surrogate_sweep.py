#!/usr/bin/env python3
"""CPU-scale surrogate sweep at H in {8, 16, 32}, N in {4, 8, 16, 32, 64}.

Reproduces the CPU-side experiments of *A Deep Risk Estimator for Known
Operator Learning*. For each image scale H the script runs five random
seeds for each training-set size N and writes a per-H JSON result file
into ``results/`` next to this script.

Outputs (under ``results/``):

  cpu_results_H{H}.json      raw per-seed test MSE plus aggregate stats
  cpu_ablation_H{H}.csv      one row per (model, train_size) aggregate
  cpu_multiH_summary.json    cross-H calibration summary

The script is self-contained: it does not import from any other module
in the bundle. Filenames are CPU-specific so this folder can be merged
with a separately-released GPU bundle without collisions.

Use ``cpu_combined_figure.py`` to produce the published Figure 3.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.linalg import solve


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ----- Geometry / sweep configuration --------------------------------------

# The three CPU operating points used in the paper.  V is set to 1.25 H so
# the parameter ratio p_FC / p_KO = H^2 holds exactly at every sweep point.
SWEEPS = [
    {"H": 8,  "V": 10},
    {"H": 16, "V": 20},
    {"H": 32, "V": 40},
]

# Training-set sizes.  The N = 128 column is intentionally excluded from
# the public sweep because the closed-form ridge solution overfits the
# fully connected baseline at that point, producing a bimodal distribution
# that pollutes the bound calibration.  See Section 6 of the paper.
TRAIN_SIZES = [4, 8, 16, 32, 64]

# Five seeds per sweep, ridge regularization grid taken from the paper.
SEEDS = [1, 2, 3, 4, 5]
LAMBDA_GRID = [1e-6, 1e-4, 1e-2, 1e0, 1e2]

NUM_TRAIN_POOL = 128
NUM_VAL = 32
NUM_TEST = 128


# ----- Data structures ------------------------------------------------------

@dataclass
class AggregateRow:
    image_side: int
    num_angles: int
    train_size: int
    model: str
    mse_mean: float
    mse_std: float
    train_time_mean_s: float
    bound_proxy: float


# ----- Surrogate definition -------------------------------------------------

class CTSurrogate:
    """Linear inverse-problem surrogate at image scale H, view count V."""

    def __init__(self, image_side: int, num_angles: int) -> None:
        self.image_side = image_side
        self.num_angles = num_angles
        self.angles = np.linspace(0.0, 180.0, num_angles, endpoint=False)
        self.num_pixels = image_side * image_side
        self.num_measurements = image_side * num_angles
        self.forward_matrix = self._build_forward_matrix()
        self.backprojection_matrix = np.linalg.inv(
            self.forward_matrix.T @ self.forward_matrix
            + 1e-1 * np.eye(self.num_pixels)
        ) @ self.forward_matrix.T
        # B^T B is reused across seeds in the KO Hadamard-trick fit.
        self.bp_gram = self.backprojection_matrix.T @ self.backprojection_matrix

    def _random_phantom(
        self, rng: np.random.Generator, n_ellipses=(1, 4)
    ) -> np.ndarray:
        yy, xx = np.mgrid[
            -1:1 : complex(0, self.image_side),
            -1:1 : complex(0, self.image_side),
        ]
        image = np.zeros((self.image_side, self.image_side), dtype=np.float64)
        for _ in range(rng.integers(n_ellipses[0], n_ellipses[1] + 1)):
            amplitude = rng.uniform(0.2, 1.0)
            x0, y0 = rng.uniform(-0.5, 0.5, size=2)
            axis_a, axis_b = rng.uniform(0.08, 0.4, size=2)
            theta = rng.uniform(0.0, np.pi)
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            xr = cos_theta * (xx - x0) + sin_theta * (yy - y0)
            yr = -sin_theta * (xx - x0) + cos_theta * (yy - y0)
            mask = (xr / axis_a) ** 2 + (yr / axis_b) ** 2 <= 1.0
            image[mask] += amplitude
        return np.clip(image, 0.0, 1.5)

    def _forward_project(self, image: np.ndarray) -> np.ndarray:
        sinogram = []
        for angle in self.angles:
            rotated = ndimage.rotate(
                image, angle, reshape=False, order=1, mode="constant",
                cval=0.0, prefilter=False,
            )
            sinogram.append(rotated.sum(axis=0))
        return np.stack(sinogram, axis=0)

    def _build_forward_matrix(self) -> np.ndarray:
        matrix = np.zeros(
            (self.num_measurements, self.num_pixels), dtype=np.float64
        )
        for pixel_index in range(self.num_pixels):
            basis = np.zeros(
                (self.image_side, self.image_side), dtype=np.float64
            )
            basis.flat[pixel_index] = 1.0
            matrix[:, pixel_index] = self._forward_project(basis).ravel()
        return matrix

    def build_dataset(self, seed: int):
        rng = np.random.default_rng(seed)
        images, sinograms = [], []
        total = NUM_TRAIN_POOL + NUM_VAL + NUM_TEST
        for _ in range(total):
            image = self._random_phantom(rng)
            sinogram = self._forward_project(image)
            images.append(image.ravel())
            sinograms.append(sinogram.ravel())
        x = np.stack(images, axis=1)
        y = np.stack(sinograms, axis=1)
        x_pool = x[:, :NUM_TRAIN_POOL]
        y_pool = y[:, :NUM_TRAIN_POOL]
        x_val = x[:, NUM_TRAIN_POOL : NUM_TRAIN_POOL + NUM_VAL]
        y_val = y[:, NUM_TRAIN_POOL : NUM_TRAIN_POOL + NUM_VAL]
        x_test = x[:, NUM_TRAIN_POOL + NUM_VAL :]
        y_test = y[:, NUM_TRAIN_POOL + NUM_VAL :]
        return x_pool, y_pool, x_val, y_val, x_test, y_test


# ----- Closed-form ridge fits ----------------------------------------------

def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((prediction - target) ** 2))


def fit_fc(y_train, x_train, lam, num_measurements):
    return x_train @ y_train.T @ np.linalg.inv(
        y_train @ y_train.T + lam * np.eye(num_measurements)
    )


def fit_ko(surrogate: CTSurrogate, y_train, x_train, lam):
    """KO weights via the Hadamard identity for the design Gram matrix.

    For learned weights w, the KO predictor is hat x = B diag(w) y.
    Stacking over training samples, the design matrix has rows
    A_s = B diag(y_s).  The Gram matrix is
    (A_s^T A_s)_{jk} = (B^T B)_{jk} * y_{s,j} * y_{s,k}.
    Summing over s gives (B^T B) (Hadamard) (Y Y^T), which is much faster
    than building a 3D tensor.
    """
    B_gram = surrogate.bp_gram
    YYt = y_train @ y_train.T
    M = B_gram * YYt
    Bt_x = surrogate.backprojection_matrix.T @ x_train
    rhs = (Bt_x * y_train).sum(axis=1)
    w = solve(
        M + lam * np.eye(surrogate.num_measurements),
        rhs, assume_a="pos",
    )
    return w


def predict_ko(surrogate: CTSurrogate, weights, y):
    return surrogate.backprojection_matrix @ (weights[:, None] * y)


# ----- Per-H sweep ----------------------------------------------------------

def run_one_H(image_side: int, num_angles: int) -> dict:
    surrogate = CTSurrogate(image_side=image_side, num_angles=num_angles)
    parameter_counts = {
        "known_operator": surrogate.num_measurements,
        "fully_connected": surrogate.num_pixels * surrogate.num_measurements,
    }
    raw: dict[str, list[dict]] = {"known_operator": [], "fully_connected": []}

    for seed in SEEDS:
        x_pool, y_pool, x_val, y_val, x_test, y_test = surrogate.build_dataset(seed)
        for ts in TRAIN_SIZES:
            x_train = x_pool[:, :ts]
            y_train = y_pool[:, :ts]

            best_fc, best_ko = None, None
            for lam in LAMBDA_GRID:
                t0 = time.perf_counter()
                fc_M = fit_fc(y_train, x_train, lam, surrogate.num_measurements)
                fc_train_t = time.perf_counter() - t0
                fc_val = mse(fc_M @ y_val, x_val)
                if best_fc is None or fc_val < best_fc["val_mse"]:
                    best_fc = {"lam": lam, "M": fc_M, "val_mse": fc_val,
                               "train_t": fc_train_t}

                t0 = time.perf_counter()
                w = fit_ko(surrogate, y_train, x_train, lam)
                ko_train_t = time.perf_counter() - t0
                ko_val = mse(predict_ko(surrogate, w, y_val), x_val)
                if best_ko is None or ko_val < best_ko["val_mse"]:
                    best_ko = {"lam": lam, "w": w, "val_mse": ko_val,
                               "train_t": ko_train_t}

            raw["fully_connected"].append({
                "seed": seed, "train_size": ts,
                "test_mse": mse(best_fc["M"] @ y_test, x_test),
                "train_time_s": float(best_fc["train_t"]),
                "selected_lambda": float(best_fc["lam"]),
            })
            raw["known_operator"].append({
                "seed": seed, "train_size": ts,
                "test_mse": mse(predict_ko(surrogate, best_ko["w"], y_test), x_test),
                "train_time_s": float(best_ko["train_t"]),
                "selected_lambda": float(best_ko["lam"]),
            })

    aggregate = []
    for model_name, runs in raw.items():
        p = parameter_counts[model_name]
        for ts in TRAIN_SIZES:
            subset = [r for r in runs if r["train_size"] == ts]
            aggregate.append(AggregateRow(
                image_side=image_side, num_angles=num_angles,
                train_size=ts, model=model_name,
                mse_mean=float(np.mean([r["test_mse"] for r in subset])),
                mse_std=float(np.std([r["test_mse"] for r in subset], ddof=1)),
                train_time_mean_s=float(np.mean([r["train_time_s"] for r in subset])),
                bound_proxy=p * math.log(ts) / ts,
            ))

    return {
        "image_side": image_side,
        "num_angles": num_angles,
        "num_pixels": surrogate.num_pixels,
        "num_measurements": surrogate.num_measurements,
        "parameter_counts": parameter_counts,
        "lambda_grid": LAMBDA_GRID,
        "train_sizes": TRAIN_SIZES,
        "seeds": SEEDS,
        "aggregate": [asdict(row) for row in aggregate],
        "raw_results": raw,
    }


# ----- Calibration ----------------------------------------------------------

def fit_calibration(aggregate, parameter_counts):
    """floor = smallest mean MSE in the sweep;
    sigma = no-intercept LS slope of (MSE - floor) against (log N) / N.
    """
    out = {}
    for model in ["known_operator", "fully_connected"]:
        rows = [r for r in aggregate if r["model"] == model]
        rows.sort(key=lambda r: r["train_size"])
        mse_arr = np.array([r["mse_mean"] for r in rows])
        N_arr = np.array([r["train_size"] for r in rows], dtype=np.float64)
        floor = float(mse_arr.min())
        excess = np.maximum(mse_arr - floor, 0.0)
        proxy = np.log(N_arr) / N_arr
        if (proxy ** 2).sum() > 0:
            sigma = float((proxy * excess).sum() / (proxy ** 2).sum())
        else:
            sigma = 0.0
        out[model] = {"floor": floor, "sigma": sigma,
                      "p": parameter_counts[model]}
    return out


def main() -> None:
    summary = {"sweeps": []}
    for cfg in SWEEPS:
        H, V = cfg["H"], cfg["V"]
        print(f"=== Running H = {H}, V = {V} ===", flush=True)
        t0 = time.perf_counter()
        out = run_one_H(image_side=H, num_angles=V)
        elapsed = time.perf_counter() - t0
        print(f"    elapsed: {elapsed:.1f} s", flush=True)

        (RESULTS_DIR / f"cpu_results_H{H}.json").write_text(
            json.dumps(out, indent=2)
        )
        with (RESULTS_DIR / f"cpu_ablation_H{H}.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(out["aggregate"][0].keys()))
            writer.writeheader()
            for row in out["aggregate"]:
                writer.writerow(row)

        cal = fit_calibration(out["aggregate"], out["parameter_counts"])
        ko, fc = cal["known_operator"], cal["fully_connected"]
        ratio = (
            fc["sigma"] * fc["p"] / (ko["sigma"] * ko["p"])
            if ko["sigma"] > 0 else float("nan")
        )
        summary["sweeps"].append({
            "H": H, "V": V, "B": H,
            "p_KO": ko["p"], "p_FC": fc["p"],
            "param_count_ratio_H2": fc["p"] / ko["p"],
            "floor_KO": ko["floor"], "sigma_KO": ko["sigma"],
            "floor_FC": fc["floor"], "sigma_FC": fc["sigma"],
            "empirical_sigma_ratio": ratio,
            "elapsed_s": elapsed,
        })
        print(f"    H={H}:  floor_KO={ko['floor']:.3e}  sigma_KO={ko['sigma']:.3e}")
        print(f"            floor_FC={fc['floor']:.3e}  sigma_FC={fc['sigma']:.3e}")
        print(f"            sigma_FC/sigma_KO = {fc['sigma']/ko['sigma']:.3f}")

    (RESULTS_DIR / "cpu_multiH_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(f"\nWrote {RESULTS_DIR / 'cpu_multiH_summary.json'}")


if __name__ == "__main__":
    main()
