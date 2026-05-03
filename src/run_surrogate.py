"""CPU-scale surrogate for the computed tomography application.

This script reproduces Figure 3 and Table 2 of the paper. It depends only on
NumPy, SciPy, and Matplotlib so it can be used as a smoke test on any modern
CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy import ndimage
from scipy.linalg import solve


@dataclass
class AggregateRow:
    train_size: int
    model: str
    mse_mean: float
    mse_std: float
    train_time_mean_s: float
    train_time_std_s: float
    infer_time_mean_ms: float
    infer_time_std_ms: float
    selected_lambda_mode: float
    bound_proxy: float


class CTSurrogate:
    def __init__(self, image_side: int = 16, num_angles: int = 20) -> None:
        self.image_side = image_side
        self.num_angles = num_angles
        self.angles = np.linspace(0.0, 180.0, num_angles, endpoint=False)
        self.num_pixels = image_side * image_side
        self.num_measurements = image_side * num_angles
        self.forward_matrix = self._build_forward_matrix()
        self.backprojection_matrix = np.linalg.inv(
            self.forward_matrix.T @ self.forward_matrix + 1e-1 * np.eye(self.num_pixels)
        ) @ self.forward_matrix.T

    def _random_phantom(self, rng: np.random.Generator) -> np.ndarray:
        yy, xx = np.mgrid[
            -1:1 : complex(0, self.image_side),
            -1:1 : complex(0, self.image_side),
        ]
        image = np.zeros((self.image_side, self.image_side), dtype=np.float64)
        for _ in range(rng.integers(1, 5)):
            amplitude = rng.uniform(0.2, 1.0)
            x0, y0 = rng.uniform(-0.5, 0.5, size=2)
            a, b = rng.uniform(0.08, 0.4, size=2)
            theta = rng.uniform(0.0, np.pi)
            ct, st = np.cos(theta), np.sin(theta)
            xr = ct * (xx - x0) + st * (yy - y0)
            yr = -st * (xx - x0) + ct * (yy - y0)
            mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
            image[mask] += amplitude
        return np.clip(image, 0.0, 1.5)

    def _forward_project(self, image: np.ndarray) -> np.ndarray:
        out = []
        for angle in self.angles:
            rotated = ndimage.rotate(image, angle, reshape=False, order=1, mode="constant", cval=0.0, prefilter=False)
            out.append(rotated.sum(axis=0))
        return np.stack(out, axis=0)

    def _build_forward_matrix(self) -> np.ndarray:
        m = np.zeros((self.num_measurements, self.num_pixels), dtype=np.float64)
        for i in range(self.num_pixels):
            basis = np.zeros((self.image_side, self.image_side), dtype=np.float64)
            basis.flat[i] = 1.0
            m[:, i] = self._forward_project(basis).ravel()
        return m

    def build_dataset(self, seed: int, n_pool: int, n_val: int, n_test: int):
        rng = np.random.default_rng(seed)
        images, sinograms = [], []
        for _ in range(n_pool + n_val + n_test):
            img = self._random_phantom(rng)
            sino = self._forward_project(img)
            images.append(img.ravel())
            sinograms.append(sino.ravel())
        x = np.stack(images, axis=1)
        y = np.stack(sinograms, axis=1)
        return (
            x[:, :n_pool], y[:, :n_pool],
            x[:, n_pool : n_pool + n_val], y[:, n_pool : n_pool + n_val],
            x[:, n_pool + n_val :], y[:, n_pool + n_val :],
        )


def mse(p: np.ndarray, t: np.ndarray) -> float:
    return float(np.mean((p - t) ** 2))


def select_mode(values: list[float]) -> float:
    counts: dict[float, int] = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    return sorted(counts.items(), key=lambda i: (-i[1], i[0]))[0][0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["reporting"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    surrogate = CTSurrogate(
        image_side=cfg["geometry"]["image_side"],
        num_angles=cfg["geometry"]["num_angles"],
    )
    lambda_grid = cfg["ablation"]["lambda_grid"]
    train_sizes = cfg["ablation"]["train_sizes"]
    seeds = cfg["ablation"]["seeds"]
    n_pool = cfg["ablation"]["num_train_pool"]
    n_val = cfg["ablation"]["num_val"]
    n_test = cfg["ablation"]["num_test"]

    parameter_counts = {
        "known_operator": surrogate.num_measurements,
        "fully_connected": surrogate.num_pixels * surrogate.num_measurements,
    }

    raw: dict[str, list[dict]] = {"known_operator": [], "fully_connected": []}

    for seed in seeds:
        x_pool, y_pool, x_val, y_val, x_test, y_test = surrogate.build_dataset(seed, n_pool, n_val, n_test)
        for n in train_sizes:
            x_tr = x_pool[:, :n]
            y_tr = y_pool[:, :n]

            best_fc = best_ko = None
            for lam in lambda_grid:
                t0 = time.perf_counter()
                fc = x_tr @ y_tr.T @ np.linalg.inv(y_tr @ y_tr.T + lam * np.eye(surrogate.num_measurements))
                t_fc = time.perf_counter() - t0
                pv = fc @ y_val
                vmse = mse(pv, x_val)
                if best_fc is None or vmse < best_fc["val_mse"]:
                    best_fc = {"lambda": lam, "matrix": fc, "val_mse": vmse, "train_time": t_fc}

                t0 = time.perf_counter()
                weighted = surrogate.backprojection_matrix[None, :, :] * y_tr.T[:, None, :]
                normal = np.einsum("sij,sik->jk", weighted, weighted)
                rhs = np.einsum("sij,si->j", weighted, x_tr.T)
                w = solve(normal + lam * np.eye(surrogate.num_measurements), rhs, assume_a="pos")
                t_ko = time.perf_counter() - t0
                pred_val = np.column_stack(
                    [surrogate.backprojection_matrix @ (w * y_val[:, i]) for i in range(y_val.shape[1])]
                )
                vmse_ko = mse(pred_val, x_val)
                if best_ko is None or vmse_ko < best_ko["val_mse"]:
                    best_ko = {"lambda": lam, "weights": w, "val_mse": vmse_ko, "train_time": t_ko}

            assert best_fc and best_ko

            t0 = time.perf_counter()
            pred = best_fc["matrix"] @ y_test
            t_inf_fc = (time.perf_counter() - t0) * 1e3 / y_test.shape[1]
            raw["fully_connected"].append({
                "seed": seed, "train_size": n,
                "test_mse": mse(pred, x_test),
                "train_time_s": float(best_fc["train_time"]),
                "infer_time_ms": float(t_inf_fc),
                "selected_lambda": float(best_fc["lambda"]),
            })
            t0 = time.perf_counter()
            pred = np.column_stack(
                [surrogate.backprojection_matrix @ (best_ko["weights"] * y_test[:, i]) for i in range(y_test.shape[1])]
            )
            t_inf_ko = (time.perf_counter() - t0) * 1e3 / y_test.shape[1]
            raw["known_operator"].append({
                "seed": seed, "train_size": n,
                "test_mse": mse(pred, x_test),
                "train_time_s": float(best_ko["train_time"]),
                "infer_time_ms": float(t_inf_ko),
                "selected_lambda": float(best_ko["lambda"]),
            })

    rows: list[AggregateRow] = []
    for model_name, runs in raw.items():
        p = parameter_counts[model_name]
        for n in train_sizes:
            sub = [r for r in runs if r["train_size"] == n]
            rows.append(AggregateRow(
                train_size=n, model=model_name,
                mse_mean=float(np.mean([r["test_mse"] for r in sub])),
                mse_std=float(np.std([r["test_mse"] for r in sub], ddof=1)),
                train_time_mean_s=float(np.mean([r["train_time_s"] for r in sub])),
                train_time_std_s=float(np.std([r["train_time_s"] for r in sub], ddof=1)),
                infer_time_mean_ms=float(np.mean([r["infer_time_ms"] for r in sub])),
                infer_time_std_ms=float(np.std([r["infer_time_ms"] for r in sub], ddof=1)),
                selected_lambda_mode=select_mode([r["selected_lambda"] for r in sub]),
                bound_proxy=float(p * math.log(n) / n),
            ))

    payload = {
        "geometry": cfg["geometry"],
        "parameter_counts": parameter_counts,
        "lambda_grid": lambda_grid,
        "train_sizes": train_sizes,
        "seeds": seeds,
        "aggregate": [asdict(r) for r in rows],
        "raw_results": raw,
    }
    (out_dir / "surrogate_results.json").write_text(json.dumps(payload, indent=2))

    with (out_dir / "surrogate_ablation.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    grouped: dict[str, list[AggregateRow]] = {"known_operator": [], "fully_connected": []}
    for row in rows:
        grouped[row.model].append(row)
    for vs in grouped.values():
        vs.sort(key=lambda r: r.train_size)
    names = {"known_operator": "Known operator", "fully_connected": "Fully connected"}
    colors = {"known_operator": "#1b7f5a", "fully_connected": "#b54b32"}
    for name, vs in grouped.items():
        axes[0].errorbar(
            [r.train_size for r in vs], [r.mse_mean for r in vs], yerr=[r.mse_std for r in vs],
            marker="o", linewidth=2, capsize=3, color=colors[name], label=names[name],
        )
        axes[1].plot(
            [r.train_size for r in vs], [r.bound_proxy for r in vs],
            marker="o", linewidth=2, color=colors[name],
            label=f"{names[name]} ($p = {parameter_counts[name]:,}$)",
        )
    axes[0].set_xlabel("Training samples $N$")
    axes[0].set_ylabel("Test MSE")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25, which="both")
    axes[0].legend(frameon=False)
    axes[0].set_title("Test MSE vs. training size")
    axes[1].set_xlabel("Training samples $N$")
    axes[1].set_ylabel(r"$p \log N / N$")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.25, which="both")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].set_title(r"Estimation proxy $p \log N / N$")
    fig.tight_layout()
    fig.savefig(out_dir / "sample_efficiency.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
