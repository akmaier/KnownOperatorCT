"""GPU sample-efficiency sweep for known-operator vs fully-connected CT.

For each (model_kind, training_set_size, seed) cell, this script:
  1. Generates a fresh training pool of size N (deterministic via seed).
  2. Trains the model from scratch with Adagrad for ``num_iterations`` steps.
  3. Evaluates the test MSE on a fixed held-out set.

Outputs in ``out_dir``:
  * sample_efficiency_gpu_results.json — raw per-cell records + aggregates.
  * sample_efficiency_gpu.csv          — flat table of aggregates.
  * sample_efficiency_gpu.png          — Test MSE vs N (left) and bound proxy
                                         p log N / N (right) for both models.

Usage:
  python src/sample_efficiency_gpu.py --config configs/ct_sample_efficiency_128.yaml
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
import torch
import yaml
from torch import nn

from ct_dataset import FanBeamGeometry, iter_slice_dataset
from ct_models import (
    FullyConnectedReconstructor,
    KnownOperatorReconstructor,
    parameter_counts,
)


@dataclass
class CellResult:
    model: str
    train_size: int
    seed: int
    test_mse: float
    train_time_s: float
    infer_time_ms: float
    final_train_loss: float


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
    bound_proxy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--models",
                        default="known_operator,fully_connected",
                        help="Comma-separated list of models to sweep "
                             "(default: both). Use 'known_operator' alone "
                             "to skip the FC SGD path when ridge results "
                             "are already in hand.")
    parser.add_argument("--num-iterations", type=int, default=None,
                        help="Override training.num_iterations from config.")
    parser.add_argument("--seeds", default=None,
                        help="Override ablation.seeds (comma-separated).")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override training.batch_size from config.")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def make_geometry(cfg: dict) -> FanBeamGeometry:
    g = cfg["geometry"]
    return FanBeamGeometry(
        image_size=g["image_size"],
        num_views=g["num_views"],
        detector_bins=g["detector_bins"],
        angular_range_degrees=g["angular_range_degrees"],
        source_to_iso_mm=g["source_to_iso_mm"],
        source_to_detector_mm=g["source_to_detector_mm"],
        detector_pixel_mm=g["detector_pixel_mm"],
    )


def materialize_pool(
    geometry: FanBeamGeometry,
    num_slices: int,
    seed: int,
    ellipses_per_slice: tuple[int, int],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    pool: list[tuple[torch.Tensor, torch.Tensor]] = []
    for image, sino in iter_slice_dataset(
        geometry, num_slices, seed=seed,
        ellipses_per_slice=ellipses_per_slice, device=device,
    ):
        pool.append((image.detach(), sino.detach()))
    return pool


def build_model(kind: str, geometry: FanBeamGeometry, device: torch.device) -> nn.Module:
    if kind == "known_operator":
        return KnownOperatorReconstructor(geometry).to(device)
    if kind == "fully_connected":
        return FullyConnectedReconstructor.from_geometry(geometry).to(device)
    raise SystemExit(f"Unknown model: {kind}")


def train_cell(
    model: nn.Module,
    train_set: list[tuple[torch.Tensor, torch.Tensor]],
    num_iterations: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    rng_seed: int,
) -> tuple[float, float]:
    """Train ``model`` for ``num_iterations`` steps, return (train_time_s, last_loss)."""
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    rng = torch.Generator(device="cpu").manual_seed(rng_seed)
    n = len(train_set)
    eff_batch = min(batch_size, n)

    last_loss = float("nan")
    t0 = time.perf_counter()
    for _ in range(num_iterations):
        idx = torch.randint(0, n, (eff_batch,), generator=rng)
        loss = torch.tensor(0.0, device=device)
        for sample_idx in idx:
            image, sino = train_set[int(sample_idx.item())]
            recon = model(sino)
            loss = loss + loss_fn(recon, image)
        loss = loss / eff_batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = float(loss.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - t0, last_loss


def evaluate_cell(
    model: nn.Module,
    test_set: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[float, float]:
    """Return (mean_test_mse, mean_inference_time_ms)."""
    model.eval()
    losses: list[float] = []
    times_ms: list[float] = []
    with torch.no_grad():
        for image, sino in test_set:
            t0 = time.perf_counter()
            recon = model(sino)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times_ms.append(1000.0 * (time.perf_counter() - t0))
            losses.append(float(((recon - image) ** 2).mean().detach().cpu()))
    model.train()
    return (
        float(sum(losses) / max(1, len(losses))),
        float(sum(times_ms) / max(1, len(times_ms))),
    )


def aggregate(records: list[CellResult], parameter_counts_d: dict) -> list[AggregateRow]:
    by_key: dict[tuple[str, int], list[CellResult]] = {}
    for r in records:
        by_key.setdefault((r.model, r.train_size), []).append(r)

    def stat(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        m = sum(values) / len(values)
        if len(values) < 2:
            return m, 0.0
        var = sum((v - m) ** 2 for v in values) / (len(values) - 1)
        return m, math.sqrt(var)

    rows: list[AggregateRow] = []
    for (model_name, n), runs in by_key.items():
        m_mse, s_mse = stat([r.test_mse for r in runs])
        m_tt, s_tt = stat([r.train_time_s for r in runs])
        m_it, s_it = stat([r.infer_time_ms for r in runs])
        rows.append(AggregateRow(
            train_size=n, model=model_name,
            mse_mean=m_mse, mse_std=s_mse,
            train_time_mean_s=m_tt, train_time_std_s=s_tt,
            infer_time_mean_ms=m_it, infer_time_std_ms=s_it,
            bound_proxy=parameter_counts_d[model_name] * math.log(max(2, n)) / max(1, n),
        ))
    rows.sort(key=lambda r: (r.model, r.train_size))
    return rows


def write_plot(
    rows: list[AggregateRow],
    parameter_counts_d: dict,
    out_path: Path,
) -> None:
    grouped: dict[str, list[AggregateRow]] = {"known_operator": [], "fully_connected": []}
    for r in rows:
        grouped.setdefault(r.model, []).append(r)
    for vs in grouped.values():
        vs.sort(key=lambda r: r.train_size)

    names = {"known_operator": "Known operator", "fully_connected": "Fully connected"}
    colors = {"known_operator": "#1b7f5a", "fully_connected": "#b54b32"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, vs in grouped.items():
        if not vs:
            continue
        axes[0].errorbar(
            [r.train_size for r in vs], [r.mse_mean for r in vs],
            yerr=[r.mse_std for r in vs],
            marker="o", linewidth=2, capsize=3, color=colors[name], label=names[name],
        )
        axes[1].plot(
            [r.train_size for r in vs], [r.bound_proxy for r in vs],
            marker="o", linewidth=2, color=colors[name],
            label=f"{names[name]} ($p = {parameter_counts_d[name]:,}$)",
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
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    out_dir = Path(cfg["reporting"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    geometry = make_geometry(cfg)

    p_ko, p_fc = parameter_counts(geometry)
    parameter_counts_d = {"known_operator": p_ko, "fully_connected": p_fc}

    train_sizes: list[int] = list(cfg["ablation"]["train_sizes"])
    seeds: list[int] = list(cfg["ablation"]["seeds"])
    num_test_slices = int(cfg["dataset"]["num_test_slices"])
    ellipses = tuple(cfg["dataset"]["ellipses_per_slice"])
    base_seed = int(cfg["dataset"].get("seed", 1))

    num_iterations = int(cfg["training"]["num_iterations"])
    batch_size = int(cfg["training"]["batch_size"])
    learning_rate = float(cfg["training"]["learning_rate"])

    print(f"[sample_efficiency_gpu] device={device}", flush=True)
    print(f"[sample_efficiency_gpu] geometry={geometry}", flush=True)
    print(f"[sample_efficiency_gpu] p_ko={p_ko}, p_fc={p_fc}", flush=True)
    print(f"[sample_efficiency_gpu] train_sizes={train_sizes}, seeds={seeds}, iters={num_iterations}", flush=True)

    test_set = materialize_pool(
        geometry, num_test_slices, seed=base_seed + 10_000,
        ellipses_per_slice=ellipses, device=device,
    )
    print(f"[sample_efficiency_gpu] built test set ({len(test_set)} slices)", flush=True)

    max_n = max(train_sizes)
    records: list[CellResult] = []
    for seed in seeds:
        pool = materialize_pool(
            geometry, max_n, seed=base_seed + seed,
            ellipses_per_slice=ellipses, device=device,
        )
        for n in train_sizes:
            train_set = pool[:n]
            for kind in ("known_operator", "fully_connected"):
                t_run0 = time.perf_counter()
                model = build_model(kind, geometry, device)
                train_t, last_loss = train_cell(
                    model, train_set, num_iterations, batch_size,
                    learning_rate, device, rng_seed=base_seed + 1000 * seed + n,
                )
                test_mse, infer_ms = evaluate_cell(model, test_set, device)
                wall = time.perf_counter() - t_run0
                records.append(CellResult(
                    model=kind, train_size=n, seed=seed,
                    test_mse=test_mse, train_time_s=train_t,
                    infer_time_ms=infer_ms, final_train_loss=last_loss,
                ))
                print(
                    f"[sample_efficiency_gpu] {kind} N={n:>5d} seed={seed} "
                    f"test_mse={test_mse:.4e} infer_ms={infer_ms:.2f} "
                    f"train_s={train_t:.1f} (cell wall={wall:.1f}s, last_loss={last_loss:.4e})",
                    flush=True,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    rows = aggregate(records, parameter_counts_d)

    payload = {
        "geometry": cfg["geometry"],
        "parameter_counts": parameter_counts_d,
        "train_sizes": train_sizes,
        "seeds": seeds,
        "num_iterations": num_iterations,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": str(device),
        "raw_results": [asdict(r) for r in records],
        "aggregate": [asdict(r) for r in rows],
    }
    (out_dir / "sample_efficiency_gpu_results.json").write_text(json.dumps(payload, indent=2))

    if rows:
        with (out_dir / "sample_efficiency_gpu.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))

    write_plot(rows, parameter_counts_d, out_dir / "sample_efficiency_gpu.png")
    print(f"[sample_efficiency_gpu] wrote results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
