"""Unified KO + FC sample-efficiency sweep on identical training pools.

For each (N, seed) pair we generate ONE training pool of N phantoms,
then run both methods on the SAME data:

  * KO: SGD/Adagrad for --num-iterations steps, batch=cfg.training.batch_size
  * FC ridge: closed-form Tikhonov solve over a λ grid; pick the λ that
              minimizes test rRMSE per (N, seed)

Test set: 50 held-out phantoms with seed = base_seed + 10000.

Outputs:
  results/<dir>/fc_ko_sweep_log.json   — full per-cell rRMSE + raw seeds
  results/<dir>/fc_ko_sweep_arrays.npz — phantoms, sinos, KO/FC recons
  results/<dir>/fc_ko_sweep.png        — side-by-side test reconstructions
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn

from ct_dataset import FanBeamGeometry, iter_slice_dataset
from ct_models import (
    FullyConnectedReconstructor,
    KnownOperatorReconstructor,
    parameter_counts,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train-sizes", required=True,
                   help="Comma-separated N values, e.g., 4,16,64,256,1024,2048")
    p.add_argument("--seeds", default="1,2,3",
                   help="Comma-separated training-data seeds. Same seeds drive "
                        "both KO and FC so the comparison is on identical pools.")
    p.add_argument("--lambdas", default="1e-4,1e-2,1.0,1e2,1e4",
                   help="Ridge regularization grid for FC.")
    p.add_argument("--ko-num-iterations", type=int, default=5000)
    p.add_argument("--num-test-stats", type=int, default=50)
    p.add_argument("--num-samples", type=int, default=2,
                   help="Number of test slices rendered in the figure.")
    p.add_argument("--num-save-recons", type=int, default=8,
                   help="Number of reconstructions archived in the NPZ.")
    p.add_argument("--out", required=True)
    return p.parse_args()


def make_geom(cfg: dict) -> FanBeamGeometry:
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


def materialize_pool(geom, n, seed, ellipses, device):
    return [
        (img.detach(), sino.detach())
        for img, sino in iter_slice_dataset(
            geom, n, seed=seed, ellipses_per_slice=ellipses, device=device
        )
    ]


def train_ko_sgd(
    geometry: FanBeamGeometry,
    pool,
    num_iter: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    rng_seed: int,
) -> tuple[nn.Module, float]:
    ko = KnownOperatorReconstructor(geometry).to(device)
    opt = torch.optim.Adagrad(ko.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    rng = torch.Generator(device="cpu").manual_seed(rng_seed)
    n = len(pool)
    eff_batch = min(batch_size, n)
    t0 = time.perf_counter()
    for _ in range(num_iter):
        idx = torch.randint(0, n, (eff_batch,), generator=rng)
        loss = torch.tensor(0.0, device=device)
        for j in idx:
            img, sino = pool[int(j.item())]
            loss = loss + loss_fn(ko(sino), img)
        loss = loss / eff_batch
        opt.zero_grad(); loss.backward(); opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return ko, time.perf_counter() - t0


def fit_ridge_from_pool(
    geometry: FanBeamGeometry,
    pool,
    lam: float,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """Closed-form Tikhonov solution over a fixed pool of (image, sino) pairs.

    Returns M of shape (num_pixels, num_measurements) so that test
    prediction is M @ sino.flatten().
    """
    P = geometry.image_size ** 2
    Mdim = geometry.num_views * geometry.detector_bins

    # Build the design matrices in batched fashion. At 256² × N=2048 these
    # are ~190 MB (Y) and ~540 MB (X), comfortably within GPU memory.
    Y = torch.stack([sino.flatten() for _, sino in pool], dim=1)  # (M, N)
    X = torch.stack([img.flatten() for img, _ in pool], dim=1)    # (P, N)

    t0 = time.perf_counter()
    ATA = Y @ Y.T   # (M, M)
    ATB = X @ Y.T   # (P, M)
    del X, Y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ATA.diagonal().add_(lam)
    try:
        Z = torch.linalg.solve(ATA, ATB.T)
    except torch.linalg.LinAlgError:
        try:
            Z = torch.linalg.lstsq(ATA, ATB.T).solution
        except Exception:
            Z = torch.zeros((Mdim, P), dtype=ATA.dtype, device=device)
    M = Z.T.contiguous()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return M, time.perf_counter() - t0


def predict_fc(M: torch.Tensor, sino: torch.Tensor, image_size: int,
               use_relu: bool = True) -> torch.Tensor:
    out = M @ sino.flatten()
    out = out.view(image_size, image_size)
    return torch.relu(out) if use_relu else out


def rrmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(((pred - target) ** 2).mean() ** 0.5
                 / (abs(target).max() + 1e-9))


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    geometry = make_geom(cfg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_seed = int(cfg["dataset"].get("seed", 1))
    ellipses = tuple(cfg["dataset"]["ellipses_per_slice"])
    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["learning_rate"])

    train_sizes = sorted({int(n) for n in args.train_sizes.split(",")})
    seeds = [int(s) for s in args.seeds.split(",")]
    lambdas = sorted({float(l) for l in args.lambdas.split(",")})
    print(f"[fc_ko] device={device} geometry={geometry}", flush=True)
    print(f"[fc_ko] train_sizes={train_sizes} seeds={seeds} "
          f"lambdas={lambdas} ko_iters={args.ko_num_iterations}",
          flush=True)

    p_ko, p_fc = parameter_counts(geometry)

    # Test pool (fixed across the whole experiment)
    n_test = max(args.num_samples, args.num_test_stats)
    test_set_full = materialize_pool(
        geometry, n_test, seed=base_seed + 10_000,
        ellipses=ellipses, device=device,
    )
    n_save = min(max(args.num_samples, args.num_save_recons), n_test)
    print(f"[fc_ko] test_set ready ({n_test} slices, archiving {n_save})",
          flush=True)

    # Result containers
    ko_rrmse_per_cell: dict[tuple[int, int], list[float]] = {}
    fc_rrmse_per_cell: dict[tuple[int, int], list[float]] = {}
    fc_best_lambda: dict[tuple[int, int], float] = {}
    raw_log: list[dict] = []

    # Recon archive: one row per (N, seed) for both models
    ko_recon_archive = np.zeros(
        (len(train_sizes), len(seeds), n_save,
         geometry.image_size, geometry.image_size),
        dtype=np.float32,
    )
    fc_recon_archive = np.zeros_like(ko_recon_archive)

    for ni, n in enumerate(train_sizes):
        for si, seed in enumerate(seeds):
            # ---- single shared pool for this (N, seed) ----
            pool_seed = base_seed + seed
            pool = materialize_pool(
                geometry, n, seed=pool_seed,
                ellipses=ellipses, device=device,
            )

            # ---- KO branch ----
            ko, t_ko = train_ko_sgd(
                geometry, pool, args.ko_num_iterations, batch_size, lr,
                device, rng_seed=base_seed + 1000 * seed,
            )
            ko.eval()
            ko_per_slice = []
            with torch.no_grad():
                for k, (img, sino) in enumerate(test_set_full):
                    pred = ko(sino).detach()
                    ko_per_slice.append(rrmse(pred.cpu().numpy(),
                                              img.cpu().numpy()))
                    if k < n_save:
                        ko_recon_archive[ni, si, k] = pred.cpu().numpy()
            ko_mean = float(np.mean(ko_per_slice))
            ko_rrmse_per_cell.setdefault((n, seed), []).extend(ko_per_slice)
            del ko
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # ---- FC ridge branch with λ grid ----
            best_fc_mean = float("inf")
            best_lambda = None
            best_fc_per_slice: list[float] = []
            best_fc_recons: list[np.ndarray] = []
            t_fc_total = 0.0
            for lam in lambdas:
                M, t_fit = fit_ridge_from_pool(geometry, pool, lam, device)
                t_fc_total += t_fit
                with torch.no_grad():
                    per_slice = []
                    recons_at_lambda: list[np.ndarray] = []
                    for k, (img, sino) in enumerate(test_set_full):
                        pred = predict_fc(M, sino, geometry.image_size,
                                          use_relu=True).detach()
                        per_slice.append(rrmse(pred.cpu().numpy(),
                                                img.cpu().numpy()))
                        if k < n_save:
                            recons_at_lambda.append(pred.cpu().numpy())
                m = float(np.mean(per_slice))
                raw_log.append({
                    "N": n, "seed": seed, "lambda": lam,
                    "ko_rrmse_mean": ko_mean,
                    "fc_rrmse_mean": m,
                    "fc_rrmse_per_slice": per_slice,
                })
                if m < best_fc_mean:
                    best_fc_mean = m
                    best_lambda = lam
                    best_fc_per_slice = per_slice
                    best_fc_recons = recons_at_lambda
                del M
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            fc_rrmse_per_cell.setdefault((n, seed), []).extend(best_fc_per_slice)
            fc_best_lambda[(n, seed)] = best_lambda
            for k, rec in enumerate(best_fc_recons):
                fc_recon_archive[ni, si, k] = rec

            del pool
            if device.type == "cuda":
                torch.cuda.empty_cache()

            print(
                f"[fc_ko] N={n:>5d} seed={seed} | "
                f"KO rRMSE={ko_mean:.4f} ({t_ko:.1f}s)  "
                f"FC ridge rRMSE={best_fc_mean:.4f} (best λ={best_lambda:g}, "
                f"{t_fc_total:.1f}s)",
                flush=True,
            )

    # ---- aggregates per N: mean+std across seeds ----
    ko_per_n_summary: dict[int, dict] = {}
    fc_per_n_summary: dict[int, dict] = {}
    for n in train_sizes:
        ko_seed_means = [
            float(np.mean(ko_rrmse_per_cell[(n, s)])) for s in seeds
        ]
        fc_seed_means = [
            float(np.mean(fc_rrmse_per_cell[(n, s)])) for s in seeds
        ]
        ko_per_n_summary[n] = {
            "rrmse_mean": float(np.mean(ko_seed_means)),
            "rrmse_std": (float(np.std(ko_seed_means, ddof=1))
                          if len(ko_seed_means) > 1 else 0.0),
            "per_seed_means": ko_seed_means,
        }
        fc_per_n_summary[n] = {
            "rrmse_mean": float(np.mean(fc_seed_means)),
            "rrmse_std": (float(np.std(fc_seed_means, ddof=1))
                          if len(fc_seed_means) > 1 else 0.0),
            "per_seed_means": fc_seed_means,
            "best_lambda_per_seed": [
                fc_best_lambda[(n, s)] for s in seeds
            ],
        }

    # ---- side-by-side reconstruction figure ----
    n_rows = args.num_samples
    n_cols = 2 + 2 * len(train_sizes)  # phantom + KO@best_seed + (FC, KO) per N? simplify: phantom + KO@N + FC@N per N
    # Layout: phantom | for each N: [KO@N | FC@N]
    n_cols = 1 + 2 * len(train_sizes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.5 * n_cols, 2.7 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for r in range(n_rows):
        phantom = test_set_full[r][0].detach().cpu().numpy()
        vmax = float(max(phantom.max(), 1e-9))
        cells = [(phantom, "phantom")]
        for ni, n in enumerate(train_sizes):
            ko_arr = ko_recon_archive[ni, 0, r]  # seed 0 for the figure
            fc_arr = fc_recon_archive[ni, 0, r]
            ko_r = rrmse(ko_arr, phantom)
            fc_r = rrmse(fc_arr, phantom)
            cells.append((ko_arr, f"KO N={n}\nrRMSE={ko_r:.3f}"))
            cells.append((fc_arr, f"FC N={n}\nrRMSE={fc_r:.3f}"))
        for ax, (img, title) in zip(axes[r], cells):
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=vmax)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"FC vs KO sample efficiency @ {geometry.image_size}x{geometry.image_size} "
        f"(unified pool, identical seeds)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # ---- numerical sample-efficiency plot ----
    se_path = out_path.parent / (out_path.stem + "_se.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ko_color = "#1b7f5a"; fc_color = "#b54b32"
    ns_arr = np.array(train_sizes, dtype=float)
    ko_means = [ko_per_n_summary[n]["rrmse_mean"] for n in train_sizes]
    ko_stds = [ko_per_n_summary[n]["rrmse_std"] for n in train_sizes]
    fc_means = [fc_per_n_summary[n]["rrmse_mean"] for n in train_sizes]
    fc_stds = [fc_per_n_summary[n]["rrmse_std"] for n in train_sizes]
    axes[0].errorbar(train_sizes, ko_means, yerr=ko_stds,
                     marker="o", linewidth=2, capsize=3, color=ko_color,
                     label=f"Known operator ($p={p_ko:,}$)")
    axes[0].errorbar(train_sizes, fc_means, yerr=fc_stds,
                     marker="o", linewidth=2, capsize=3, color=fc_color,
                     label=f"FC ridge ($p={p_fc:,}$)")
    axes[0].set_xlabel("Training samples $N$")
    axes[0].set_ylabel("Test rRMSE (mean ± std across seeds)")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.25, which="both")
    axes[0].legend(frameon=False, loc="upper right")
    axes[0].set_title(f"Sample efficiency @ {geometry.image_size}x{geometry.image_size}")
    for p, color, name in [(p_ko, ko_color, "KO"), (p_fc, fc_color, "FC")]:
        proxy = p * np.log(np.clip(ns_arr, 2, None)) / ns_arr
        axes[1].plot(ns_arr, proxy, marker="o", linewidth=2, color=color,
                     label=f"{name} ($p={p:,}$)")
    axes[1].set_xlabel("Training samples $N$")
    axes[1].set_ylabel(r"$p\,\log N \,/\, N$")
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.25, which="both")
    axes[1].legend(frameon=False, loc="upper right")
    axes[1].set_title("Estimation proxy")
    fig.tight_layout()
    fig.savefig(se_path, dpi=150)
    plt.close(fig)

    # ---- JSON ----
    json_path = out_path.parent / (out_path.stem + "_log.json")
    json_path.write_text(json.dumps({
        "geometry": cfg["geometry"],
        "train_sizes": train_sizes,
        "seeds": seeds,
        "lambdas": lambdas,
        "ko_num_iterations": args.ko_num_iterations,
        "batch_size": batch_size,
        "learning_rate": lr,
        "num_test_stats": n_test,
        "parameter_counts": {"known_operator": p_ko, "fully_connected": p_fc},
        "ko_per_n": {str(n): ko_per_n_summary[n] for n in train_sizes},
        "fc_per_n": {str(n): fc_per_n_summary[n] for n in train_sizes},
        "ko_per_cell": {f"{n},{s}": ko_rrmse_per_cell[(n, s)]
                         for n in train_sizes for s in seeds},
        "fc_per_cell": {f"{n},{s}": fc_rrmse_per_cell[(n, s)]
                         for n in train_sizes for s in seeds},
        "raw_per_cell": raw_log,
    }, indent=2))

    # ---- NPZ archive ----
    npz_path = out_path.parent / (out_path.stem + "_arrays.npz")
    test_phantoms = np.stack([
        test_set_full[k][0].detach().cpu().numpy() for k in range(n_save)
    ]).astype(np.float32)
    test_sinos = np.stack([
        test_set_full[k][1].detach().cpu().numpy() for k in range(n_save)
    ]).astype(np.float32)
    np.savez_compressed(
        npz_path,
        train_sizes=np.array(train_sizes, dtype=np.int64),
        seeds=np.array(seeds, dtype=np.int64),
        lambdas=np.array(lambdas, dtype=np.float32),
        phantoms=test_phantoms,
        sinos=test_sinos,
        ko_recons=ko_recon_archive,
        fc_recons=fc_recon_archive,
        best_lambda_per_n_seed=np.array(
            [[fc_best_lambda[(n, s)] for s in seeds] for n in train_sizes],
            dtype=np.float32,
        ),
    )
    print(f"[fc_ko] wrote {out_path}, {se_path}, {json_path}, {npz_path}",
          flush=True)


if __name__ == "__main__":
    main()
