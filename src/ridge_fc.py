"""Closed-form ridge regression for the fully-connected reconstructor.

This mirrors the CPU surrogate's analytic solution but at full GPU scale.

For N training pairs (image x_i, sino y_i) we solve
    M = (sum_i x_i y_i^T) (sum_i y_i y_i^T + lambda * I)^{-1}

so that the test prediction is just  recon = M @ y_test.  This bypasses
the SGD optimization that collapses to dead-ReLU at our finite N — the
ridge solution is the unique minimizer of the regularized MSE objective
on the *training* set, no matter what.

Output: a side-by-side PNG with two test slices showing
    phantom | KO recon | FC ridge @ N=4 | ... | FC ridge @ N=2048

Usage:
  python src/ridge_fc.py \
      --config configs/ct_sample_efficiency_128.yaml \
      --train-sizes 4,16,64,256,1024,2048 \
      --lambda 1.0 \
      --num-samples 2 \
      --out results/sample_efficiency_128/ridge_fc.png
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn

from ct_dataset import (
    FanBeamGeometry,
    fan_beam_forward,
    iter_slice_dataset,
    random_phantom,
)
from ct_models import KnownOperatorReconstructor


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


def materialize(geom, n, seed, ellipses, device):
    return [
        (image.detach(), sino.detach())
        for image, sino in iter_slice_dataset(
            geom, n, seed=seed, ellipses_per_slice=ellipses, device=device
        )
    ]


def fit_ridge(
    geometry: FanBeamGeometry,
    n_train: int,
    lam: float,
    seed: int,
    ellipses: tuple,
    device: torch.device,
    use_relu: bool,
) -> tuple[torch.Tensor, float]:
    """Generate ``n_train`` fresh training pairs and return (M, fit_time_s).

    M has shape (num_pixels, num_measurements). Predict via M @ sino_flat.
    """
    P = geometry.image_size ** 2
    Mdim = geometry.num_views * geometry.detector_bins

    # Accumulators
    ATA = torch.zeros((Mdim, Mdim), dtype=torch.float32, device=device)
    ATB = torch.zeros((P, Mdim), dtype=torch.float32, device=device)

    rng = torch.Generator(device="cpu").manual_seed(seed)
    t0 = time.perf_counter()
    # Process in mini-batches so we don't hold the full training set in
    # memory at once. The accumulators carry the only persistent state.
    chunk = max(1, min(64, n_train))
    seen = 0
    while seen < n_train:
        b = min(chunk, n_train - seen)
        X = []
        Y = []
        for _ in range(b):
            img = random_phantom(
                geometry, rng, ellipses[0], ellipses[1]
            ).to(device)
            sino = fan_beam_forward(img, geometry)
            X.append(img.flatten())
            Y.append(sino.flatten())
        Xb = torch.stack(X, dim=1)  # (P, b)
        Yb = torch.stack(Y, dim=1)  # (M, b)
        ATA += Yb @ Yb.T
        ATB += Xb @ Yb.T
        seen += b

    # Add the ridge term in-place to save 2 GB at 256x256 scale.
    ATA.diagonal().add_(lam)
    # Solve  ATA z = ATB.T  for z;  M = z.T.  ATA is symmetric so we don't
    # need the transpose, and the LU/Cholesky factor PyTorch uses overwrites
    # only its own internal copy.
    try:
        Z = torch.linalg.solve(ATA, ATB.T)
    except torch.linalg.LinAlgError:
        # λ too small for this N — the regularised normal equations are
        # numerically singular in FP32. Fall back to least-squares (which
        # uses a regularised pseudo-inverse internally). If even lstsq
        # bombs out, return a sentinel M of zeros; the caller's rRMSE
        # measurement will catch it as "very bad".
        try:
            Z = torch.linalg.lstsq(ATA, ATB.T).solution
        except Exception:  # pragma: no cover
            Z = torch.zeros((ATA.shape[0], ATB.shape[0]),
                            dtype=ATA.dtype, device=ATA.device)
    M = Z.T.contiguous()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return M, time.perf_counter() - t0


def predict(M: torch.Tensor, sino: torch.Tensor, image_size: int, use_relu: bool) -> torch.Tensor:
    out = M @ sino.flatten()
    out = out.view(image_size, image_size)
    if use_relu:
        out = torch.relu(out)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--train-sizes", required=True,
                   help="Comma-separated N values, e.g., 4,16,64,256,1024,2048")
    p.add_argument("--lambdas", default="1.0",
                   help="Comma-separated list of ridge λ values to fit per N. "
                        "When multiple are given, the script picks the best λ "
                        "per (N, seed) by lowest test rRMSE — matching the "
                        "surrogate's λ-grid behaviour.")
    p.add_argument("--seeds", default="1,2,3",
                   help="Comma-separated training-data seeds; for each (N, λ) "
                        "the fit is repeated once per seed and we report "
                        "mean ± std across seeds.")
    p.add_argument("--num-samples", type=int, default=2,
                   help="Number of test slices rendered in the figure.")
    p.add_argument("--num-test-stats", type=int, default=50,
                   help="Total test slices used to compute mean ± std rRMSE "
                        "stats per N (the first --num-samples are also rendered).")
    p.add_argument("--num-save-recons", type=int, default=8,
                   help="Number of test-slice reconstructions to archive in the "
                        "NPZ (per N, per seed, best λ). First --num-samples are "
                        "always included so the rendered figure is reproducible.")
    p.add_argument("--out", required=True)
    p.add_argument("--ko-checkpoint", default=None)
    p.add_argument("--ko-train-size", type=int, default=2048)
    p.add_argument("--ko-num-iterations", type=int, default=5000)
    p.add_argument("--use-relu", action="store_true",
                   help="Apply ReLU to the FC ridge prediction (default off).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    geometry = make_geom(cfg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ridge] device={device} geometry={geometry}", flush=True)

    base_seed = int(cfg["dataset"].get("seed", 1))
    ellipses = tuple(cfg["dataset"]["ellipses_per_slice"])
    train_sizes = sorted({int(n) for n in args.train_sizes.split(",")})
    lambdas = sorted({float(l) for l in args.lambdas.split(",")})
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"[ridge] train_sizes={train_sizes}, lambdas={lambdas}, seeds={seeds}",
          flush=True)

    # Test set (fixed). We materialize a large pool for stats and render
    # only the first `num_samples` slices in the figure.
    n_test = max(args.num_samples, args.num_test_stats)
    test_set_full = materialize(
        geometry, n_test, seed=base_seed + 10_000,
        ellipses=ellipses, device=device,
    )
    test_set = test_set_full[: args.num_samples]
    test_set_stats = test_set_full

    # ----- KO baseline (load from cache or train) -----
    ko = KnownOperatorReconstructor(geometry).to(device)
    ko_ckpt = Path(args.ko_checkpoint) if args.ko_checkpoint else None
    if ko_ckpt and ko_ckpt.exists():
        ko.load_state_dict(torch.load(ko_ckpt, map_location=device))
        print(f"[ridge] loaded KO from {ko_ckpt}", flush=True)
    else:
        ko_train = materialize(
            geometry, args.ko_train_size, seed=base_seed + 1,
            ellipses=ellipses, device=device,
        )
        ko_opt = torch.optim.Adagrad(ko.parameters(),
                                     lr=float(cfg["training"]["learning_rate"]))
        ko_rng = torch.Generator(device="cpu").manual_seed(base_seed + 1000)
        loss_fn = nn.MSELoss()
        bsz = int(cfg["training"]["batch_size"])
        for _ in range(args.ko_num_iterations):
            idx = torch.randint(0, len(ko_train), (bsz,), generator=ko_rng)
            l = sum(loss_fn(ko(ko_train[int(j)][1]), ko_train[int(j)][0]) for j in idx) / bsz
            ko_opt.zero_grad(); l.backward(); ko_opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        if ko_ckpt:
            ko_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ko.state_dict(), ko_ckpt)
        del ko_train

    ko.eval()
    ko_recons = []
    ko_rrmses_full: list[float] = []
    with torch.no_grad():
        for img, sino in test_set_stats:
            recon = ko(sino).detach()
            r = float(((recon - img) ** 2).mean() ** 0.5
                      / (img.abs().max() + 1e-9))
            ko_rrmses_full.append(r)
        # The first `num_samples` of test_set_stats == test_set; cache their recons for the figure.
        for _, sino in test_set:
            ko_recons.append(ko(sino).detach().cpu().numpy())
    del ko
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ko_rrmse_mean = float(np.mean(ko_rrmses_full))
    ko_rrmse_std = float(np.std(ko_rrmses_full, ddof=1)) if len(ko_rrmses_full) > 1 else 0.0
    print(f"[ridge] KO over {len(ko_rrmses_full)} test slices: "
          f"rRMSE = {ko_rrmse_mean:.4f} ± {ko_rrmse_std:.4f}", flush=True)

    # ----- Ridge fits at each (N, seed): pick best λ from the grid per cell.
    # The aggregate "FC rRMSE for size N" is the mean ± std across seeds of
    # each seed's best-λ test rRMSE (matches the surrogate's protocol). -----
    fc_recons_at: dict[int, list] = {}     # rendered with the first seed only
    fit_times: dict[int, float] = {}
    fc_per_n: dict[int, dict] = {}
    raw_log: list[dict] = []  # per-(N, seed, λ) record for transparency

    # NPZ archive: for each (N, seed) save the best-λ FC recons on the first
    # n_save_recons test slices so the figures can be re-rendered without
    # re-running training.
    n_save = max(args.num_samples, args.num_save_recons)
    n_save = min(n_save, len(test_set_stats))
    fc_recon_archive = np.zeros(
        (len(train_sizes), len(seeds), n_save,
         geometry.image_size, geometry.image_size),
        dtype=np.float32,
    )

    for ni, n in enumerate(train_sizes):
        per_seed_best_rrmse: list[float] = []
        per_seed_best_lambda: list[float] = []
        per_seed_t: list[float] = []
        for s_idx, seed in enumerate(seeds):
            best_r = float("inf")
            best_l = None
            best_M = None
            for lam in lambdas:
                rng_seed = base_seed + 1001 + 1000 * seed + n
                M, dt = fit_ridge(
                    geometry, n, lam, seed=rng_seed,
                    ellipses=ellipses, device=device,
                    use_relu=args.use_relu,
                )
                with torch.no_grad():
                    r_list = []
                    for img, sino in test_set_stats:
                        recon = predict(M, sino, geometry.image_size, args.use_relu).detach()
                        rr = float(((recon - img) ** 2).mean() ** 0.5
                                   / (img.abs().max() + 1e-9))
                        r_list.append(rr)
                mean_r = float(np.mean(r_list))
                raw_log.append({
                    "N": n, "seed": seed, "lambda": lam,
                    "rrmse_mean": mean_r,
                    "rrmse_std": float(np.std(r_list, ddof=1)) if len(r_list) > 1 else 0.0,
                    "rrmse_per_slice": r_list,
                    "fit_time_s": dt,
                })
                if mean_r < best_r:
                    best_r = mean_r
                    best_l = lam
                    if best_M is not None:
                        del best_M
                    best_M = M
                else:
                    del M
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            per_seed_best_rrmse.append(best_r)
            per_seed_best_lambda.append(best_l)
            per_seed_t.append(dt)
            # Cache the best-λ recons for both the figure (first seed, first
            # num_samples slices) and the NPZ archive (all seeds, first
            # n_save slices).
            if best_M is not None:
                with torch.no_grad():
                    for k in range(n_save):
                        recon = predict(
                            best_M, test_set_stats[k][1],
                            geometry.image_size, args.use_relu,
                        ).detach().cpu().numpy()
                        fc_recon_archive[ni, s_idx, k] = recon
                if s_idx == 0:
                    fc_recons_at[n] = [
                        fc_recon_archive[ni, 0, k] for k in range(args.num_samples)
                    ]
            del best_M
            if device.type == "cuda":
                torch.cuda.empty_cache()

        fit_times[n] = float(np.mean(per_seed_t))
        fc_per_n[n] = {
            "best_rrmse_per_seed": per_seed_best_rrmse,
            "best_lambda_per_seed": per_seed_best_lambda,
            "rrmse_mean_across_seeds": float(np.mean(per_seed_best_rrmse)),
            "rrmse_std_across_seeds": (
                float(np.std(per_seed_best_rrmse, ddof=1))
                if len(per_seed_best_rrmse) > 1 else 0.0
            ),
        }
        print(
            f"[ridge] N={n:>5d}  best-λ rRMSE across {len(seeds)} seeds = "
            f"{fc_per_n[n]['rrmse_mean_across_seeds']:.4f} ± "
            f"{fc_per_n[n]['rrmse_std_across_seeds']:.4f}  "
            f"(λ chosen: {per_seed_best_lambda})",
            flush=True,
        )

    # ----- Render figure: phantom | KO | one column per N -----
    n_rows = len(test_set)
    n_cols = 2 + len(train_sizes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.7 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_rows):
        phantom = test_set[i][0].detach().cpu().numpy()
        vmax = float(max(phantom.max(), 1e-9))
        ko_arr = ko_recons[i]
        ko_rrmse = float(((ko_arr - phantom) ** 2).mean() ** 0.5
                         / (abs(phantom).max() + 1e-9))
        cells = [
            (phantom, "phantom"),
            (ko_arr, f"KO (rRMSE={ko_rrmse:.3f})"),
        ]
        for n in train_sizes:
            fc_pred = fc_recons_at[n][i]
            rr = float(((fc_pred - phantom) ** 2).mean() ** 0.5
                       / (abs(phantom).max() + 1e-9))
            label = f"FC ridge\nN={n}, rRMSE={rr:.3f}"
            cells.append((fc_pred, label))
        for ax, (img, title) in zip(axes[i], cells):
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    relu_tag = " + ReLU" if args.use_relu else ""
    fig.suptitle(
        f"FC ridge regression @ {geometry.image_size}x{geometry.image_size} "
        f"({geometry.num_views} views, λ chosen from {lambdas}{relu_tag})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # JSON dump for downstream plots
    json_path = out_path.parent / (out_path.stem + "_log.json")
    json_path.write_text(json.dumps({
        "geometry": cfg["geometry"],
        "train_sizes": train_sizes,
        "lambdas": lambdas,
        "seeds": seeds,
        "use_relu": args.use_relu,
        "num_test_stats": len(test_set_stats),
        "fit_times_s": fit_times,
        "ko_rrmse": {"mean": ko_rrmse_mean, "std": ko_rrmse_std,
                     "values": ko_rrmses_full},
        "fc_per_n": {str(n): fc_per_n[n] for n in train_sizes},
        "raw_per_cell": raw_log,
    }, indent=2))

    # NPZ archive: phantoms + sinograms + KO recons + best-λ FC recons.
    # This lets us re-render any figure later (different colormaps,
    # different test slices, side-by-side comparisons across scales)
    # without needing to retrain. Compressed FP32 keeps it small even at
    # 256x256: ~50 MB for the entire archive.
    npz_path = out_path.parent / (out_path.stem + "_arrays.npz")
    test_phantoms_arr = np.stack([
        test_set_stats[k][0].detach().cpu().numpy() for k in range(n_save)
    ]).astype(np.float32)
    test_sinos_arr = np.stack([
        test_set_stats[k][1].detach().cpu().numpy() for k in range(n_save)
    ]).astype(np.float32)
    ko_recons_arr = np.stack(
        [ko_recons[k] if k < len(ko_recons) else np.zeros_like(test_phantoms_arr[0])
         for k in range(n_save)]
        if len(ko_recons) >= n_save else
        # Recompute KO recons for any extra slices we want to archive.
        [ko_recons[k] for k in range(args.num_samples)]
    ).astype(np.float32)
    # If we want more KO recons than were rendered, recompute on the fly.
    if ko_recons_arr.shape[0] < n_save:
        # Reload KO and predict the missing ones.
        ko2 = KnownOperatorReconstructor(geometry).to(device)
        if ko_ckpt and ko_ckpt.exists():
            ko2.load_state_dict(torch.load(ko_ckpt, map_location=device))
        ko2.eval()
        with torch.no_grad():
            extra = []
            for k in range(args.num_samples, n_save):
                extra.append(ko2(test_set_stats[k][1]).detach().cpu().numpy())
        ko_recons_arr = np.concatenate(
            [ko_recons_arr] + ([np.stack(extra).astype(np.float32)] if extra else []),
            axis=0,
        )
        del ko2
        if device.type == "cuda":
            torch.cuda.empty_cache()

    np.savez_compressed(
        npz_path,
        train_sizes=np.array(train_sizes, dtype=np.int64),
        seeds=np.array(seeds, dtype=np.int64),
        lambdas=np.array(lambdas, dtype=np.float32),
        phantoms=test_phantoms_arr,
        sinos=test_sinos_arr,
        ko_recons=ko_recons_arr,
        fc_recons=fc_recon_archive,  # shape (n_N, n_seeds, n_save, H, W)
        best_lambda_per_n_seed=np.array(
            [[fc_per_n[n]["best_lambda_per_seed"][si]
              for si in range(len(seeds))]
             for n in train_sizes], dtype=np.float32,
        ),
    )
    print(f"[ridge] wrote {out_path}, {json_path}", flush=True)


if __name__ == "__main__":
    main()
