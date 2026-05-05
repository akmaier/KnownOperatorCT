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

    # Add the ridge term and solve.  Use linalg.solve via the transpose
    # because we want M (P, Mdim) such that M @ ATA = ATB.
    eye = torch.eye(Mdim, dtype=torch.float32, device=device)
    ATA_reg = ATA + lam * eye
    # solve(ATA_reg.T, ATB.T) returns Z (Mdim, P) with ATA_reg.T @ Z = ATB.T
    # so M = Z.T.
    Z = torch.linalg.solve(ATA_reg.T, ATB.T)
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
    p.add_argument("--lambda", dest="lam", type=float, default=1.0,
                   help="Ridge regularization weight (default 1.0).")
    p.add_argument("--num-samples", type=int, default=2)
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
    print(f"[ridge] train_sizes={train_sizes}, lambda={args.lam}", flush=True)

    # Test set (fixed)
    test_set = materialize(
        geometry, args.num_samples, seed=base_seed + 10_000,
        ellipses=ellipses, device=device,
    )

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
    with torch.no_grad():
        for _, sino in test_set:
            ko_recons.append(ko(sino).detach().cpu().numpy())
    del ko
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ----- Ridge fits at each N -----
    fc_recons_at: dict[int, list] = {}
    fit_times: dict[int, float] = {}
    train_mse: dict[int, float] = {}
    for n in train_sizes:
        seed = base_seed + 1001 + n  # different per N (matches surrogate convention)
        M, dt = fit_ridge(
            geometry, n, args.lam, seed=seed,
            ellipses=ellipses, device=device,
            use_relu=args.use_relu,
        )
        fit_times[n] = dt
        with torch.no_grad():
            fc_recons_at[n] = [
                predict(M, sino, geometry.image_size, args.use_relu)
                .detach().cpu().numpy()
                for _, sino in test_set
            ]
        # Free M aggressively before next fit
        del M
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[ridge] N={n:>5d}  fit_time={dt:.1f}s", flush=True)

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
        f"FC ridge regression progression @ {geometry.image_size}x{geometry.image_size} "
        f"({geometry.num_views} views, λ={args.lam}{relu_tag})",
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
        "lambda": args.lam,
        "use_relu": args.use_relu,
        "fit_times_s": fit_times,
    }, indent=2))
    print(f"[ridge] wrote {out_path}, {json_path}", flush=True)


if __name__ == "__main__":
    main()
