"""Render FC training progression: how does test reconstruction quality
evolve as the iteration budget grows from 100 to 50,000+?

Trains a single FC model to the largest checkpoint with on-the-fly fresh
phantoms (so storage stays at zero), saves the test reconstructions at
each intermediate milestone, then plots:

  results/<dir>/iter_progression.png       — phantom | KO | FC@cp1 | FC@cp2 | ...
  results/<dir>/iter_progression_loss.png  — training loss vs. iteration
  results/<dir>/iter_progression_log.json  — raw loss trajectory

Usage:
  python src/render_iter_progression.py \
      --config configs/ct_sample_efficiency_128.yaml \
      --checkpoints 100,1000,10000,50000 \
      --out results/sample_efficiency_128/iter_progression.png \
      --ko-checkpoint results/checkpoints/cache/ko_ct_sample_efficiency_128_n2048_i5000.pt
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

from ct_dataset import (
    FanBeamGeometry,
    fan_beam_forward,
    iter_slice_dataset,
    random_phantom,
)
from ct_models import (
    FullyConnectedReconstructor,
    KnownOperatorReconstructor,
)


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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoints", required=True,
                   help="Comma-separated iter milestones, e.g. 100,1000,10000,50000")
    p.add_argument("--out", required=True, help="Path to the side-by-side PNG.")
    p.add_argument("--num-samples", type=int, default=2,
                   help="Number of test slices rendered.")
    p.add_argument("--ko-checkpoint", default=None,
                   help="KO weights cache. Loaded if it exists; otherwise "
                        "trained and saved here.")
    p.add_argument("--ko-train-size", type=int, default=2048)
    p.add_argument("--ko-num-iterations", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override config's training.batch_size for both KO and FC.")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    geometry = make_geom(cfg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[progression] device={device} geometry={geometry}", flush=True)

    base_seed = int(cfg["dataset"].get("seed", 1))
    ellipses = tuple(cfg["dataset"]["ellipses_per_slice"])
    batch_size = (
        args.batch_size if args.batch_size is not None
        else int(cfg["training"]["batch_size"])
    )
    lr = float(cfg["training"]["learning_rate"])
    print(f"[progression] batch_size={batch_size}", flush=True)

    checkpoints = sorted({int(c) for c in args.checkpoints.split(",")})
    print(f"[progression] checkpoints: {checkpoints}", flush=True)

    # Test set (fixed across the whole experiment)
    test_set = materialize(
        geometry, args.num_samples, seed=base_seed + 10_000,
        ellipses=ellipses, device=device,
    )

    # ------- KO baseline -------
    ko = KnownOperatorReconstructor(geometry).to(device)
    ko_ckpt = Path(args.ko_checkpoint) if args.ko_checkpoint else None
    if ko_ckpt and ko_ckpt.exists():
        ko.load_state_dict(torch.load(ko_ckpt, map_location=device))
        print(f"[progression] loaded KO from {ko_ckpt}", flush=True)
    else:
        ko_train = materialize(
            geometry, args.ko_train_size, seed=base_seed + 1,
            ellipses=ellipses, device=device,
        )
        ko_opt = torch.optim.Adagrad(ko.parameters(), lr=lr)
        ko_rng = torch.Generator(device="cpu").manual_seed(base_seed + 1000)
        loss_fn = nn.MSELoss()
        t0 = time.perf_counter()
        for _ in range(args.ko_num_iterations):
            idx = torch.randint(0, len(ko_train), (batch_size,), generator=ko_rng)
            loss = torch.tensor(0.0, device=device)
            for j in idx:
                image, sino = ko_train[int(j.item())]
                loss = loss + loss_fn(ko(sino), image)
            loss = loss / batch_size
            ko_opt.zero_grad(); loss.backward(); ko_opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        print(
            f"[progression] trained KO ({time.perf_counter()-t0:.1f}s, "
            f"last_loss={float(loss.detach().cpu()):.4e})",
            flush=True,
        )
        if ko_ckpt:
            ko_ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ko.state_dict(), ko_ckpt)
            print(f"[progression] cached KO weights to {ko_ckpt}", flush=True)
        del ko_train

    # KO recons (computed once), then free the model
    ko.eval()
    ko_recons = []
    with torch.no_grad():
        for _, sino in test_set:
            ko_recons.append(ko(sino).detach().cpu().numpy())
    del ko
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ------- FC progression -------
    fc = FullyConnectedReconstructor.from_geometry(geometry).to(device)
    fc_opt = torch.optim.Adagrad(fc.parameters(), lr=lr)
    data_rng = torch.Generator(device="cpu").manual_seed(base_seed + 1001)
    loss_fn = nn.MSELoss()

    fc_recons_at: dict[int, list] = {}
    loss_log: list[dict] = []

    # Log loss every 1% of the longest run, capped at every iter for very short.
    log_step = max(1, max(checkpoints) // 200)

    cur_iter = 0
    last_loss = float("nan")
    t_start = time.perf_counter()
    for cp in checkpoints:
        # Train from cur_iter to cp
        for it in range(cur_iter, cp):
            loss = torch.tensor(0.0, device=device)
            for _b in range(batch_size):
                image = random_phantom(geometry, data_rng, ellipses[0], ellipses[1]).to(device)
                sino = fan_beam_forward(image, geometry)
                loss = loss + loss_fn(fc(sino), image)
            loss = loss / batch_size
            fc_opt.zero_grad(); loss.backward(); fc_opt.step()
            last_loss = float(loss.detach().cpu())
            if (it + 1) % log_step == 0 or it == cp - 1:
                loss_log.append({"iter": it + 1, "loss": last_loss})
        cur_iter = cp
        # Eval at this checkpoint
        fc.eval()
        with torch.no_grad():
            fc_recons_at[cp] = [fc(sino).detach().cpu().numpy() for _, sino in test_set]
        fc.train()
        elapsed = time.perf_counter() - t_start
        print(
            f"[progression] @ iter={cp:>6d}  last_loss={last_loss:.4e}  "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    # ------- Side-by-side figure -------
    n_rows = len(test_set)
    n_cols = 2 + len(checkpoints)  # phantom, KO, then one column per FC checkpoint
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.7 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_rows):
        phantom = test_set[i][0].detach().cpu().numpy()
        vmax = float(max(phantom.max(), 1e-9))
        ko = ko_recons[i]
        ko_rrmse = float(
            ((ko - phantom) ** 2).mean() ** 0.5
            / (abs(phantom).max() + 1e-9)
        )
        cells = [
            (phantom, "phantom"),
            (ko, f"KO (rRMSE={ko_rrmse:.3f})"),
        ]
        for cp in checkpoints:
            fc_pred = fc_recons_at[cp][i]
            rr = float(
                ((fc_pred - phantom) ** 2).mean() ** 0.5
                / (abs(phantom).max() + 1e-9)
            )
            cells.append((fc_pred, f"FC @ {cp:,} iter\nrRMSE={rr:.3f}"))
        for ax, (img, title) in zip(axes[i], cells):
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=vmax)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"FC training progression @ {geometry.image_size}x{geometry.image_size} "
        f"(online: fresh phantoms each iter)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # ------- Loss curve -------
    loss_path = out_path.parent / (out_path.stem + "_loss.png")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    its = np.array([r["iter"] for r in loss_log], dtype=np.float64)
    ls = np.array([r["loss"] for r in loss_log], dtype=np.float64)
    # Raw per-batch loss is heavily fluctuating (single batch of 4 fresh
    # phantoms each step), so we overlay a geometric moving average that
    # exposes the smoothed trend.
    ax.plot(its, ls, color="#b54b32", linewidth=0.6, alpha=0.35,
            label="FC training loss (raw)")
    if len(ls) >= 5:
        # 5%-of-points window or 21, whichever is bigger; force an odd window.
        win = max(21, (len(ls) // 20) | 1)
        win = min(win, len(ls) - (1 - len(ls) % 2))
        kernel = np.ones(win) / win
        smoothed = np.convolve(ls, kernel, mode="same")
        # Fix edge bias: at the ends, the convolution averages over less data.
        norm = np.convolve(np.ones_like(ls), kernel, mode="same")
        smoothed /= norm
        ax.plot(its, smoothed, color="#b54b32", linewidth=1.8,
                label=f"FC training loss (moving avg, w={win})")
    # Reference horizontal: E[phantom²], the predict-zero floor.
    sq = np.concatenate([
        (test_set[i][0].detach().cpu().numpy() ** 2).ravel()
        for i in range(len(test_set))
    ])
    ey2 = float(sq.mean())
    ax.axhline(ey2, color="gray", linestyle="--", linewidth=0.9,
               label=f"$E[y^2]$ on test slices = {ey2:.3f}")
    for cp in checkpoints:
        ax.axvline(cp, color="black", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training iteration (online: 4 fresh phantoms / iter)")
    ax.set_ylabel("FC training MSE")
    ax.set_title(
        f"FC loss decay vs. iteration @ "
        f"{geometry.image_size}x{geometry.image_size}, {geometry.num_views} views"
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)

    # ------- JSON dump for downstream plotting -------
    json_path = out_path.parent / (out_path.stem + "_log.json")
    json_path.write_text(json.dumps({
        "geometry": cfg["geometry"],
        "checkpoints": checkpoints,
        "batch_size": batch_size,
        "learning_rate": lr,
        "online": True,
        "loss_log": loss_log,
        "ey2_on_test": ey2,
    }, indent=2))
    print(f"[progression] wrote {out_path}, {loss_path}, {json_path}", flush=True)


if __name__ == "__main__":
    main()
