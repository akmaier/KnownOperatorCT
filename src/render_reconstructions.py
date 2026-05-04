"""Render side-by-side reconstruction figures for the paper.

For a given experiment config, trains one Known Operator and one Fully
Connected reconstructor from scratch on N training slices, then plots
``--num-samples`` test slices as

    phantom | KO recon | KO error | FC recon | FC error

with consistent grayscale across phantom/recon and a symmetric red-blue
diverging colormap on the error panels. Saves a single PNG.

Usage:
  python src/render_reconstructions.py \
      --config configs/ct_sample_efficiency_128.yaml \
      --train-size 2048 --num-iterations 5000 --num-samples 2 \
      --out results/sample_efficiency_128/reconstructions.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch import nn

from ct_dataset import FanBeamGeometry, iter_slice_dataset
from ct_models import (
    FullyConnectedReconstructor,
    KnownOperatorReconstructor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-size", type=int, required=True,
                        help="Number of training slices (sampled from the configured pool).")
    parser.add_argument("--num-iterations", type=int, required=True)
    parser.add_argument("--num-samples", type=int, default=2,
                        help="How many test slices to render in the figure.")
    parser.add_argument("--out", required=True, help="Output PNG path.")
    return parser.parse_args()


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


def materialize(geometry, n, seed, ellipses, device):
    return [(image.detach(), sino.detach())
            for image, sino in iter_slice_dataset(
                geometry, n, seed=seed, ellipses_per_slice=ellipses, device=device)]


def train(model: nn.Module, train_set, num_iter: int, batch_size: int,
          learning_rate: float, device, rng_seed: int) -> float:
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    rng = torch.Generator(device="cpu").manual_seed(rng_seed)
    n = len(train_set)
    eff_batch = min(batch_size, n)
    last = float("nan")
    t0 = time.perf_counter()
    for _ in range(num_iter):
        idx = torch.randint(0, n, (eff_batch,), generator=rng)
        loss = torch.tensor(0.0, device=device)
        for j in idx:
            image, sino = train_set[int(j.item())]
            loss = loss + loss_fn(model(sino), image)
        loss = loss / eff_batch
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last = float(loss.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - t0, last


def render(test_set, ko_recons, fc_recons, out_path: Path,
           image_size: int, num_views: int) -> None:
    n = len(test_set)
    fig, axes = plt.subplots(n, 5, figsize=(13, 2.7 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        phantom = test_set[i][0].detach().cpu().numpy()
        ko = ko_recons[i]
        fc = fc_recons[i]
        ko_err = ko - phantom
        fc_err = fc - phantom

        vmax = float(max(phantom.max(), ko.max(), fc.max()))
        emax = float(max(abs(ko_err).max(), abs(fc_err).max(), 1e-9))

        ims = [
            (phantom, "phantom", "gray", 0.0, vmax),
            (ko,      f"KO recon (rRMSE={(((ko-phantom)**2).mean()**0.5/(abs(phantom).max()+1e-9)):.3f})", "gray", 0.0, vmax),
            (ko_err,  "KO error",  "RdBu_r", -emax, emax),
            (fc,      f"FC recon (rRMSE={(((fc-phantom)**2).mean()**0.5/(abs(phantom).max()+1e-9)):.3f})", "gray", 0.0, vmax),
            (fc_err,  "FC error",  "RdBu_r", -emax, emax),
        ]
        for ax, (img, title, cmap, vmin, vmaxi) in zip(axes[i], ims):
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmaxi)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        # colourbar on the right of each row's last error panel for scale
        plt.colorbar(im, ax=axes[i, -1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Sample reconstructions @ {image_size}x{image_size}, "
        f"{num_views} views",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    geometry = make_geometry(cfg)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"[render] device={device} geometry={geometry}", flush=True)

    base_seed = int(cfg["dataset"].get("seed", 1))
    ellipses = tuple(cfg["dataset"]["ellipses_per_slice"])

    test_set = materialize(geometry, args.num_samples,
                            seed=base_seed + 10_000,
                            ellipses=ellipses, device=device)
    train_set = materialize(geometry, args.train_size,
                             seed=base_seed + 1,
                             ellipses=ellipses, device=device)
    print(f"[render] train_size={len(train_set)}, num_test={len(test_set)}", flush=True)

    batch_size = int(cfg["training"]["batch_size"])
    lr = float(cfg["training"]["learning_rate"])

    ko = KnownOperatorReconstructor(geometry).to(device)
    t_ko, last_ko = train(ko, train_set, args.num_iterations, batch_size, lr,
                           device, rng_seed=base_seed + 1000)
    print(f"[render] trained KO ({t_ko:.1f}s, last_loss={last_ko:.4e})", flush=True)

    fc = FullyConnectedReconstructor.from_geometry(geometry).to(device)
    t_fc, last_fc = train(fc, train_set, args.num_iterations, batch_size, lr,
                           device, rng_seed=base_seed + 1001)
    print(f"[render] trained FC ({t_fc:.1f}s, last_loss={last_fc:.4e})", flush=True)

    ko.eval(); fc.eval()
    ko_recons = []
    fc_recons = []
    with torch.no_grad():
        for _, sino in test_set:
            ko_recons.append(ko(sino).detach().cpu().numpy())
            fc_recons.append(fc(sino).detach().cpu().numpy())

    render(test_set, ko_recons, fc_recons, out_path,
           image_size=geometry.image_size, num_views=geometry.num_views)
    print(f"[render] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
