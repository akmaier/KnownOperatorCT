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
    # FC ablation knobs — try alternative recipes to escape the dead-ReLU
    # plateau the default config falls into. Defaults match the config.
    parser.add_argument("--fc-optimizer", choices=["adagrad", "adam"], default="adagrad")
    parser.add_argument("--fc-lr", type=float, default=None,
                        help="Override learning rate for the FC model only.")
    parser.add_argument("--fc-bias", action="store_true",
                        help="Replace FC's bias-free Linear with a Linear that has bias.")
    parser.add_argument("--fc-init", choices=["kaiming", "xavier"], default="kaiming",
                        help="Re-initialize FC's weight matrix with this scheme.")
    parser.add_argument("--fc-tag", default="",
                        help="Annotation appended to the figure title.")
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
          learning_rate: float, device, rng_seed: int,
          optimizer_kind: str = "adagrad"):
    if optimizer_kind == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
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
           image_size: int, num_views: int, subtitle: str = "") -> None:
    """6-column figure:
        phantom | KO recon | KO error | FC recon (shared scale) | FC recon (auto) | FC error

    The first FC column shares vmax with the phantom so the magnitude
    collapse (FC output ≪ phantom intensity) is visible. The second FC
    column uses FC's own 99-th percentile as vmax, so whatever faint
    structure FC actually produced becomes legible. Errors are clipped to
    the joint 99-th percentile to keep a few outlier pixels from blowing
    up the colormap.
    """
    import numpy as np
    n = len(test_set)
    fig, axes = plt.subplots(n, 6, figsize=(15.5, 2.7 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    for i in range(n):
        phantom = test_set[i][0].detach().cpu().numpy()
        ko = ko_recons[i]
        fc = fc_recons[i]
        ko_err = ko - phantom
        fc_err = fc - phantom

        vmax = float(max(phantom.max(), ko.max(), fc.max()))
        # Clip the error colormap at the joint 99.5th percentile so a few
        # outlier pixels in FC don't squash the visible range to white.
        joint_abs = np.concatenate([np.abs(ko_err).ravel(), np.abs(fc_err).ravel()])
        emax = float(max(np.percentile(joint_abs, 99.5), 1e-9))
        # FC own dynamic range, also percentile-clipped for the auto panel.
        fc_clip_max = float(max(np.percentile(fc, 99.5), 1e-9))

        ko_rrmse = float(((ko - phantom) ** 2).mean() ** 0.5
                         / (abs(phantom).max() + 1e-9))
        fc_rrmse = float(((fc - phantom) ** 2).mean() ** 0.5
                         / (abs(phantom).max() + 1e-9))

        ims = [
            (phantom, "phantom", "gray", 0.0, vmax),
            (ko,      f"KO recon (rRMSE={ko_rrmse:.3f})", "gray", 0.0, vmax),
            (ko_err,  "KO error",  "RdBu_r", -emax, emax),
            (fc,      f"FC recon (rRMSE={fc_rrmse:.3f}, max={float(fc.max()):.3f})",
                      "gray", 0.0, vmax),
            (fc,      f"FC recon auto (vmax={fc_clip_max:.3f})",
                      "gray", 0.0, fc_clip_max),
            (fc_err,  "FC error",  "RdBu_r", -emax, emax),
        ]
        for ax, (img, title, cmap, vmin, vmaxi) in zip(axes[i], ims):
            im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmaxi)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        # colourbar on the right of each row's last error panel for scale
        plt.colorbar(im, ax=axes[i, -1], fraction=0.046, pad=0.04)

    title = f"Sample reconstructions @ {image_size}x{image_size}, {num_views} views"
    if subtitle:
        title = f"{title} — {subtitle}"
    fig.suptitle(title, fontsize=12)
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
                           device, rng_seed=base_seed + 1000,
                           optimizer_kind="adagrad")
    print(f"[render] trained KO ({t_ko:.1f}s, last_loss={last_ko:.4e})", flush=True)

    # Compute KO recons up front and drop the model + its allocator cache
    # before bringing FC up. The 256x256 FC needs ~17 GB peak with Adagrad
    # (weights + state + gradient buffer); on a 24 GB GPU the residual
    # caching allocator from KO is enough to OOM the FC's first .step().
    ko.eval()
    ko_recons: list = []
    with torch.no_grad():
        for _, sino in test_set:
            ko_recons.append(ko(sino).detach().cpu().numpy())
    del ko
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # FC: optionally swap in a Linear with bias and re-init.
    fc = FullyConnectedReconstructor.from_geometry(geometry).to(device)
    if args.fc_bias:
        n_in = fc.linear.in_features
        n_out = fc.linear.out_features
        fc.linear = nn.Linear(n_in, n_out, bias=True).to(device)
    if args.fc_init == "xavier":
        nn.init.xavier_uniform_(fc.linear.weight)
    else:
        # kaiming-uniform with a=sqrt(5) matches the default in nn.Linear
        import math
        nn.init.kaiming_uniform_(fc.linear.weight, a=math.sqrt(5))
    if fc.linear.bias is not None:
        nn.init.zeros_(fc.linear.bias)

    fc_lr = args.fc_lr if args.fc_lr is not None else lr
    t_fc, last_fc = train(fc, train_set, args.num_iterations, batch_size, fc_lr,
                           device, rng_seed=base_seed + 1001,
                           optimizer_kind=args.fc_optimizer)
    print(
        f"[render] trained FC ({t_fc:.1f}s, last_loss={last_fc:.4e}, "
        f"opt={args.fc_optimizer}, lr={fc_lr}, "
        f"bias={args.fc_bias}, init={args.fc_init})",
        flush=True,
    )

    fc.eval()
    fc_recons = []
    with torch.no_grad():
        for _, sino in test_set:
            fc_recons.append(fc(sino).detach().cpu().numpy())

    render(test_set, ko_recons, fc_recons, out_path,
           image_size=geometry.image_size, num_views=geometry.num_views,
           subtitle=args.fc_tag)
    print(f"[render] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
