"""GPU training entry point for the operator-aware CT network."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from torch import nn

from ct_dataset import FanBeamGeometry, iter_slice_dataset
from ct_models import FullyConnectedReconstructor, KnownOperatorReconstructor, parameter_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["known_operator", "fully_connected"],
                        default="known_operator")
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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["reporting"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    geometry = make_geometry(cfg)

    if args.model == "known_operator":
        model = KnownOperatorReconstructor(geometry).to(device)
    elif args.model == "fully_connected":
        model = FullyConnectedReconstructor.from_geometry(geometry).to(device)
    else:
        raise SystemExit(f"Unknown model: {args.model}")

    optimizer = torch.optim.Adagrad(model.parameters(), lr=cfg["training"]["learning_rate"])
    loss_fn = nn.MSELoss()

    metrics: dict = {
        "model": args.model,
        "device": str(device),
        "config_path": args.config,
        "geometry": cfg["geometry"],
        "training": cfg["training"],
        "iterations": [],
    }
    metrics["parameter_counts"] = dict(zip(["known_operator", "fully_connected"], parameter_counts(geometry)))

    seed = cfg["dataset"]["seed"]
    num_train = cfg["dataset"]["num_train_slices"]
    num_iter = cfg["training"]["num_iterations"]
    batch_size = cfg["training"]["batch_size"]
    log_every = cfg["training"]["log_every"]

    dataset_iter = iter_slice_dataset(
        geometry,
        num_train,
        seed=seed,
        ellipses_per_slice=tuple(cfg["dataset"]["ellipses_per_slice"]),
        device=device,
    )

    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    for image, sino in dataset_iter:
        cached.append((image.detach(), sino.detach()))

    if not cached:
        raise SystemExit("Empty training set")

    rng = torch.Generator(device="cpu").manual_seed(seed + 1)
    start = time.perf_counter()
    peak_mem_bytes = 0

    for it in range(num_iter):
        idx = torch.randint(0, len(cached), (batch_size,), generator=rng)
        loss = torch.tensor(0.0, device=device)
        for sample_idx in idx:
            image, sino = cached[int(sample_idx.item())]
            recon = model(sino)
            loss = loss + loss_fn(recon, image)
        loss = loss / batch_size
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            peak_mem_bytes = max(peak_mem_bytes, torch.cuda.max_memory_allocated(device))
        if it % log_every == 0:
            metrics["iterations"].append({"iter": it, "loss": float(loss.detach().cpu())})

    elapsed = time.perf_counter() - start
    metrics["wall_time_seconds"] = elapsed
    metrics["peak_gpu_memory_bytes"] = int(peak_mem_bytes)
    metrics["num_training_slices"] = num_train

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"{args.model}.pt")

    metrics_path = out_dir / f"ct_{args.model}_metrics.json"
    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)


if __name__ == "__main__":
    main()
