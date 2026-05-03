"""Evaluation script for the operator-aware CT network."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml

from ct_dataset import FanBeamGeometry, iter_slice_dataset
from ct_models import KnownOperatorReconstructor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["known_operator"], default="known_operator")
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


def rrmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    diff_sq = torch.mean((pred - target) ** 2)
    norm = target.abs().max().clamp_min(1e-9)
    return float(torch.sqrt(diff_sq) / norm)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["reporting"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    geometry = make_geometry(cfg)

    model = KnownOperatorReconstructor(geometry).to(device)
    ckpt_path = out_dir / "checkpoints" / f"{args.model}.pt"
    if ckpt_path.exists():
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    seed = cfg["dataset"]["seed"] + 1000
    num_test = cfg["dataset"]["num_test_slices"]

    rrmse_trained = []
    rrmse_analytic = []
    inference_times_ms = []

    start = time.perf_counter()
    for image, sino in iter_slice_dataset(
        geometry,
        num_test,
        seed=seed,
        ellipses_per_slice=tuple(cfg["dataset"]["ellipses_per_slice"]),
        device=device,
    ):
        t0 = time.perf_counter()
        with torch.no_grad():
            recon = model(sino)
        inference_times_ms.append(1000.0 * (time.perf_counter() - t0))
        rrmse_trained.append(rrmse(recon, image))

        # Analytic baseline: same architecture with weights frozen at the
        # initialization, i.e. without any training of the diagonal layer.
        baseline = KnownOperatorReconstructor(geometry).to(device)
        with torch.no_grad():
            recon_baseline = baseline(sino)
        rrmse_analytic.append(rrmse(recon_baseline, image))

    elapsed = time.perf_counter() - start

    summary = {
        "model": args.model,
        "device": str(device),
        "num_test_slices": num_test,
        "wall_time_seconds": elapsed,
        "inference_time_ms": {
            "mean": float(sum(inference_times_ms) / max(1, len(inference_times_ms))),
            "min": float(min(inference_times_ms) if inference_times_ms else math.nan),
            "max": float(max(inference_times_ms) if inference_times_ms else math.nan),
        },
        "rrmse_trained": {
            "mean": float(sum(rrmse_trained) / max(1, len(rrmse_trained))),
            "values": rrmse_trained,
        },
        "rrmse_analytic_baseline": {
            "mean": float(sum(rrmse_analytic) / max(1, len(rrmse_analytic))),
            "values": rrmse_analytic,
        },
    }

    out_path = out_dir / f"ct_{args.model}_eval.json"
    with open(out_path, "w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
