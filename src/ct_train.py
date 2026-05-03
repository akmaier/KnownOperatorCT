"""GPU training entry point for the operator-aware CT network.

Supports interrupt-safe training under Slurm wall-time limits:
  * a resume snapshot (model + optimizer + RNG + accumulated metrics) is
    written every ``training.checkpoint_every`` iterations and on SIGTERM /
    SIGUSR1
  * on startup, an existing snapshot at
    ``out_dir/checkpoints/<model>.resume.pt`` is reloaded and training picks
    up from the next iteration
  * when the full iteration budget completes, the snapshot is replaced with
    the final ``<model>.pt`` and a ``<model>.done`` sentinel is written
"""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import torch
import yaml
from torch import nn

from ct_dataset import FanBeamGeometry, iter_slice_dataset
from ct_models import FullyConnectedReconstructor, KnownOperatorReconstructor, parameter_counts


_SHOULD_STOP = False


def _stop_handler(signum, frame):  # noqa: ARG001
    global _SHOULD_STOP
    _SHOULD_STOP = True
    print(f"[ct_train] received signal {signum}, will checkpoint and exit", flush=True)


def _install_stop_handlers() -> None:
    signal.signal(signal.SIGTERM, _stop_handler)
    signal.signal(signal.SIGUSR1, _stop_handler)
    signal.signal(signal.SIGINT, _stop_handler)


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
    _install_stop_handlers()

    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["reporting"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    resume_path = ckpt_dir / f"{args.model}.resume.pt"
    final_path = ckpt_dir / f"{args.model}.pt"
    done_path = ckpt_dir / f"{args.model}.done"

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
    checkpoint_every = int(cfg["training"].get("checkpoint_every", 500))

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

    start_iter = 0
    cumulative_elapsed = 0.0
    peak_mem_bytes = 0

    if done_path.exists():
        print(f"[ct_train] {done_path} exists; nothing to do.", flush=True)
        return

    if resume_path.exists():
        snap = torch.load(resume_path, map_location=device)
        model.load_state_dict(snap["model"])
        optimizer.load_state_dict(snap["optimizer"])
        rng.set_state(snap["rng"])
        start_iter = int(snap["next_iter"])
        cumulative_elapsed = float(snap.get("elapsed_seconds", 0.0))
        peak_mem_bytes = int(snap.get("peak_gpu_memory_bytes", 0))
        metrics["iterations"] = list(snap.get("iterations", []))
        print(
            f"[ct_train] resumed from {resume_path} at iter {start_iter} "
            f"(elapsed so far: {cumulative_elapsed:.1f}s)",
            flush=True,
        )

    def save_resume(next_iter: int, run_elapsed: float) -> None:
        snap = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": rng.get_state(),
            "next_iter": next_iter,
            "elapsed_seconds": cumulative_elapsed + run_elapsed,
            "peak_gpu_memory_bytes": int(peak_mem_bytes),
            "iterations": metrics["iterations"],
        }
        tmp = resume_path.with_suffix(resume_path.suffix + ".tmp")
        torch.save(snap, tmp)
        tmp.replace(resume_path)

    start = time.perf_counter()
    last_iter = start_iter

    for it in range(start_iter, num_iter):
        if _SHOULD_STOP:
            break
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
        last_iter = it
        if checkpoint_every > 0 and (it + 1) % checkpoint_every == 0 and (it + 1) < num_iter:
            save_resume(it + 1, time.perf_counter() - start)

    run_elapsed = time.perf_counter() - start
    total_elapsed = cumulative_elapsed + run_elapsed
    completed = (not _SHOULD_STOP) and last_iter >= num_iter - 1

    if completed:
        metrics["wall_time_seconds"] = total_elapsed
        metrics["peak_gpu_memory_bytes"] = int(peak_mem_bytes)
        metrics["num_training_slices"] = num_train
        torch.save(model.state_dict(), final_path)
        metrics_path = out_dir / f"ct_{args.model}_metrics.json"
        with open(metrics_path, "w") as handle:
            json.dump(metrics, handle, indent=2)
        if resume_path.exists():
            resume_path.unlink()
        done_path.write_text(f"completed {num_iter} iterations in {total_elapsed:.1f}s\n")
        print(f"[ct_train] complete. final checkpoint -> {final_path}", flush=True)
    else:
        save_resume(last_iter + 1, run_elapsed)
        print(
            f"[ct_train] interrupted at iter {last_iter + 1}/{num_iter}; "
            f"resume snapshot -> {resume_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
