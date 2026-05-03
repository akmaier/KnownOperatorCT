"""Distributed GPU training with FSDP for the fully connected CT model.

The FC model at full resolution (512x512, 180 views) has ~24 billion
parameters (~90 GB FP32 weights, ~360 GB Adam state).  This exceeds any
single GPU.  We use PyTorch FSDP (Fully Sharded Data Parallel) to shard
model parameters, gradients, and optimizer state across all available GPUs,
with CPU offloading for the optimizer state to fit within GPU memory.

Designed for Helma at RRZE (4x H100 94 GB per node).

Usage (via torchrun):
    torchrun --nproc_per_node=4 src/ct_train_distributed.py \
        --config configs/ct_full_resolution.yaml --model fully_connected
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

from ct_dataset import FanBeamGeometry, iter_slice_dataset
from ct_models import (
    FullyConnectedReconstructor,
    KnownOperatorReconstructor,
    parameter_counts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model",
        choices=["known_operator", "fully_connected"],
        default="fully_connected",
    )
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


def log_rank0(msg: str) -> None:
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(msg, flush=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["reporting"]["out_dir"])

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    geometry = make_geometry(cfg)

    log_rank0(f"World size: {world_size}, model: {args.model}")
    log_rank0(f"GPU: {torch.cuda.get_device_name(device)}")

    if args.model == "known_operator":
        base_model = KnownOperatorReconstructor(geometry)
    elif args.model == "fully_connected":
        base_model = FullyConnectedReconstructor.from_geometry(geometry)
    else:
        raise SystemExit(f"Unknown model: {args.model}")

    p_ko, p_fc = parameter_counts(geometry)
    n_params = sum(p.numel() for p in base_model.parameters())
    mem_gb = n_params * 4 / 1024**3
    log_rank0(f"Model params: {n_params:,} ({mem_gb:.2f} GB FP32)")
    log_rank0(f"Per-GPU shard (params only): ~{mem_gb / world_size:.2f} GB")

    use_cpu_offload = mem_gb > 40.0
    if use_cpu_offload:
        log_rank0("Enabling FSDP CPU offload for optimizer state")

    model = FSDP(
        base_model.to(device),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=False) if use_cpu_offload else None,
        device_id=device,
    )

    optimizer = torch.optim.Adagrad(
        model.parameters(), lr=cfg["training"]["learning_rate"]
    )
    loss_fn = nn.MSELoss()

    seed = cfg["dataset"]["seed"]
    num_train = cfg["dataset"]["num_train_slices"]
    num_iter = cfg["training"]["num_iterations"]
    batch_size = cfg["training"]["batch_size"]
    log_every = cfg["training"]["log_every"]

    log_rank0(f"Generating {num_train} training slices...")
    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    for image, sino in iter_slice_dataset(
        geometry,
        num_train,
        seed=seed,
        ellipses_per_slice=tuple(cfg["dataset"]["ellipses_per_slice"]),
        device=device,
    ):
        cached.append((image.detach(), sino.detach()))

    if not cached:
        raise SystemExit("Empty training set")
    log_rank0(f"Dataset ready: {len(cached)} slices on {device}")

    rng = torch.Generator(device="cpu").manual_seed(seed + 1 + local_rank)
    iterations_log: list[dict] = []
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

        peak_mem_bytes = max(peak_mem_bytes, torch.cuda.max_memory_allocated(device))

        if it % log_every == 0:
            dist.all_reduce(loss.detach(), op=dist.ReduceOp.AVG)
            loss_val = float(loss.detach().cpu())
            iterations_log.append({"iter": it, "loss": loss_val})
            log_rank0(f"  iter {it:>6d}/{num_iter}  loss={loss_val:.6f}")

    elapsed = time.perf_counter() - start

    if local_rank == 0:
        metrics = {
            "model": args.model,
            "device": str(device),
            "world_size": world_size,
            "fsdp_cpu_offload": use_cpu_offload,
            "config_path": args.config,
            "geometry": cfg["geometry"],
            "training": cfg["training"],
            "iterations": iterations_log,
            "parameter_counts": {
                "known_operator": p_ko,
                "fully_connected": p_fc,
            },
            "wall_time_seconds": elapsed,
            "peak_gpu_memory_bytes": int(peak_mem_bytes),
            "num_training_slices": num_train,
        }

        metrics_path = out_dir / f"ct_{args.model}_metrics.json"
        with open(metrics_path, "w") as handle:
            json.dump(metrics, handle, indent=2)
        log_rank0(f"Metrics written to {metrics_path}")

        ckpt_dir = out_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state = model.state_dict()
        torch.save(state, ckpt_dir / f"{args.model}.pt")
        log_rank0(f"Checkpoint saved to {ckpt_dir / args.model}.pt")

    dist.barrier()
    dist.destroy_process_group()
    log_rank0("Done.")


if __name__ == "__main__":
    main()
