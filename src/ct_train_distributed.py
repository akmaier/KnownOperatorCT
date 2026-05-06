"""Distributed GPU training with FSDP for the fully connected CT model.

The FC model at full resolution (512x512, 180 views) has ~24 billion
parameters (~90 GB FP32 weights, ~360 GB Adam state).  This exceeds any
single GPU.  We use PyTorch FSDP (Fully Sharded Data Parallel) to shard
model parameters, gradients, and optimizer state across all available GPUs,
with CPU offloading for the optimizer state to fit within GPU memory.

Designed for an HPC node with 4x H100 94 GB GPUs. Also runs on smaller
4-GPU nodes with V100 / 1080 Ti / RTX 5000 / RTX 6000 cards (with CPU
offload enabled for the parameter shards).

Usage (via torchrun):
    torchrun --nproc_per_node=4 src/ct_train_distributed.py \
        --config configs/ct_full_resolution.yaml --model fully_connected

Resume support: same convention as ``ct_train.py`` — a periodic snapshot
at ``out_dir/checkpoints/<model>.resume.pt`` lets a later job pick up where
the previous one left off when Slurm hits its wall-time limit.  Only model
weights are restored across resumes (not the FSDP-sharded optimizer state),
so Adagrad's accumulator restarts from zero on each resume. This is
acceptable when a single chunk fits comfortably within one job slot.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
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


_SHOULD_STOP = False


def _stop_handler(signum, frame):  # noqa: ARG001
    global _SHOULD_STOP
    _SHOULD_STOP = True
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[ct_train_distributed] received signal {signum}, will checkpoint and exit", flush=True)


def _install_stop_handlers() -> None:
    signal.signal(signal.SIGTERM, _stop_handler)
    signal.signal(signal.SIGUSR1, _stop_handler)
    signal.signal(signal.SIGINT, _stop_handler)


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
    _install_stop_handlers()

    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["reporting"]["out_dir"])
    ckpt_dir = out_dir / "checkpoints"
    resume_path = ckpt_dir / f"{args.model}.resume.pt"
    final_path = ckpt_dir / f"{args.model}.pt"
    done_path = ckpt_dir / f"{args.model}.done"

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if local_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(exist_ok=True)
    dist.barrier()

    if done_path.exists():
        log_rank0(f"[ct_train_distributed] {done_path} exists; nothing to do.")
        dist.destroy_process_group()
        return

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

    cumulative_elapsed = 0.0
    start_iter = 0
    iterations_log: list[dict] = []

    if resume_path.exists():
        snap = torch.load(resume_path, map_location="cpu")
        base_model.load_state_dict(snap["model"])
        start_iter = int(snap["next_iter"])
        cumulative_elapsed = float(snap.get("elapsed_seconds", 0.0))
        iterations_log = list(snap.get("iterations", []))
        log_rank0(
            f"Resumed weights from {resume_path} at iter {start_iter} "
            f"(elapsed so far: {cumulative_elapsed:.1f}s)"
        )
    dist.barrier()

    # Decide whether to offload parameter shards to CPU. With FSDP FULL_SHARD,
    # the weights are still all-gathered to every rank during forward and
    # backward, so each rank must hold one full copy of the unsharded weights
    # plus the gradients. On a 4-GPU FSDP run with 16 GB GPUs that means
    # ~5.6 GB weights + ~5.6 GB grads + activations + cache > 16 GB and OOMs
    # (observed in jobs 760108 and 760111 (anonymized) on a 4x 16GB node). CPU offload trades
    # bandwidth for memory: parameter shards live on host, get H2D-copied for
    # each forward/backward, freeing GPU room for the all-gather buffer.
    #
    # Trigger offload either when the model is bigger than the largest available GPU
    # (>= 40 GB ≈ A6000), or when the per-rank shard plus an unsharded weight
    # buffer wouldn't comfortably fit a 16 GB GPU. Override via FSDP_CPU_OFFLOAD=1.
    per_gpu_gib = (
        torch.cuda.get_device_properties(device).total_memory / 1024**3
    )
    auto_offload = mem_gb > 40.0 or (mem_gb * 2 + 1.0) > 0.6 * per_gpu_gib
    use_cpu_offload = (
        os.environ.get("FSDP_CPU_OFFLOAD", "").lower() in ("1", "true", "yes")
        or auto_offload
    )
    if use_cpu_offload:
        log_rank0(
            f"Enabling FSDP CPU offload for parameter shards "
            f"(mem_gb={mem_gb:.2f}, per_gpu_gib={per_gpu_gib:.1f})"
        )

    model = FSDP(
        base_model.to(device),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        cpu_offload=CPUOffload(offload_params=True) if use_cpu_offload else None,
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
    checkpoint_every = int(cfg["training"].get("checkpoint_every", 500))

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

    # Each rank advances its own RNG by start_iter steps so that the post-resume
    # batch sequence matches what would have been drawn without the interruption.
    rng = torch.Generator(device="cpu").manual_seed(seed + 1 + local_rank)
    for _ in range(start_iter):
        torch.randint(0, len(cached), (batch_size,), generator=rng)

    def save_resume(next_iter: int, run_elapsed: float) -> None:
        # Collective: gather full (unsharded) weights to rank 0 and save.
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state = model.state_dict()
        if local_rank == 0:
            snap = {
                "model": state,
                "next_iter": next_iter,
                "elapsed_seconds": cumulative_elapsed + run_elapsed,
                "iterations": iterations_log,
            }
            tmp = resume_path.with_suffix(resume_path.suffix + ".tmp")
            torch.save(snap, tmp)
            tmp.replace(resume_path)
        dist.barrier()

    start = time.perf_counter()
    peak_mem_bytes = 0
    last_iter = start_iter

    for it in range(start_iter, num_iter):
        # Sync the stop flag across ranks: rank 0 propagates its decision so
        # everyone exits the loop together (any collective inside save_resume
        # would otherwise deadlock).
        stop_flag = torch.tensor(int(_SHOULD_STOP), device=device)
        dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
        if stop_flag.item() > 0:
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

        peak_mem_bytes = max(peak_mem_bytes, torch.cuda.max_memory_allocated(device))

        if it % log_every == 0:
            dist.all_reduce(loss.detach(), op=dist.ReduceOp.AVG)
            loss_val = float(loss.detach().cpu())
            iterations_log.append({"iter": it, "loss": loss_val})
            log_rank0(f"  iter {it:>6d}/{num_iter}  loss={loss_val:.6f}")

        last_iter = it
        if checkpoint_every > 0 and (it + 1) % checkpoint_every == 0 and (it + 1) < num_iter:
            save_resume(it + 1, time.perf_counter() - start)

    run_elapsed = time.perf_counter() - start
    total_elapsed = cumulative_elapsed + run_elapsed
    completed = (not _SHOULD_STOP) and last_iter >= num_iter - 1

    if completed:
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
            state = model.state_dict()
        if local_rank == 0:
            torch.save(state, final_path)
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
                "wall_time_seconds": total_elapsed,
                "peak_gpu_memory_bytes": int(peak_mem_bytes),
                "num_training_slices": num_train,
            }
            metrics_path = out_dir / f"ct_{args.model}_metrics.json"
            with open(metrics_path, "w") as handle:
                json.dump(metrics, handle, indent=2)
            if resume_path.exists():
                resume_path.unlink()
            done_path.write_text(
                f"completed {num_iter} iterations in {total_elapsed:.1f}s\n"
            )
            log_rank0(f"Final checkpoint -> {final_path}")
    else:
        save_resume(last_iter + 1, run_elapsed)
        log_rank0(
            f"Interrupted at iter {last_iter + 1}/{num_iter}; "
            f"resume snapshot -> {resume_path}"
        )

    dist.barrier()
    dist.destroy_process_group()
    log_rank0("Done.")


if __name__ == "__main__":
    main()
