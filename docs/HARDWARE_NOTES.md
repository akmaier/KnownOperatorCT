# Hardware Notes

These are the resource requirements and the reasons behind each.

## Operator-aware CT network at full resolution

- Image size: $512 \times 512$
- Views: $180$
- Detector bins: $512$
- Trainable parameters: $9.22 \cdot 10^4$ (one diagonal weight per detector bin and angle)
- FP32 weight memory: $\sim 0.35$ MB
- Forward and backward operators (analytic): negligible memory beyond the data tensors

A single $512 \times 512$ slice with $180$ projections occupies under $1$ MB in FP32. Mini-batch sizes up to $32$ slices fit comfortably on a $16$ GB GPU. The recommended target hardware is one NVIDIA A100 40 GB or H100 to leave headroom for cuDNN workspaces and validation tensors.

## Fully connected counterfactual at full resolution

- Trainable parameters: $2.42 \cdot 10^{10}$
- FP32 weight memory: $\sim 90$ GB
- Adam state memory (FP32 weights, FP32 moments): $\sim 360$ GB

This exceeds the memory of a single A100 80 GB or H100 80 GB. The bundle therefore does not run this configuration; the harvest script reports the deep risk estimator's prediction for it.

## Recommended cloud SKUs

| Provider | Instance | GPU | Notes |
|---|---|---|---|
| AWS | `p4d.24xlarge` | $8 \times$ A100 40 GB | Single-GPU run uses one of the eight |
| GCP | `a2-highgpu-1g` | $1 \times$ A100 40 GB | Smallest sufficient option |
| Azure | `Standard_ND96asr_A100_v4` | $8 \times$ A100 40 GB | Single-GPU run uses one of the eight |

For sanity checks the bundle can also run on a single $16$ GB GPU (e.g. T4 or A10) at reduced batch size; in that case set `batch_size: 4` in the config.

## CPU surrogate

The surrogate experiment in `src/run_surrogate.py` runs on a single CPU in under five minutes. It does not require a GPU and is useful as a smoke test of the bundle.
