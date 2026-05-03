"""Synthetic slice-wise CT phantom population.

This module produces randomized ellipsoid phantoms and their fan-beam sinograms
at full resolution. The forward and backward operators are implemented in
PyTorch so the training pipeline can run on a single GPU without external CUDA
operator libraries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import math
import torch
from torch import Tensor


@dataclass
class FanBeamGeometry:
    image_size: int
    num_views: int
    detector_bins: int
    angular_range_degrees: float
    source_to_iso_mm: float
    source_to_detector_mm: float
    detector_pixel_mm: float

    @property
    def angles_rad(self) -> Tensor:
        end = self.angular_range_degrees * math.pi / 180.0
        return torch.linspace(0.0, end, self.num_views, dtype=torch.float32)


def random_phantom(
    geometry: FanBeamGeometry,
    rng: torch.Generator,
    ellipses_min: int,
    ellipses_max: int,
) -> Tensor:
    side = geometry.image_size
    coords = torch.linspace(-1.0, 1.0, side)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    image = torch.zeros(side, side, dtype=torch.float32)
    n = int(torch.randint(ellipses_min, ellipses_max + 1, (1,), generator=rng).item())
    for _ in range(n):
        amplitude = (torch.rand((), generator=rng) * 0.8 + 0.2).item()
        cx, cy = (torch.rand(2, generator=rng) - 0.5).tolist()
        a = (torch.rand((), generator=rng) * 0.32 + 0.08).item()
        b = (torch.rand((), generator=rng) * 0.32 + 0.08).item()
        theta = (torch.rand((), generator=rng) * math.pi).item()
        ct, st = math.cos(theta), math.sin(theta)
        xr = ct * (xx - cx) + st * (yy - cy)
        yr = -st * (xx - cx) + ct * (yy - cy)
        mask = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
        image = image + amplitude * mask.float()
    return torch.clamp(image, 0.0, 1.5)


def fan_beam_forward(image: Tensor, geometry: FanBeamGeometry) -> Tensor:
    """Differentiable fan-beam forward projection on a single device.

    The implementation rotates the image by each view angle and accumulates
    line integrals along the detector axis. The result is a sinogram of shape
    ``[num_views, detector_bins]``.
    """
    angles = geometry.angles_rad.to(image.device)
    side = geometry.image_size
    bins = geometry.detector_bins
    sino = torch.zeros(geometry.num_views, bins, dtype=image.dtype, device=image.device)

    coords = torch.linspace(-1.0, 1.0, side, device=image.device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)

    for view_idx, angle in enumerate(angles):
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)
        rotation = torch.stack(
            [
                torch.stack([cos_a, -sin_a]),
                torch.stack([sin_a, cos_a]),
            ]
        )
        rotated = grid @ rotation.T
        sample_grid = rotated.unsqueeze(0)
        rotated_image = torch.nn.functional.grid_sample(
            image.unsqueeze(0).unsqueeze(0),
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze()
        line_integrals = rotated_image.sum(dim=0)
        if line_integrals.numel() != bins:
            indices = torch.linspace(0, line_integrals.numel() - 1, bins, device=image.device)
            line_integrals = torch.nn.functional.interpolate(
                line_integrals[None, None, :],
                size=bins,
                mode="linear",
                align_corners=True,
            ).squeeze()
        sino[view_idx] = line_integrals
    return sino


def iter_slice_dataset(
    geometry: FanBeamGeometry,
    num_slices: int,
    seed: int,
    ellipses_per_slice: tuple[int, int],
    device: torch.device,
) -> Iterator[tuple[Tensor, Tensor]]:
    rng = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(num_slices):
        image = random_phantom(geometry, rng, ellipses_per_slice[0], ellipses_per_slice[1]).to(device)
        sino = fan_beam_forward(image, geometry)
        yield image, sino
