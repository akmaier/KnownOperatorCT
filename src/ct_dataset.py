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

    @property
    def fov_radius_mm(self) -> float:
        """Half-width of the reconstructed FOV in mm."""
        return (0.5 * self.detector_bins * self.detector_pixel_mm
                * self.source_to_iso_mm / self.source_to_detector_mm)


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

    Implements divergent-ray fan-beam geometry.  For each source position the
    fan of rays is traced through the image and accumulated into detector
    bins via bilinear sampling, matching the PYRO-NN reference.

    Returns a sinogram of shape ``[num_views, detector_bins]``.
    """
    angles = geometry.angles_rad.to(image.device)
    side = geometry.image_size
    bins = geometry.detector_bins
    D_si = geometry.source_to_iso_mm
    D_sd = geometry.source_to_detector_mm
    det_pixel = geometry.detector_pixel_mm
    fov_r = geometry.fov_radius_mm

    sino = torch.zeros(geometry.num_views, bins, dtype=image.dtype, device=image.device)

    det_center = 0.5 * (bins - 1)
    det_offsets = (torch.arange(bins, device=image.device, dtype=image.dtype) - det_center) * det_pixel
    num_steps = int(2.0 * side)
    t_vals = torch.linspace(0.0, 1.0, num_steps, device=image.device)

    img4d = image.unsqueeze(0).unsqueeze(0)

    for view_idx, angle in enumerate(angles):
        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        src_x = -D_si * sin_a
        src_y = D_si * cos_a

        det_x = (D_sd - D_si) * sin_a + det_offsets * cos_a
        det_y = -(D_sd - D_si) * cos_a + det_offsets * sin_a

        dx = det_x - src_x
        dy = det_y - src_y
        ray_len = torch.sqrt(dx * dx + dy * dy)

        ray_x = src_x + t_vals[:, None] * dx[None, :]
        ray_y = src_y + t_vals[:, None] * dy[None, :]

        grid_x = ray_x / fov_r
        grid_y = ray_y / fov_r
        grid_pts = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, num_steps * bins, 2)

        samples = torch.nn.functional.grid_sample(
            img4d, grid_pts,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        ).view(num_steps, bins)

        sino[view_idx] = samples.sum(dim=0) * (ray_len / num_steps)

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
