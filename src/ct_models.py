"""Known operator and fully connected reconstruction models.

The known operator network mirrors the architecture of Wuerfl et al. (2018) and
Maier et al. (2019): a trainable diagonal weighting layer in the projection
domain followed by a fixed reconstruction filter and a fixed backprojection.
The fully connected counterfactual replaces the backprojection by a dense
learned linear layer.

Reference GPU implementation: https://doi.org/10.24433/CO.2164960.v1
"""

from __future__ import annotations

from typing import Tuple

import math
import torch
from torch import Tensor, nn

from ct_dataset import FanBeamGeometry, fan_beam_forward


def ramp_filter(num_bins: int) -> Tensor:
    freqs = torch.fft.fftfreq(num_bins).abs()
    filt = freqs.clone().to(torch.float32)
    return filt


def parker_cosine_weights(geometry: FanBeamGeometry) -> Tensor:
    """Compute Parker x cosine weights for short-scan fan-beam FBP.

    Follows the reference implementation (Wuerfl et al. / PYRO-NN):
    Parker weights handle redundancy in the 180-degree scan,
    cosine weights correct for detector-bin obliquity.
    """
    n_views = geometry.num_views
    n_bins = geometry.detector_bins
    D_si = geometry.source_to_iso_mm
    D_sd = geometry.source_to_detector_mm
    det_pixel = geometry.detector_pixel_mm

    det_center = 0.5 * (n_bins - 1)
    det_offsets = (torch.arange(n_bins, dtype=torch.float32) - det_center) * det_pixel
    gamma = torch.atan2(det_offsets, torch.tensor(D_sd, dtype=torch.float32))
    gamma_max = gamma.abs().max().item()

    angular_range = geometry.angular_range_degrees * math.pi / 180.0
    angles = torch.linspace(0.0, angular_range, n_views, dtype=torch.float32)

    epsilon = angular_range - math.pi
    weights = torch.zeros(n_views, n_bins, dtype=torch.float32)

    for v in range(n_views):
        beta = angles[v].item()
        for b in range(n_bins):
            g = gamma[b].item()
            val = 0.0
            if 0 <= beta < 2.0 * (epsilon - g):
                denom = epsilon - g
                if abs(denom) > 1e-12:
                    arg = (math.pi * beta) / (4.0 * denom)
                    val = math.sin(arg) ** 2
                else:
                    val = 1.0
            elif 2.0 * (epsilon - g) <= beta <= math.pi - 2.0 * g:
                val = 1.0
            elif math.pi - 2.0 * g < beta <= math.pi + 2.0 * epsilon:
                denom = epsilon + g
                if abs(denom) > 1e-12:
                    arg = (math.pi * (math.pi + 2.0 * epsilon - beta)) / (4.0 * denom)
                    val = math.sin(arg) ** 2
                else:
                    val = 1.0

            cos_w = D_sd / math.sqrt(D_sd ** 2 + det_offsets[b].item() ** 2)
            weights[v, b] = val * cos_w

    return weights


class KnownOperatorReconstructor(nn.Module):
    """KO net: trainable diagonal W, fixed K, fixed A^T, ReLU."""

    def __init__(self, geometry: FanBeamGeometry) -> None:
        super().__init__()
        self.geometry = geometry
        weights_init = parker_cosine_weights(geometry)
        self.weights = nn.Parameter(weights_init)
        self.register_buffer("ramp", ramp_filter(geometry.detector_bins))

    def forward(self, sinogram: Tensor) -> Tensor:
        weighted = sinogram * self.weights
        filtered_freq = torch.fft.fft(weighted, dim=-1) * self.ramp.to(weighted.device)
        filtered = torch.real(torch.fft.ifft(filtered_freq, dim=-1))
        image = self._fan_beam_backproject(filtered)
        return torch.relu(image)

    def _fan_beam_backproject(self, filtered: Tensor) -> Tensor:
        """Fan-beam backprojection with distance weighting (D_si^2 / U^2).

        Follows the standard fan-beam FBP formulation and the PYRO-NN
        reference: for each view, project each pixel onto the detector and
        weight by (D_si / U)^2 where U is the distance from the source to
        the pixel projected along the central ray direction.
        """
        geom = self.geometry
        side = geom.image_size
        D_si = geom.source_to_iso_mm
        D_sd = geom.source_to_detector_mm
        det_pixel = geom.detector_pixel_mm
        fov_r = geom.fov_radius_mm

        coords = torch.linspace(-fov_r, fov_r, side, device=filtered.device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        angles = geom.angles_rad.to(filtered.device)
        det_center = 0.5 * (geom.detector_bins - 1)
        accum = torch.zeros(side, side, dtype=filtered.dtype, device=filtered.device)
        angular_step = (geom.angular_range_degrees * math.pi / 180.0) / geom.num_views

        for view_idx, angle in enumerate(angles):
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)

            U = D_si + xx * sin_a - yy * cos_a
            s = D_sd * (xx * cos_a + yy * sin_a) / U

            det_idx = s / det_pixel + det_center
            grid_x = det_idx / (0.5 * (geom.detector_bins - 1)) - 1.0
            grid_y = torch.zeros_like(grid_x)

            grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
            row_data = filtered[view_idx].view(1, 1, 1, -1)

            interp = torch.nn.functional.grid_sample(
                row_data, grid, mode="bilinear", padding_mode="zeros",
                align_corners=True,
            ).squeeze()

            weight = (D_si * D_si) / (U * U)
            accum = accum + interp * weight

        return accum * angular_step


class FullyConnectedReconstructor(nn.Module):
    """FC net: learned dense projection-to-image map.

    Works at both surrogate and full resolution.  At full resolution the weight
    matrix is very large (image_size^2 x num_views*detector_bins) so this
    should only be run on a device with sufficient memory.
    """

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs, bias=False)
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

    @classmethod
    def from_geometry(cls, geometry: FanBeamGeometry) -> "FullyConnectedReconstructor":
        n_in = geometry.num_views * geometry.detector_bins
        n_out = geometry.image_size ** 2
        return cls(n_in, n_out)

    def forward(self, sinogram: Tensor) -> Tensor:
        flat = sinogram.flatten(start_dim=-2)
        out = self.linear(flat)
        side = int(out.shape[-1] ** 0.5)
        return torch.relu(out.view(*out.shape[:-1], side, side))


def parameter_counts(geometry: FanBeamGeometry) -> Tuple[int, int]:
    p_ko = geometry.num_views * geometry.detector_bins
    p_fc = (geometry.image_size ** 2) * p_ko
    return p_ko, p_fc
