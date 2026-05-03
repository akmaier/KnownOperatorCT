"""Known operator and fully connected reconstruction models.

The known operator network mirrors the architecture of W\"urfl et al. (2018) and
Maier et al. (2019): a trainable diagonal weighting layer in the projection
domain followed by a fixed reconstruction filter and a fixed backprojection.
The fully connected counterfactual replaces the backprojection by a dense
learned linear layer.
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


class KnownOperatorReconstructor(nn.Module):
    """KO net: trainable diagonal W, fixed K, fixed A^T, ReLU."""

    def __init__(self, geometry: FanBeamGeometry) -> None:
        super().__init__()
        self.geometry = geometry
        # Trainable diagonal weighting: one weight per detector bin and angle.
        weights_init = self._parker_short_scan_weights()
        self.weights = nn.Parameter(weights_init)
        self.register_buffer("ramp", ramp_filter(geometry.detector_bins))
        self.register_buffer("backproj_matrix", self._build_backprojection_matrix())

    def _parker_short_scan_weights(self) -> Tensor:
        n_views, n_bins = self.geometry.num_views, self.geometry.detector_bins
        return torch.ones(n_views, n_bins, dtype=torch.float32)

    def _build_backprojection_matrix(self) -> Tensor:
        # We materialize the transpose of the forward projection of a basis
        # image. For full-resolution settings this matrix is large; we keep it
        # implicit through fan_beam_forward instead. Here we return an empty
        # tensor and use the implicit operator path in forward().
        return torch.empty(0)

    def forward(self, sinogram: Tensor) -> Tensor:
        geometry = self.geometry
        weighted = sinogram * self.weights
        # Filter along the detector axis using the ramp filter.
        filtered_freq = torch.fft.fft(weighted, dim=-1) * self.ramp.to(weighted.device)
        filtered = torch.real(torch.fft.ifft(filtered_freq, dim=-1))
        image = self._adjoint_forward(filtered)
        return torch.relu(image)

    def _adjoint_forward(self, filtered: Tensor) -> Tensor:
        # Implicit backprojection: sum the filtered sinogram smeared back
        # along each view direction.
        side = self.geometry.image_size
        coords = torch.linspace(-1.0, 1.0, side, device=filtered.device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        accum = torch.zeros(side, side, dtype=filtered.dtype, device=filtered.device)
        for view_idx, angle in enumerate(self.geometry.angles_rad.to(filtered.device)):
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            t = cos_a * xx + sin_a * yy
            sample = torch.nn.functional.grid_sample(
                filtered[view_idx].view(1, 1, 1, -1),
                torch.stack([t, torch.zeros_like(t)], dim=-1).unsqueeze(0),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze()
            accum = accum + sample
        return accum / self.geometry.num_views


class FullyConnectedReconstructor(nn.Module):
    """FC net at surrogate scale: learned dense projection-to-image map."""

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_inputs, num_outputs, bias=False)
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

    def forward(self, sinogram: Tensor) -> Tensor:
        flat = sinogram.flatten(start_dim=-2)
        out = self.linear(flat)
        side = int(out.shape[-1] ** 0.5)
        return torch.relu(out.view(*out.shape[:-1], side, side))


def parameter_counts(geometry: FanBeamGeometry) -> Tuple[int, int]:
    p_ko = geometry.num_views * geometry.detector_bins
    p_fc = (geometry.image_size ** 2) * p_ko
    return p_ko, p_fc
