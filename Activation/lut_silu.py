"""Differentiable piecewise-linear LUT replacement for PyTorch SiLU."""

from __future__ import annotations

import torch
from torch import nn


class LUTSiLU(nn.Module):
    """Approximate SiLU with 21 uniformly spaced values and interpolation.

    The tails intentionally match ``Activation/comparesilu.cpp``: values at or
    below -5 become zero, while values at or above 5 use the identity.
    """

    def __init__(self, points: int = 21, minimum: float = -5.0, maximum: float = 5.0) -> None:
        super().__init__()
        if points < 2 or maximum <= minimum:
            raise ValueError("LUT requires at least two points and maximum > minimum")

        samples = torch.linspace(minimum, maximum, points, dtype=torch.float32)
        values = samples * torch.sigmoid(samples)
        self.register_buffer("values", values)
        self.register_buffer("deltas", values[1:] - values[:-1])
        self.points = points
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.scale = (points - 1) / (maximum - minimum)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        clipped = tensor.clamp(self.minimum, self.maximum)
        position = (clipped - self.minimum) * self.scale
        index = position.to(torch.long).clamp_(0, self.points - 2)
        fraction = position - index.to(position.dtype)
        interpolated = self.values[index] + fraction * self.deltas[index]
        below = torch.where(tensor <= self.minimum, torch.zeros_like(interpolated), interpolated)
        return torch.where(tensor >= self.maximum, tensor, below)

    def extra_repr(self) -> str:
        return f"points={self.points}, range=({self.minimum:g}, {self.maximum:g})"


def replace_silu(module: nn.Module, replacement: LUTSiLU | None = None) -> int:
    """Replace every registered ``nn.SiLU`` reference, preserving sharing."""
    lut = replacement if replacement is not None else LUTSiLU()
    count = 0
    for name, child in list(module._modules.items()):
        if child is None:
            continue
        if isinstance(child, nn.SiLU):
            module._modules[name] = lut
            count += 1
        else:
            count += replace_silu(child, lut)
    return count
