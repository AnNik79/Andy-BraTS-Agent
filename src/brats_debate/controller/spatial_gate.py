"""Matched pointwise and local-context gates.

These modules do not replace the frozen Beta 1 ``GatingNetwork``.
Both use the same masked-softmax fusion. Only the spatial support differs.
"""

import torch
from torch import nn

from ..config import SEGMENTATION_EXPERTS


def _masked_fusion(logits, probabilities, availability):
    if (availability.sum(1) == 0).any():
        raise ValueError("At least one expert must cover every voxel")
    weights = logits.masked_fill(availability <= 0, -1e4).softmax(1)
    return {"weights": weights, "probabilities": (probabilities * weights.unsqueeze(2)).sum(1)}


class PointwiseMatchedGate(nn.Module):
    """Two kernel-size-1 convolutions, with width chosen near the spatial gate's parameter count."""

    def __init__(self, in_channels, width=11, experts=None):
        experts = len(SEGMENTATION_EXPERTS) if experts is None else experts
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(in_channels, width, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(width, experts, kernel_size=1),
        )

    def forward(self, features, probabilities, availability):
        return _masked_fusion(self.network(features), probabilities, availability)


class SpatialContextGate(nn.Module):
    """Pointwise projection, depthwise 3x3x3 context, then pointwise expert logits."""

    def __init__(self, in_channels, width=8, experts=None):
        experts = len(SEGMENTATION_EXPERTS) if experts is None else experts
        super().__init__()
        self.reduce = nn.Conv3d(in_channels, width, kernel_size=1)
        self.context = nn.Conv3d(width, width, kernel_size=3, padding=1, groups=width)
        self.project = nn.Conv3d(width, experts, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, features, probabilities, availability):
        hidden = self.activation(self.context(self.activation(self.reduce(features))))
        return _masked_fusion(self.project(hidden), probabilities, availability)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
