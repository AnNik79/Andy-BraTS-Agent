import torch
from torch import nn
from ..config import SEGMENTATION_EXPERTS


class GatingNetwork(nn.Module):
    """Voxel-level gate over the four segmentation experts only. The LLM is not a gate input."""

    def __init__(self, in_channels, width=8, experts=None):
        experts = len(SEGMENTATION_EXPERTS) if experts is None else experts
        super().__init__()
        self.network = nn.Sequential(nn.Conv3d(in_channels, width, 1), nn.GELU(),
                                     nn.Conv3d(width, experts, 1))

    def forward(self, features, probabilities, availability):
        if (availability.sum(1) == 0).any():
            raise ValueError("At least one expert must cover every voxel")
        logits = self.network(features).masked_fill(availability <= 0, -1e4)
        weights = logits.softmax(1)
        return {"weights": weights, "probabilities": (probabilities * weights.unsqueeze(2)).sum(1)}
