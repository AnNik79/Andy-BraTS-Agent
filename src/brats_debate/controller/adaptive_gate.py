"""Residual refinement of the frozen four-expert gate.

The frozen Beta 1 weights stay an input. A small network adds a logit
residual, unavailable experts are set to weight zero, and the class
distribution is the weighted mixture of the expert probabilities.
Ground truth is not an input.
"""

from typing import Sequence

import torch
from torch import nn

from ..config import SEGMENTATION_EXPERTS


BASE_FEATURE_NAMES = (
    "js_disagreement",
    "predictive_entropy",
    "boundary_probability",
    "boundary_distance",
    "mri_t1n",
    "mri_t1c",
    "mri_t2w",
    "mri_t2f",
)
EXPERT_ORDER = tuple(SEGMENTATION_EXPERTS)
CLASS_COUNT = 4


def feature_names(
    expert_order: Sequence[str] = EXPERT_ORDER,
    class_count: int = CLASS_COUNT,
) -> tuple[str, ...]:
    names = [f"prob_{expert}_class{label}" for expert in expert_order for label in range(class_count)]
    names.extend(f"entropy_{expert}" for expert in expert_order)
    names.extend(f"available_{expert}" for expert in expert_order)
    names.extend(f"frozen_weight_{expert}" for expert in expert_order)
    names.extend(BASE_FEATURE_NAMES)
    return tuple(names)


def assemble_features(
    expert_probabilities: torch.Tensor,
    expert_entropy: torch.Tensor,
    availability: torch.Tensor,
    frozen_weights: torch.Tensor,
    base_features: torch.Tensor,
) -> torch.Tensor:
    """Build the gate input. None of these tensors may be a label."""

    if expert_probabilities.ndim != 3:
        raise ValueError("expert probabilities must be [voxel, expert, class]")
    voxels, experts, classes = expert_probabilities.shape
    expected = (
        (expert_entropy, experts),
        (availability, experts),
        (frozen_weights, experts),
        (base_features, len(BASE_FEATURE_NAMES)),
    )
    for tensor, width in expected:
        if tensor.shape != (voxels, width):
            raise ValueError(f"gate feature width mismatch: expected {(voxels, width)}, got {tuple(tensor.shape)}")
    if not torch.isfinite(expert_probabilities).all() or not torch.isfinite(base_features).all():
        raise ValueError("gate features must be finite")
    return torch.cat(
        (
            expert_probabilities.reshape(voxels, experts * classes),
            expert_entropy,
            availability,
            frozen_weights,
            base_features,
        ),
        dim=1,
    )


class ResidualAdaptiveGate(nn.Module):
    """Add a zero-initialized residual to the frozen gate logits."""

    def __init__(self, in_features: int, hidden: int = 32, experts: int = 4):
        super().__init__()
        if hidden < 1 or experts < 1:
            raise ValueError("gate width must be positive")
        self.in_features = in_features
        self.experts = experts
        self.network = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, experts),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        expert_probabilities: torch.Tensor,
        availability: torch.Tensor,
        frozen_weights: torch.Tensor,
        correction_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if features.shape[-1] != self.in_features:
            raise ValueError("feature width does not match the trained gate")
        if expert_probabilities.shape[1] != self.experts:
            raise ValueError("expert axis does not match the trained gate")
        if (availability.sum(1) <= 0).any():
            raise ValueError("at least one expert must be available at every voxel")
        available = availability > 0
        residual = self.network(features)
        if correction_mask is not None:
            mask = correction_mask.reshape(-1, 1).to(dtype=residual.dtype)
            if mask.shape[0] != residual.shape[0]:
                raise ValueError("correction mask does not match the voxel count")
            residual = residual * mask
        logits = frozen_weights.clamp_min(1e-8).log() + residual
        weights = logits.masked_fill(~available, -1e4).softmax(1)
        weights = torch.where(available, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(1, keepdim=True)
        probabilities = (expert_probabilities * weights.unsqueeze(2)).sum(1)
        return {"weights": weights, "probabilities": probabilities, "residual_logits": residual}
