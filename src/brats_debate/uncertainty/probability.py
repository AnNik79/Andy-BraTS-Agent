"""Validated probability representation for frozen segmentation experts."""

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from ..config import SEGMENTATION_EXPERTS


@dataclass(frozen=True)
class ExpertProbabilityBatch:
    """Aligned expert distributions and their voxel-level availability.

    probabilities has shape ``[B, E, C, D, H, W]`` and availability has
    shape ``[B, E, 1, D, H, W]``. Class channels follow ``class_order``.
    """

    names: tuple[str, ...]
    class_order: tuple[int, ...]
    probabilities: torch.Tensor
    availability: torch.Tensor


def validate_probability_tensor(
    probabilities: torch.Tensor,
    *,
    classes: int | None = None,
    atol: float = 1e-4,
    normalize: bool = False,
) -> torch.Tensor:
    """Validate one expert's ``[B,C,D,H,W]`` simplex-valued prediction.

    ``normalize=True`` is an explicit numerical repair for nonnegative,
    nonzero inputs. It must not be used to hide malformed model outputs.
    """

    if not torch.is_tensor(probabilities):
        raise TypeError("probabilities must be a torch.Tensor")
    if probabilities.ndim != 5:
        raise ValueError("probabilities must have shape [B,C,D,H,W]")
    if probabilities.shape[0] < 1 or probabilities.shape[1] < 2:
        raise ValueError("probabilities require a nonempty batch and at least two classes")
    if classes is not None and probabilities.shape[1] != classes:
        raise ValueError(
            f"expected {classes} probability classes, got {probabilities.shape[1]}"
        )
    p = probabilities.float()
    if not torch.isfinite(p).all():
        raise ValueError("probabilities must be finite")
    if (p < -atol).any():
        raise ValueError("probabilities must be nonnegative")
    p = p.clamp_min(0)
    totals = p.sum(1, keepdim=True)
    if (totals <= atol).any():
        raise ValueError("probabilities must have positive class mass at every voxel")
    if normalize:
        p = p / totals
    elif not torch.allclose(totals, torch.ones_like(totals), atol=atol, rtol=0):
        maximum_error = float((totals - 1).abs().max().detach().cpu())
        raise ValueError(
            f"probabilities must sum to one across classes; max error={maximum_error:.3g}"
        )
    return p


def _availability_tensor(
    availability: torch.Tensor,
    *,
    batch: int,
    spatial: tuple[int, int, int],
) -> torch.Tensor:
    mask = torch.as_tensor(availability).float()
    if mask.ndim == 4:
        mask = mask[:, None]
    if mask.shape != (batch, 1, *spatial):
        raise ValueError(
            "each availability map must have shape [B,1,D,H,W] or [B,D,H,W]"
        )
    if not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
        raise ValueError("availability values must be finite and in [0,1]")
    return mask


def stack_expert_probabilities(
    expert_probabilities: Mapping[str, torch.Tensor],
    *,
    availability: Mapping[str, torch.Tensor] | None = None,
    expert_order: Sequence[str] = SEGMENTATION_EXPERTS,
    class_order: Sequence[int] | None = None,
    require_all: bool = False,
    normalize: bool = False,
) -> ExpertProbabilityBatch:
    """Align available expert predictions on one class order and voxel grid.

    Missing experts are omitted and recorded by ``names``; they are never
    silently replaced by a uniform distribution. Voxel-level abstention is
    represented only by the explicit availability tensor.
    """

    if not expert_probabilities:
        raise ValueError("at least one expert probability tensor is required")
    order = tuple(expert_order)
    unknown = set(expert_probabilities) - set(order)
    if unknown:
        raise ValueError(f"unknown experts: {sorted(unknown)}")
    missing = [name for name in order if name not in expert_probabilities]
    if require_all and missing:
        raise ValueError(f"missing required experts: {missing}")
    names = tuple(name for name in order if name in expert_probabilities)
    if not names:
        raise ValueError("none of the requested experts are present")

    first = validate_probability_tensor(
        expert_probabilities[names[0]], normalize=normalize
    )
    classes = first.shape[1]
    shape = tuple(first.shape)
    class_ids = tuple(range(classes)) if class_order is None else tuple(class_order)
    if len(class_ids) != classes or len(set(class_ids)) != classes:
        raise ValueError("class_order must contain one unique ID per probability channel")

    tensors = [first]
    for name in names[1:]:
        current = validate_probability_tensor(
            expert_probabilities[name], classes=classes, normalize=normalize
        )
        if tuple(current.shape) != shape:
            raise ValueError(
                f"expert {name} is not aligned to the common batch/class/voxel grid"
            )
        if current.device != first.device:
            raise ValueError("all expert probabilities must be on the same device")
        tensors.append(current)

    provided = availability or {}
    extra_availability = set(provided) - set(expert_probabilities)
    if extra_availability:
        raise ValueError(
            f"availability supplied for absent experts: {sorted(extra_availability)}"
        )
    masks = []
    for name in names:
        raw = provided.get(name)
        if raw is None:
            raw = torch.ones(
                shape[0], 1, *shape[2:], dtype=first.dtype, device=first.device
            )
        mask = _availability_tensor(raw, batch=shape[0], spatial=shape[2:])
        if mask.device != first.device:
            mask = mask.to(first.device)
        masks.append(mask)

    return ExpertProbabilityBatch(
        names=names,
        class_order=class_ids,
        probabilities=torch.stack(tensors, dim=1),
        availability=torch.stack(masks, dim=1),
    )


def masked_mean_probability(batch: ExpertProbabilityBatch) -> torch.Tensor:
    """Mean available expert distribution at each voxel.

    This is exactly the professor's equal-weight mean when all four expert
    availability maps are one.
    """

    count = batch.availability.sum(1)
    if (count <= 0).any():
        raise ValueError("at least one expert must be available at every voxel")
    return (batch.probabilities * batch.availability).sum(1) / count
