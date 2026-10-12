"""Beta 2 uncertainty views built on the frozen Beta 1 debate computation."""

import math
from typing import Mapping

import torch
import torch.nn.functional as F

from ..config import SEGMENTATION_EXPERTS
from ..debate.disagreement import analyze_disagreement
from .probability import masked_mean_probability, stack_expert_probabilities


def entropy_map(
    probabilities: torch.Tensor,
    *,
    class_dim: int = 1,
    normalized: bool = True,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Shannon entropy with the class dimension retained."""

    p = probabilities.float()
    terms = torch.where(
        p > 0, p * p.clamp_min(epsilon).log(), torch.zeros_like(p)
    )
    entropy = -terms.sum(class_dim, keepdim=True)
    if normalized:
        entropy = entropy / math.log(probabilities.shape[class_dim])
    return entropy


def _output_availability(output: Mapping[str, object]) -> torch.Tensor:
    probabilities = output["probabilities"]
    if not torch.is_tensor(probabilities):
        raise TypeError("expert output probabilities must be a torch.Tensor")
    extra = output.get("extra") or {}
    availability = extra.get(
        "availability",
        torch.ones(
            probabilities.shape[0],
            1,
            *probabilities.shape[2:],
            device=probabilities.device,
        ),
    )
    if availability.ndim == 4:
        availability = availability[:, None]
    return availability


def _as_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(mask, device=reference.device)
    if value.ndim == reference.ndim - 1:
        value = value[:, None]
    if value.ndim == reference.ndim - 2:
        value = value[None, None]
    if value.shape != reference.shape:
        try:
            value = torch.broadcast_to(value, reference.shape)
        except RuntimeError as exc:
            raise ValueError("summary mask is not aligned with uncertainty maps") from exc
    return value.bool()


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> float | None:
    selected = value[mask]
    if selected.numel() == 0:
        return None
    return float(selected.float().mean().detach().cpu())


def summarize_uncertainty_maps(
    analysis: Mapping[str, object],
    masks: Mapping[str, torch.Tensor],
    *,
    boundary_mask: torch.Tensor | None = None,
) -> dict[str, dict[str, float | None]]:
    """Summarize uncertainty/disagreement over caller-defined spatial masks.

    Region masks may come from prediction or ground truth, but callers must
    label that provenance; this function does not infer anatomy.
    """

    maps = {
        "mean_predictive_entropy": analysis["mean_predictive_entropy"],
        "mean_individual_entropy": analysis["mean_individual_entropy"],
        "js_divergence": analysis["js_divergence"],
        "vote_disagreement": analysis["vote_disagreement"],
        "disagreement_map": analysis["production"]["map"],
    }
    reference = maps["js_divergence"]
    named_masks = dict(masks)
    if boundary_mask is not None:
        boundary = _as_mask(boundary_mask, reference)
        named_masks["boundary"] = boundary
        named_masks["non_boundary"] = ~boundary

    summaries = {}
    for name, raw_mask in named_masks.items():
        mask = _as_mask(raw_mask, reference)
        summaries[name] = {
            metric: _masked_mean(value, mask) for metric, value in maps.items()
        }
        summaries[name]["voxel_count"] = int(mask.sum().detach().cpu())
    return summaries


def analyze_uncertainty(
    outputs: Mapping[str, Mapping[str, object]],
    *,
    class_order: tuple[int, ...] | None = None,
    region_masks: Mapping[str, torch.Tensor] | None = None,
    boundary_mask: torch.Tensor | None = None,
) -> dict[str, object]:
    """Separate individual entropy from cross-expert disagreement.

    The production Beta 1 debate result is reused verbatim and returned under
    ``production``. Additional terms expose both definitions of mixture JS:
    weighted KL to the mean and entropy-of-mean minus mean entropy.
    """

    ordered_names = tuple(name for name in SEGMENTATION_EXPERTS if name in outputs)
    if not ordered_names:
        raise ValueError("at least one segmentation expert output is required")
    ordered_outputs = {}
    for name in ordered_names:
        extra = dict(outputs[name].get("extra") or {})
        extra["availability"] = _output_availability(outputs[name])
        ordered_outputs[name] = {**outputs[name], "extra": extra}
    batch = stack_expert_probabilities(
        {name: outputs[name]["probabilities"] for name in ordered_names},
        availability={
            name: ordered_outputs[name]["extra"]["availability"]
            for name in ordered_names
        },
        expert_order=SEGMENTATION_EXPERTS,
        class_order=class_order,
    )
    mean = masked_mean_probability(batch)
    classes = batch.probabilities.shape[2]
    log_classes = math.log(classes)

    individual_raw = entropy_map(
        batch.probabilities, class_dim=2, normalized=False
    )
    count = batch.availability.sum(1)
    expected_raw = (individual_raw * batch.availability).sum(1) / count
    mean_raw = entropy_map(mean, class_dim=1, normalized=False)
    js_identity_raw = (mean_raw - expected_raw).clamp_min(0)

    p = batch.probabilities
    mixture = mean[:, None].clamp_min(1e-8)
    safe_p = p.clamp_min(1e-8)
    kl_terms = torch.where(
        p > 0, p * (safe_p.log() - mixture.log()), torch.zeros_like(p)
    )
    kl = kl_terms.sum(2, keepdim=True)
    js_kl_raw = (kl * batch.availability).sum(1) / count

    if len(ordered_outputs) > 1:
        production = analyze_disagreement(ordered_outputs)
    else:
        availability = batch.availability[:, 0]
        votes = (
            F.one_hot(mean.argmax(1), classes)
            .movedim(-1, 1)
            .float()
            * availability
        )
        zeros = torch.zeros_like(mean_raw)
        production = {
            "map": zeros,
            "js_divergence": zeros,
            "variance": torch.zeros_like(mean),
            "prediction_entropy": mean_raw / log_classes,
            "votes": votes,
            "pairwise": {},
            "average_probabilities": mean,
            "majority_segmentation": votes.argmax(1),
            "available_experts": count,
        }
    vote_disagreement = (
        1 - production["votes"].max(1, keepdim=True).values / count
    ).clamp(0, 1)
    normalized_individual = individual_raw / log_classes
    result = {
        "expert_names": batch.names,
        "class_order": batch.class_order,
        "individual_entropy": {
            name: normalized_individual[:, index]
            for index, name in enumerate(batch.names)
        },
        "individual_entropy_raw": {
            name: individual_raw[:, index]
            for index, name in enumerate(batch.names)
        },
        "mean_individual_entropy": expected_raw / log_classes,
        "mean_predictive_entropy": mean_raw / log_classes,
        "mean_probabilities": mean,
        "js_divergence": js_identity_raw / log_classes,
        "js_from_weighted_kl": js_kl_raw / log_classes,
        "js_identity_residual": (js_identity_raw - js_kl_raw).abs() / log_classes,
        "probability_variance": production["variance"],
        "vote_disagreement": vote_disagreement,
        "pairwise": production["pairwise"],
        "available_experts": count,
        "equal_weight_four_expert_formula": (
            batch.names == tuple(SEGMENTATION_EXPERTS)
            and bool((batch.availability == 1).all().detach().cpu())
        ),
        "production": production,
    }
    if region_masks or boundary_mask is not None:
        result["summaries"] = summarize_uncertainty_maps(
            result, region_masks or {}, boundary_mask=boundary_mask
        )
    return result
