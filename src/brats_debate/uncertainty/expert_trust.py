"""Ground-truth analysis of how the frozen controller weights its experts."""

from typing import Mapping, Sequence

import torch

from ..config import SEGMENTATION_EXPERTS
from ..evaluation.metrics import BRATS_REGIONS
from .probability import stack_expert_probabilities, validate_probability_tensor


DEFAULT_REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}


def _target_tensor(
    target: torch.Tensor, *, batch: int, spatial: tuple[int, int, int], classes: int
) -> torch.Tensor:
    value = torch.as_tensor(target)
    if value.ndim == 3:
        value = value[None]
    if value.shape != (batch, *spatial):
        raise ValueError("target must have shape [B,D,H,W] or [D,H,W]")
    if value.is_floating_point() and not torch.equal(value, value.round()):
        raise ValueError("target labels must be integers")
    value = value.long()
    if value.min() < 0 or value.max() >= classes:
        raise ValueError("target contains a class outside the probability channels")
    return value


def _spatial_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    value = torch.as_tensor(mask, device=reference.device)
    if value.ndim == 3:
        value = value[None]
    if value.ndim == 5 and value.shape[1] == 1:
        value = value[:, 0]
    if value.shape != reference.shape:
        raise ValueError("stratification mask must align with target [B,D,H,W]")
    return value.bool()


def _fraction(event: torch.Tensor, mask: torch.Tensor) -> float | None:
    denominator = int(mask.sum().detach().cpu())
    if denominator == 0:
        return None
    return float(event[mask].float().mean().detach().cpu())


def _mean(value: torch.Tensor, mask: torch.Tensor) -> float | None:
    denominator = int(mask.sum().detach().cpu())
    if denominator == 0:
        return None
    return float(value[mask].float().mean().detach().cpu())


def _correlation(x: torch.Tensor, y: torch.Tensor) -> float | None:
    x, y = x.float().flatten(), y.float().flatten()
    if x.numel() < 2:
        return None
    x, y = x - x.mean(), y - y.mean()
    denominator = x.square().sum().sqrt() * y.square().sum().sqrt()
    if float(denominator.detach().cpu()) == 0:
        return None
    return float(((x * y).sum() / denominator).detach().cpu())


def _summarize(
    mask: torch.Tensor,
    *,
    controller_correct: torch.Tensor,
    oracle_coverage: torch.Tensor,
    all_experts_wrong: torch.Tensor,
    all_experts_correct: torch.Tensor,
    maximum_weight_correct: torch.Tensor,
    correct_weight_mass: torch.Tensor,
    fusion_fix_no_hard_expert: torch.Tensor,
    fusion_fix_partial_failure: torch.Tensor,
    fusion_harm: torch.Tensor,
) -> dict[str, float | int | None]:
    voxels = int(mask.sum().detach().cpu())
    controller_accuracy = _fraction(controller_correct, mask)
    oracle_accuracy = _fraction(oracle_coverage, mask)
    return {
        "voxel_count": voxels,
        "controller_accuracy": controller_accuracy,
        "hard_expert_oracle_coverage": oracle_accuracy,
        "controller_vs_hard_oracle_gap": (
            None
            if controller_accuracy is None or oracle_accuracy is None
            else oracle_accuracy - controller_accuracy
        ),
        "all_experts_wrong_fraction": _fraction(all_experts_wrong, mask),
        "all_experts_correct_fraction": _fraction(all_experts_correct, mask),
        "maximum_weight_expert_correct_fraction": _fraction(
            maximum_weight_correct, mask
        ),
        "mean_correct_expert_weight_mass": _mean(correct_weight_mass, mask),
        "mean_correct_expert_weight_mass_when_covered": _mean(
            correct_weight_mass, mask & oracle_coverage
        ),
        "fusion_correct_without_hard_expert_fraction": _fraction(
            fusion_fix_no_hard_expert, mask
        ),
        "fusion_recovers_partial_failure_fraction": _fraction(
            fusion_fix_partial_failure, mask
        ),
        "fusion_harm_when_correct_expert_available_fraction": _fraction(
            fusion_harm, mask
        ),
    }


def analyze_expert_trust(
    expert_probabilities: Mapping[str, torch.Tensor],
    controller_weights: torch.Tensor,
    target: torch.Tensor,
    *,
    controller_probabilities: torch.Tensor | None = None,
    availability: Mapping[str, torch.Tensor] | None = None,
    expert_order: Sequence[str] = SEGMENTATION_EXPERTS,
    regions: Mapping[str, Sequence[int]] | None = None,
    boundary_mask: torch.Tensor | None = None,
    disagreement: torch.Tensor | None = None,
    high_disagreement_threshold: float = 0.25,
) -> dict[str, object]:
    """Compare controller allocation with voxel-level expert correctness.

    Ground truth is mandatory, so this function is for permitted labeled
    analysis only. It neither trains nor changes the frozen gate.
    """

    batch = stack_expert_probabilities(
        expert_probabilities,
        availability=availability,
        expert_order=expert_order,
    )
    p = batch.probabilities
    available = batch.availability[:, :, 0] > 0
    batch_size, experts, classes, *spatial_list = p.shape
    spatial = tuple(spatial_list)
    truth = _target_tensor(
        target, batch=batch_size, spatial=spatial, classes=classes
    ).to(p.device)

    weights = torch.as_tensor(controller_weights, device=p.device).float()
    if weights.shape != (batch_size, experts, *spatial):
        raise ValueError("controller_weights must have shape [B,E,D,H,W]")
    if (
        not torch.isfinite(weights).all()
        or (weights < -1e-6).any()
        or (weights > 1 + 1e-6).any()
    ):
        raise ValueError("controller weights must be finite and in [0,1]")
    if not torch.allclose(
        weights.sum(1), torch.ones_like(weights[:, 0]), atol=1e-5, rtol=0
    ):
        raise ValueError("controller weights must sum to one at every voxel")
    if (weights.masked_select(~available).abs() > 1e-6).any():
        raise ValueError("unavailable experts must have zero controller weight")
    if (~available).all(1).any():
        raise ValueError("at least one expert must be available at every voxel")

    fused = (
        (p * weights[:, :, None]).sum(1)
        if controller_probabilities is None
        else validate_probability_tensor(
            torch.as_tensor(controller_probabilities, device=p.device),
            classes=classes,
        )
    )
    if tuple(fused.shape) != (batch_size, classes, *spatial):
        raise ValueError("controller_probabilities must align with expert predictions")

    expert_labels = p.argmax(2)
    expert_correct = (expert_labels == truth[:, None]) & available
    available_count = available.sum(1)
    correct_count = expert_correct.sum(1)
    oracle_coverage = correct_count > 0
    all_experts_wrong = ~oracle_coverage
    all_experts_correct = correct_count == available_count
    correct_weight_mass = (weights * expert_correct).sum(1)
    incorrect_weight_mass = (weights * (available & ~expert_correct)).sum(1)
    maximum_weight_index = weights.argmax(1, keepdim=True)
    maximum_weight_correct = expert_correct.gather(
        1, maximum_weight_index
    ).squeeze(1)

    controller_prediction = fused.argmax(1)
    controller_correct = controller_prediction == truth
    partial_failure = oracle_coverage & ~all_experts_correct
    fusion_fix_no_hard_expert = controller_correct & all_experts_wrong
    fusion_fix_partial_failure = controller_correct & partial_failure
    fusion_harm = ~controller_correct & oracle_coverage

    maps = {
        "expert_correct": expert_correct,
        "correct_expert_count": correct_count,
        "correct_expert_weight_mass": correct_weight_mass,
        "incorrect_expert_weight_mass": incorrect_weight_mass,
        "maximum_weight_expert_index": maximum_weight_index.squeeze(1),
        "maximum_weight_expert_correct": maximum_weight_correct,
        "hard_expert_oracle_coverage": oracle_coverage,
        "all_experts_wrong": all_experts_wrong,
        "all_experts_correct": all_experts_correct,
        "controller_correct": controller_correct,
        "fusion_correct_without_hard_expert": fusion_fix_no_hard_expert,
        "fusion_recovers_partial_failure": fusion_fix_partial_failure,
        "fusion_harm": fusion_harm,
    }
    all_voxels = torch.ones_like(truth, dtype=torch.bool)
    summary = _summarize(
        all_voxels,
        controller_correct=controller_correct,
        oracle_coverage=oracle_coverage,
        all_experts_wrong=all_experts_wrong,
        all_experts_correct=all_experts_correct,
        maximum_weight_correct=maximum_weight_correct,
        correct_weight_mass=correct_weight_mass,
        fusion_fix_no_hard_expert=fusion_fix_no_hard_expert,
        fusion_fix_partial_failure=fusion_fix_partial_failure,
        fusion_harm=fusion_harm,
    )

    stratification_masks = {"all": all_voxels}
    for name, ids in (regions or DEFAULT_REGIONS).items():
        if name in BRATS_REGIONS:
            region = torch.zeros_like(truth, dtype=torch.bool)
            for class_id in ids:
                region |= truth == int(class_id)
            stratification_masks[f"ground_truth_{name}"] = region
    if boundary_mask is not None:
        boundary = _spatial_mask(boundary_mask, truth)
        stratification_masks["boundary"] = boundary
        stratification_masks["non_boundary"] = ~boundary
    if disagreement is not None:
        score = torch.as_tensor(disagreement, device=truth.device).float()
        if score.ndim == 5 and score.shape[1] == 1:
            score = score[:, 0]
        if score.shape != truth.shape or not torch.isfinite(score).all():
            raise ValueError("disagreement must be a finite map aligned with target")
        if not 0 <= high_disagreement_threshold <= 1:
            raise ValueError("high_disagreement_threshold must be in [0,1]")
        stratification_masks["low_disagreement"] = (
            score < high_disagreement_threshold
        )
        stratification_masks["high_disagreement"] = (
            score >= high_disagreement_threshold
        )

    stratified = {
        name: _summarize(
            mask,
            controller_correct=controller_correct,
            oracle_coverage=oracle_coverage,
            all_experts_wrong=all_experts_wrong,
            all_experts_correct=all_experts_correct,
            maximum_weight_correct=maximum_weight_correct,
            correct_weight_mass=correct_weight_mass,
            fusion_fix_no_hard_expert=fusion_fix_no_hard_expert,
            fusion_fix_partial_failure=fusion_fix_partial_failure,
            fusion_harm=fusion_harm,
        )
        for name, mask in stratification_masks.items()
    }
    correlations = {}
    for index, name in enumerate(batch.names):
        valid = available[:, index]
        correlations[name] = {
            "weight_vs_correctness": _correlation(
                weights[:, index][valid], expert_correct[:, index][valid]
            ),
            "weight_vs_error": _correlation(
                weights[:, index][valid], (~expert_correct[:, index][valid])
            ),
        }

    return {
        "definition": (
            "Correct expert means its hard top-class prediction equals ground truth; "
            "probability fusion can still outperform hard expert selection."
        ),
        "expert_names": batch.names,
        "summary": summary,
        "stratified": stratified,
        "weight_correctness_correlations": correlations,
        "maps": maps,
        "controller_prediction": controller_prediction,
        "controller_probabilities": fused,
    }
