"""Selective extra HighRes windows for uncertain, currently uncovered voxels.

The four frozen experts are not modified. Ground truth is not an input.
A case with no selected window keeps the existing HighRes probabilities
and availability exactly.
"""

from typing import Mapping, Sequence

import torch

from ..inference.patches import patch_slices, window_starts


ACTIVE_PROTOCOL = {
    "action": "additional_highres_windows",
    "model_change": "none",
    "patch": [32, 32, 32],
    "overlap": 0.5,
    "trigger": "highres_unavailable_and_hard_disagreement_or_entropy",
    "entropy_threshold": 0.5,
    "min_eligible_fraction": 0.02,
    "max_actions": 16,
    "blend": "fill_previously_unavailable_voxels",
    "thresholds_from": "prespecified_stage2_entropy_and_hard_disagreement",
    "stage2_evaluation_subjects_used_for_thresholds": False,
}


def _spatial(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach()
    while value.ndim > 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 3:
        raise ValueError(f"expected a spatial map, got {tuple(value.shape)}")
    return value


def eligible_voxels(
    highres_availability: torch.Tensor,
    expert_probabilities: torch.Tensor,
    expert_availability: torch.Tensor,
    entropy: torch.Tensor,
    *,
    entropy_threshold: float = 0.5,
) -> torch.Tensor:
    """Mark voxels that lack HighRes evidence and already look unreliable.

    ``expert_probabilities`` is ``[expert, class, D, H, W]``. Labels are not read.
    """

    if expert_probabilities.ndim != 5:
        raise ValueError("expert probabilities must be [expert, class, D, H, W]")
    availability = expert_availability.detach()
    if availability.ndim == 6 and availability.shape[1] == 1:
        availability = availability[:, 0]
    if availability.shape[0] != expert_probabilities.shape[0] or availability.shape[1:] != expert_probabilities.shape[2:]:
        raise ValueError("expert availability does not match the probability grid")
    hard = expert_probabilities.detach().argmax(1)
    present = availability > 0
    highest = torch.where(present, hard, torch.full_like(hard, -1)).max(0).values
    lowest = torch.where(present, hard, torch.full_like(hard, 99)).min(0).values
    disagree = (highest != lowest) & (present.sum(0) >= 2)
    uncertain = disagree | (_spatial(entropy) >= entropy_threshold)
    return (_spatial(highres_availability) <= 0) & uncertain


def rank_windows(
    eligible: torch.Tensor,
    *,
    patch: Sequence[int] = (32, 32, 32),
    overlap: float = 0.5,
    min_eligible_fraction: float = 0.02,
) -> list[tuple[float, tuple[int, ...]]]:
    """Rank in-grid windows by the fraction of eligible voxels. Ties use window origin."""

    mask = eligible.detach().bool()
    if mask.ndim != 3:
        raise ValueError("eligibility must be a 3D map")
    patch = tuple(int(size) for size in patch)
    ranked = []
    for start in window_starts(tuple(mask.shape), patch, overlap):
        window = mask[patch_slices(start, patch)]
        if window.numel() == 0:
            continue
        fraction = float(window.float().mean())
        if fraction >= min_eligible_fraction:
            ranked.append((fraction, tuple(int(value) for value in start)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return ranked


def select_actions(
    ranked: Sequence[tuple[float, tuple[int, ...]]],
    *,
    max_actions: int = 16,
) -> list[tuple[int, ...]]:
    if max_actions < 0:
        raise ValueError("the action budget cannot be negative")
    return [start for _score, start in list(ranked)[:max_actions]]


def _ranked_proposal_starts(proposal: torch.Tensor, shape: Sequence[int], patch: Sequence[int], overlap: float):
    score_map = _spatial(proposal).float()
    patch = tuple(int(size) for size in patch)
    shape = tuple(int(size) for size in shape)
    scored = []
    for start in window_starts(shape, patch, overlap):
        origin = tuple(int(value) for value in start)
        scored.append((float(score_map[patch_slices(origin, patch)].mean()), origin))
    ordered = [start for score, start in sorted(scored, reverse=True) if score > 0]
    if not ordered:
        ordered = [tuple(max(0, (size - length) // 2) for size, length in zip(shape, patch))]
    return ordered


def executed_highres_starts(
    proposal: torch.Tensor,
    shape: Sequence[int],
    *,
    patch: Sequence[int] = (32, 32, 32),
    overlap: float = 0.5,
    max_patches: int = 128,
) -> list[tuple[int, ...]]:
    """Reproduce the proposal-limited windows ``sliding_predict`` already ran."""

    return _ranked_proposal_starts(proposal, shape, patch, overlap)[: max(1, int(max_patches))]


def plan_proposal_extension(
    proposal: torch.Tensor,
    shape: Sequence[int],
    *,
    patch: Sequence[int] = (32, 32, 32),
    overlap: float = 0.5,
    already_run: int = 128,
    max_actions: int = 16,
) -> dict[str, object]:
    """Request the next proposal-ranked HighRes windows after the ones already run.

    This does not use the 2% uncovered-uncertainty rule and does not read labels.
    """

    starts = matched_control_starts(
        proposal,
        shape,
        patch=patch,
        overlap=overlap,
        already_run=already_run,
        count=max_actions,
    )
    return {
        "action": "next_uncomputed_proposal_windows",
        "starts": starts,
        "fallback_to_stage2": len(starts) == 0,
        "max_actions": int(max_actions),
    }


def matched_control_starts(
    proposal: torch.Tensor,
    shape: Sequence[int],
    *,
    patch: Sequence[int] = (32, 32, 32),
    overlap: float = 0.5,
    already_run: int = 128,
    count: int = 0,
) -> list[tuple[int, ...]]:
    """Take the next proposal-ranked windows after the ones already computed."""

    ordered = _ranked_proposal_starts(proposal, shape, patch, overlap)
    executed = set(ordered[: max(1, int(already_run))])
    remainder = [start for start in ordered if start not in executed]
    return remainder[: max(0, int(count))]


def far_proposal_starts(
    proposal: torch.Tensor,
    shape: Sequence[int],
    *,
    patch: Sequence[int] = (32, 32, 32),
    overlap: float = 0.5,
    already_run: int = 128,
    count: int = 0,
) -> list[tuple[int, ...]]:
    """Equal-budget control: the weakest positive proposal windows not already run."""

    ordered = _ranked_proposal_starts(proposal, shape, patch, overlap)
    executed = set(ordered[: max(1, int(already_run))])
    remainder = [start for start in ordered if start not in executed]
    return remainder[-max(0, int(count)) :]


def plan_additional_highres(
    highres_availability: torch.Tensor,
    expert_probabilities: torch.Tensor,
    expert_availability: torch.Tensor,
    entropy: torch.Tensor,
    *,
    patch: Sequence[int] = (32, 32, 32),
    overlap: float = 0.5,
    entropy_threshold: float = 0.5,
    min_eligible_fraction: float = 0.02,
    max_actions: int = 16,
    excluded_starts: Sequence[Sequence[int]] = (),
) -> dict[str, object]:
    """Choose extra HighRes windows. This function does not run a network."""

    eligible = eligible_voxels(
        highres_availability,
        expert_probabilities,
        expert_availability,
        entropy,
        entropy_threshold=entropy_threshold,
    )
    excluded = {tuple(int(value) for value in start) for start in excluded_starts}
    ranked = [
        item
        for item in rank_windows(
            eligible,
            patch=patch,
            overlap=overlap,
            min_eligible_fraction=min_eligible_fraction,
        )
        if item[1] not in excluded
    ]
    starts = select_actions(ranked, max_actions=max_actions)
    return {
        "eligible_voxels": int(eligible.sum()),
        "candidate_windows": len(ranked),
        "starts": starts,
        "scores": [score for score, start in ranked if start in starts],
        "stopped_by_budget": len(ranked) > len(starts),
        "fallback_to_stage2": len(starts) == 0,
    }


def fill_unavailable(
    probabilities: torch.Tensor,
    availability: torch.Tensor,
    updates: Sequence[tuple[Sequence[int], torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average new windows only on voxels HighRes did not already cover.

    Voxels outside every new window, and voxels already covered, are copied
    unchanged. Overlapping new windows are averaged with equal window weight.
    """

    existing = probabilities.detach().clone()
    if existing.ndim != 4:
        raise ValueError("probabilities must be [class, D, H, W]")
    covered = _spatial(availability).clone()
    if not updates:
        return existing, covered
    accumulated = torch.zeros_like(existing)
    counts = torch.zeros(existing.shape[1:], dtype=existing.dtype)
    for start, patch in updates:
        patch = patch.detach()
        if patch.ndim != 4 or patch.shape[0] != existing.shape[0]:
            raise ValueError("a HighRes window must be [class, depth, height, width]")
        if not torch.allclose(patch.sum(0), torch.ones_like(patch.sum(0)), atol=1e-4):
            raise ValueError("a HighRes window is not a probability distribution")
        placed = _placement(tuple(int(value) for value in start), tuple(patch.shape[1:]), tuple(existing.shape[1:]))
        if placed is None:
            continue
        destination, source = placed
        missing = covered[destination] <= 0
        accumulated[(slice(None), *destination)] += patch[(slice(None), *source)] * missing.to(patch.dtype)
        counts[destination] += missing.to(counts.dtype)
    fresh = counts > 0
    averaged = accumulated / counts.clamp_min(1)
    merged = torch.where(fresh.unsqueeze(0), averaged, existing)
    merged = merged / merged.sum(0, keepdim=True).clamp_min(1e-8)
    availability_out = torch.where(fresh, torch.ones_like(covered), covered)
    return merged, availability_out


def _placement(start, patch_shape, volume_shape):
    destination, source = [], []
    for origin, length, limit in zip(start, patch_shape, volume_shape):
        begin = max(0, origin)
        end = min(limit, origin + length)
        if end <= begin:
            return None
        destination.append(slice(begin, end))
        source.append(slice(begin - origin, begin - origin + (end - begin)))
    return tuple(destination), tuple(source)


def merge_highres_compact(output: dict, starts: Sequence[Sequence[int]], windows: Sequence[Mapping]) -> dict:
    """Fill previously uncovered HighRes voxels and refresh probabilities.

    Already covered probabilities and features are not changed. Uncertainty,
    confidence, and margin are recomputed from the merged probabilities.
    """

    from ..experts.base import uncertainty

    probabilities = output["probabilities"].detach().clone()
    features = output["features"].detach().clone()
    availability = output["extra"]["availability"].detach().clone()
    if not starts:
        return output
    accumulated_probabilities = torch.zeros_like(probabilities)
    accumulated_features = torch.zeros_like(features)
    counts = torch.zeros_like(availability)
    for start, window in zip(starts, windows):
        sl = patch_slices(tuple(int(value) for value in start), tuple(window["probabilities"].shape[2:]))
        missing = (availability[(..., *sl)] <= 0).to(probabilities.dtype)
        accumulated_probabilities[(..., *sl)] += window["probabilities"].detach().cpu() * missing
        accumulated_features[(..., *sl)] += window["features"].detach().cpu() * missing.to(features.dtype)
        counts[(..., *sl)] += missing
    fresh = counts > 0
    probabilities = torch.where(fresh, accumulated_probabilities / counts.clamp_min(1), probabilities)
    features = torch.where(fresh, accumulated_features / counts.clamp_min(1), features)
    availability = torch.where(fresh, torch.ones_like(availability), availability)
    stats = uncertainty(probabilities)
    extra = dict(output["extra"])
    extra.update(availability=availability, margin=stats["margin"], entropy=stats["entropy"], top_probability=stats["top_probability"])
    return {
        **output,
        "probabilities": probabilities,
        "features": features,
        "uncertainty": stats["entropy"],
        "confidence": stats["top_probability"],
        "segmentation": probabilities.argmax(1),
        "logits": probabilities.clamp_min(1e-8).log(),
        "extra": extra,
    }


def proposal_continuation(
    proposal: torch.Tensor,
    highres_availability: torch.Tensor,
    *,
    patch: Sequence[int] = (32, 32, 32),
    overlap: float = 0.5,
    max_actions: int = 16,
) -> list[tuple[int, ...]]:
    """Cost-matched control: extra windows ranked by the existing proposal, not uncertainty."""

    score_map = _spatial(proposal).float()
    availability = _spatial(highres_availability)
    patch = tuple(int(size) for size in patch)
    ranked = []
    for start in window_starts(tuple(score_map.shape), patch, overlap):
        sl = patch_slices(start, patch)
        if bool((availability[sl] > 0).all()):
            continue
        score = float(score_map[sl].mean())
        if score > 0:
            ranked.append((score, tuple(int(value) for value in start)))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return select_actions(ranked, max_actions=max_actions)
