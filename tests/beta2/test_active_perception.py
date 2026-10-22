import inspect

import torch

from brats_debate.perception.active import (
    ACTIVE_PROTOCOL,
    eligible_voxels,
    fill_unavailable,
    plan_additional_highres,
    proposal_continuation,
    rank_windows,
    select_actions,
)


def _experts(shape=(8, 8, 8)):
    probabilities = torch.zeros(4, 4, *shape)
    probabilities[:, 0] = 1
    probabilities[0, :, 2:5, 2:5, 2:5] = 0
    probabilities[0, 1, 2:5, 2:5, 2:5] = 1
    availability = torch.ones(4, *shape)
    availability[3] = 0
    highres = torch.zeros(shape)
    entropy = torch.zeros(shape)
    entropy[2:5, 2:5, 2:5] = 0.8
    return highres, probabilities, availability, entropy


def test_trigger_requires_missing_highres_and_uncertainty_without_labels():
    signature = inspect.signature(plan_additional_highres)
    assert not any(name in signature.parameters for name in ("target", "label", "ground_truth", "correct"))
    highres, probabilities, availability, entropy = _experts()
    eligible = eligible_voxels(highres, probabilities, availability, entropy)
    assert bool(eligible[3, 3, 3])
    assert not bool(eligible[0, 0, 0])
    highres_known = highres.clone()
    highres_known[2:5, 2:5, 2:5] = 1
    assert not bool(eligible_voxels(highres_known, probabilities, availability, entropy)[3, 3, 3])
    quiet = entropy.clone()
    quiet[:] = 0
    agreed = probabilities.clone()
    agreed[0, :, 2:5, 2:5, 2:5] = 0
    agreed[0, 0, 2:5, 2:5, 2:5] = 1
    assert not bool(eligible_voxels(highres, agreed, availability, quiet).any())


def test_region_selection_respects_budget_fraction_and_order():
    eligible = torch.zeros(32, 32, 32, dtype=torch.bool)
    eligible[:8, :8, :8] = True
    eligible[16:20, 16:20, 16:20] = True
    ranked = rank_windows(eligible, patch=(16, 16, 16), overlap=0.5, min_eligible_fraction=0.02)
    assert ranked[0][1] == (0, 0, 0)
    assert select_actions(ranked, max_actions=1) == [(0, 0, 0)]
    weak = torch.zeros(32, 32, 32, dtype=torch.bool)
    weak[0, 0, 0] = True
    assert rank_windows(weak, patch=(16, 16, 16), overlap=0.5, min_eligible_fraction=0.02) == []
    repeated = rank_windows(eligible, patch=(16, 16, 16), overlap=0.5, min_eligible_fraction=0.02)
    assert repeated == ranked


def test_fill_preserves_covered_voxels_and_averages_new_overlap():
    probabilities = torch.zeros(4, 8, 8, 8)
    probabilities[0] = 1
    availability = torch.zeros(8, 8, 8)
    availability[:4] = 1
    first = torch.zeros(4, 4, 8, 8)
    first[1] = 1
    second = torch.zeros(4, 4, 8, 8)
    second[3] = 1
    merged, covered = fill_unavailable(
        probabilities,
        availability,
        [((2, 0, 0), first), ((4, 0, 0), second)],
    )
    assert torch.equal(merged[:, :2], probabilities[:, :2])
    assert torch.equal(covered[:2], availability[:2])
    overlap = merged[:, 4:6]
    assert torch.allclose(overlap[1], torch.full_like(overlap[1], 0.5))
    assert torch.allclose(overlap[3], torch.full_like(overlap[3], 0.5))
    assert torch.allclose(merged[:, 6:], second[:, 2:])
    assert torch.allclose(merged.sum(0), torch.ones(8, 8, 8), atol=1e-5)
    assert torch.equal(covered[4:], torch.ones(4, 8, 8))


def test_no_action_matches_the_stage2_highres_output():
    probabilities = torch.rand(4, 6, 6, 6)
    probabilities = probabilities / probabilities.sum(0, keepdim=True)
    availability = torch.randint(0, 2, (6, 6, 6)).float()
    merged, covered = fill_unavailable(probabilities, availability, [])
    assert torch.equal(merged, probabilities)
    assert torch.equal(covered, availability)
    highres, expert_probabilities, expert_availability, entropy = _experts((16, 16, 16))
    entropy[:] = 0
    expert_probabilities[0] = expert_probabilities[1]
    plan = plan_additional_highres(
        highres,
        expert_probabilities,
        expert_availability,
        entropy,
        patch=(8, 8, 8),
        max_actions=ACTIVE_PROTOCOL["max_actions"],
    )
    assert plan["fallback_to_stage2"] is True
    assert plan["starts"] == []


def test_a_previously_zero_weight_is_not_the_anchor_after_new_evidence():
    from brats_debate.controller.adaptive_gate import ResidualAdaptiveGate, feature_names

    probabilities = torch.zeros(1, 4, 4)
    probabilities[0, 0, 0] = 1
    probabilities[0, 1, 0] = 1
    probabilities[0, 2, 0] = 1
    probabilities[0, 3, 3] = 1
    availability = torch.ones(1, 4)
    features = torch.zeros(1, len(feature_names()))
    gate = ResidualAdaptiveGate(features.shape[1])
    locked = gate(features, probabilities, availability, torch.tensor([[0.5, 0.5, 0.0, 0.0]]))
    recomputed = gate(features, probabilities, availability, torch.tensor([[0.25, 0.25, 0.25, 0.25]]))
    assert locked["weights"][0, 3].item() < 1e-4
    assert recomputed["weights"][0, 3].item() > 0.2


def test_merged_highres_keeps_covered_features_and_refreshes_new_voxels():
    from brats_debate.perception.active import merge_highres_compact

    probabilities = torch.zeros(1, 2, 4, 4, 4)
    probabilities[:, 0] = 1
    features = torch.ones(1, 1, 4, 4, 4)
    availability = torch.zeros(1, 1, 4, 4, 4)
    availability[:, :, :2] = 1
    window_probabilities = torch.zeros(1, 2, 2, 4, 4)
    window_probabilities[:, 1] = 1
    window_features = torch.full((1, 1, 2, 4, 4), 4)
    merged = merge_highres_compact(
        {
            "probabilities": probabilities,
            "features": features,
            "extra": {"availability": availability},
        },
        [(1, 0, 0)],
        [{"probabilities": window_probabilities, "features": window_features}],
    )
    assert torch.equal(merged["probabilities"][:, :, :1], probabilities[:, :, :1])
    assert torch.equal(merged["features"][:, :, :1], features[:, :, :1])
    assert torch.allclose(merged["probabilities"][:, 1, 2:3], torch.ones_like(merged["probabilities"][:, 1, 2:3]))
    assert torch.equal(merged["probabilities"][:, :, 3:], probabilities[:, :, 3:])
    assert float(merged["extra"]["availability"][:, :, 2:3].min()) == 1


def test_budget_zero_stops_and_proposal_control_ignores_uncertainty():
    eligible = torch.ones(16, 16, 16, dtype=torch.bool)
    ranked = rank_windows(eligible, patch=(8, 8, 8), overlap=0.5, min_eligible_fraction=0.02)
    assert select_actions(ranked, max_actions=0) == []
    proposal = torch.zeros(16, 16, 16)
    proposal[8:] = 1
    availability = torch.zeros(16, 16, 16)
    availability[:8] = 1
    chosen = proposal_continuation(proposal, availability, patch=(8, 8, 8), overlap=0.5, max_actions=2)
    assert chosen[0][0] == 8
    assert (0, 0, 0) not in chosen
    assert len(chosen) == 2


def test_proposal_extension_selects_uncomputed_windows_and_stops_at_the_budget():
    import inspect

    from brats_debate.perception.active import executed_highres_starts, plan_proposal_extension

    signature = inspect.signature(plan_proposal_extension)
    assert not any(name in signature.parameters for name in ("target", "label", "ground_truth", "correct"))
    proposal = torch.zeros(32, 32, 32)
    proposal[:16] = torch.linspace(1, 0.1, 16)[:, None, None]
    plan = plan_proposal_extension(proposal, (32, 32, 32), patch=(16, 16, 16), overlap=0.5, already_run=1, max_actions=2)
    executed = executed_highres_starts(proposal, (32, 32, 32), patch=(16, 16, 16), overlap=0.5, max_patches=1)
    assert len(plan["starts"]) == 2
    assert set(plan["starts"]).isdisjoint(executed)
    assert plan["fallback_to_stage2"] is False
    quiet = torch.zeros(16, 16, 16)
    assert plan_proposal_extension(quiet, (16, 16, 16), patch=(16, 16, 16), already_run=1, max_actions=16)["fallback_to_stage2"] is True
    capped = plan_proposal_extension(proposal, (32, 32, 32), patch=(16, 16, 16), overlap=0.5, already_run=1, max_actions=16)
    assert len(capped["starts"]) <= 16
    from brats_debate.perception.active import far_proposal_starts

    near = plan["starts"]
    far = far_proposal_starts(proposal, (32, 32, 32), patch=(16, 16, 16), overlap=0.5, already_run=1, count=len(near))
    assert len(far) == len(near)
    assert set(far).isdisjoint(near)
    assert set(far).isdisjoint(executed)
