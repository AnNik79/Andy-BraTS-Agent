import pytest
import torch

from brats_debate.controller.gating_network import GatingNetwork
from brats_debate.experts.base import pack_output
from brats_debate.uncertainty.disagreement import (
    analyze_uncertainty,
    entropy_map,
)
from brats_debate.uncertainty.expert_trust import analyze_expert_trust
from brats_debate.uncertainty.probability import (
    masked_mean_probability,
    stack_expert_probabilities,
    validate_probability_tensor,
)


EXPERTS = ("cnn", "transformer", "boundary", "highres")


def _output(probabilities, availability=None):
    features = torch.zeros(
        probabilities.shape[0], 1, *probabilities.shape[2:]
    )
    extra = {} if availability is None else {"availability": availability}
    return pack_output(
        probabilities.clamp_min(1e-8).log(),
        features,
        extra=extra,
        probabilities=probabilities,
    )


def _one_hot(class_id, spatial=(2, 2, 2), classes=4):
    value = torch.zeros(1, classes, *spatial)
    value[:, class_id] = 1
    return value


def test_probability_validation_normalization_and_zero_edges():
    p = _one_hot(0)
    validated = validate_probability_tensor(p)
    torch.testing.assert_close(validated, p)
    repaired = validate_probability_tensor(p * 2, normalize=True)
    torch.testing.assert_close(repaired, p)
    with pytest.raises(ValueError, match="sum to one"):
        validate_probability_tensor(p * 2)
    with pytest.raises(ValueError, match="nonnegative"):
        validate_probability_tensor(p - 0.1)
    with pytest.raises(ValueError, match="finite"):
        validate_probability_tensor(torch.full_like(p, float("nan")))
    with pytest.raises(ValueError, match="positive class mass"):
        validate_probability_tensor(torch.zeros_like(p), normalize=True)


def test_probability_alignment_missing_experts_and_availability():
    p = _one_hot(0)
    q = _one_hot(1)
    batch = stack_expert_probabilities(
        {"cnn": p, "boundary": q}, class_order=(0, 1, 2, 3)
    )
    assert batch.names == ("cnn", "boundary")
    assert batch.probabilities.shape == (1, 2, 4, 2, 2, 2)
    torch.testing.assert_close(
        masked_mean_probability(batch), (p + q) / 2
    )
    with pytest.raises(ValueError, match="missing required experts"):
        stack_expert_probabilities({"cnn": p}, require_all=True)
    with pytest.raises(ValueError, match="not aligned"):
        stack_expert_probabilities(
            {"cnn": p, "transformer": _one_hot(1, spatial=(2, 2, 3))}
        )
    zero = torch.zeros(1, 1, 2, 2, 2)
    unavailable = stack_expert_probabilities(
        {"cnn": p}, availability={"cnn": zero}
    )
    with pytest.raises(ValueError, match="at least one expert"):
        masked_mean_probability(unavailable)


def test_entropy_and_js_identity_distinguish_uncertainty_from_disagreement():
    uniform = torch.full((1, 4, 2, 2, 2), 0.25)
    assert entropy_map(_one_hot(0)).eq(0).all()
    outputs = {name: _output(uniform) for name in EXPERTS}
    analysis = analyze_uncertainty(outputs)
    assert analysis["equal_weight_four_expert_formula"] is True
    torch.testing.assert_close(
        entropy_map(uniform), torch.ones(1, 1, 2, 2, 2)
    )
    torch.testing.assert_close(
        analysis["mean_individual_entropy"],
        torch.ones_like(analysis["mean_individual_entropy"]),
    )
    assert analysis["js_divergence"].abs().max() < 1e-6
    assert analysis["vote_disagreement"].abs().max() < 1e-6

    split = {
        EXPERTS[0]: _output(_one_hot(0)),
        EXPERTS[1]: _output(_one_hot(0)),
        EXPERTS[2]: _output(_one_hot(1)),
        EXPERTS[3]: _output(_one_hot(1)),
    }
    disagreement = analyze_uncertainty(split)
    torch.testing.assert_close(
        disagreement["js_divergence"],
        torch.full_like(disagreement["js_divergence"], 0.5),
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        disagreement["js_from_weighted_kl"],
        disagreement["js_divergence"],
        atol=1e-6,
        rtol=0,
    )
    assert disagreement["js_identity_residual"].max() < 1e-6
    torch.testing.assert_close(
        disagreement["production"]["js_divergence"],
        disagreement["js_divergence"],
        atol=1e-6,
        rtol=0,
    )
    torch.testing.assert_close(
        disagreement["vote_disagreement"],
        torch.full_like(disagreement["vote_disagreement"], 0.5),
    )


def test_unavailable_expert_and_region_boundary_summaries():
    # The public probability API accepts [B,D,H,W] as well as [B,1,D,H,W].
    availability = torch.ones(1, 2, 2, 2)
    availability[:, 0] = 0
    outputs = {
        "cnn": _output(_one_hot(0)),
        "transformer": _output(_one_hot(1), availability),
    }
    region = torch.zeros(2, 2, 2, dtype=torch.bool)
    region[0] = True
    analysis = analyze_uncertainty(
        outputs, region_masks={"synthetic_region": region}, boundary_mask=~region
    )
    assert analysis["equal_weight_four_expert_formula"] is False
    assert analysis["available_experts"][:, :, 0].eq(1).all()
    assert analysis["available_experts"][:, :, 1].eq(2).all()
    assert analysis["summaries"]["synthetic_region"]["voxel_count"] == 4
    assert analysis["summaries"]["boundary"]["voxel_count"] == 4
    assert set(analysis["pairwise"]) == {"cnn__transformer"}


def test_single_available_expert_has_zero_disagreement():
    analysis = analyze_uncertainty({"cnn": _output(_one_hot(2))})
    assert analysis["expert_names"] == ("cnn",)
    assert analysis["js_divergence"].eq(0).all()
    assert analysis["vote_disagreement"].eq(0).all()
    assert analysis["pairwise"] == {}


def test_gate_weights_remain_a_masked_simplex():
    features = torch.randn(1, 3, 2, 2, 2)
    probabilities = torch.stack([_one_hot(i) for i in range(4)], dim=1)
    availability = torch.ones(1, 4, 2, 2, 2)
    availability[:, 3] = 0
    result = GatingNetwork(3, width=2)(features, probabilities, availability)
    torch.testing.assert_close(
        result["weights"].sum(1), torch.ones(1, 2, 2, 2)
    )
    assert result["weights"][:, 3].eq(0).all()


def test_expert_trust_oracle_fix_harm_and_stratification():
    target = torch.tensor([[[[0, 1, 2, 3]]]])
    p = torch.zeros(1, 4, 4, 1, 1, 4)

    # Voxel 0: every expert is correct.
    p[:, :, 0, 0, 0, 0] = 1
    # Voxel 1: only CNN is correct; fusion follows the incorrect experts.
    p[:, 0, 1, 0, 0, 1] = 1
    p[:, 1:, 0, 0, 0, 1] = 1
    # Voxel 2: no hard expert is correct, but their shared second choice is.
    wrong = (0, 1, 3, 0)
    for index, class_id in enumerate(wrong):
        p[:, index, class_id, 0, 0, 2] = 0.6
        p[:, index, 2, 0, 0, 2] = 0.4
    # Voxel 3: CNN is correct and fusion recovers the partial expert failure.
    p[:, 0, 3, 0, 0, 3] = 1
    p[:, 1:, 0, 0, 0, 3] = 1

    expert_probabilities = {
        name: p[:, index] for index, name in enumerate(EXPERTS)
    }
    weights = torch.tensor(
        [
            [
                [[[0.25, 0.20, 0.25, 0.60]]],
                [[[0.25, 0.60, 0.25, 0.20]]],
                [[[0.25, 0.10, 0.25, 0.10]]],
                [[[0.25, 0.10, 0.25, 0.10]]],
            ]
        ]
    )
    boundary = torch.tensor([[[[False, True, True, False]]]])
    disagreement = torch.tensor([[[[[0.0, 0.5, 0.8, 0.2]]]]])
    trust = analyze_expert_trust(
        expert_probabilities,
        weights,
        target,
        boundary_mask=boundary,
        disagreement=disagreement,
    )
    summary = trust["summary"]
    assert summary["controller_accuracy"] == pytest.approx(0.75)
    assert summary["hard_expert_oracle_coverage"] == pytest.approx(0.75)
    assert summary["controller_vs_hard_oracle_gap"] == pytest.approx(0.0)
    assert summary["all_experts_wrong_fraction"] == pytest.approx(0.25)
    assert summary["maximum_weight_expert_correct_fraction"] == pytest.approx(0.5)
    assert summary["mean_correct_expert_weight_mass"] == pytest.approx(0.45)
    assert summary["fusion_correct_without_hard_expert_fraction"] == pytest.approx(
        0.25
    )
    assert summary[
        "fusion_harm_when_correct_expert_available_fraction"
    ] == pytest.approx(0.25)
    assert trust["stratified"]["boundary"]["voxel_count"] == 2
    assert trust["stratified"]["high_disagreement"]["voxel_count"] == 2
    assert trust["stratified"]["ground_truth_ET"]["voxel_count"] == 1
    assert set(trust["weight_correctness_correlations"]) == set(EXPERTS)


def test_expert_trust_rejects_invalid_or_unavailable_weights():
    p = {name: _one_hot(0) for name in EXPERTS}
    target = torch.zeros(1, 2, 2, 2, dtype=torch.long)
    weights = torch.full((1, 4, 2, 2, 2), 0.25)
    bad = weights.clone()
    bad[:, 0] = 0.5
    with pytest.raises(ValueError, match="sum to one"):
        analyze_expert_trust(p, bad, target)

    availability = {name: torch.ones(1, 1, 2, 2, 2) for name in EXPERTS}
    availability["highres"] = torch.zeros(1, 1, 2, 2, 2)
    with pytest.raises(ValueError, match="zero controller weight"):
        analyze_expert_trust(
            p, weights, target, availability=availability
        )

    inconsistent = _one_hot(1)
    with pytest.raises(ValueError, match="must equal"):
        analyze_expert_trust(
            p, weights, target, controller_probabilities=inconsistent
        )


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS is unavailable"
)
def test_foundation_tensor_paths_run_on_mps():
    device = torch.device("mps")
    p = {name: _one_hot(0).to(device) for name in EXPERTS}
    analysis = analyze_uncertainty(
        {name: _output(value) for name, value in p.items()}
    )
    assert analysis["js_divergence"].device.type == "mps"

    weights = torch.full((1, 4, 2, 2, 2), 0.25, device=device)
    target = torch.zeros(1, 2, 2, 2, dtype=torch.long, device=device)
    trust = analyze_expert_trust(p, weights, target)
    assert trust["controller_probabilities"].device.type == "mps"
