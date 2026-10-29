import torch

from brats_debate.controller.gating_network import GatingNetwork
from brats_debate.controller.spatial_gate import (
    PointwiseMatchedGate,
    SpatialContextGate,
    parameter_count,
)


def _inputs():
    features = torch.zeros(1, 4, 3, 3, 3)
    probabilities = torch.zeros(1, 4, 4, 3, 3, 3)
    probabilities[:, :, 0] = 1
    availability = torch.ones(1, 4, 3, 3, 3)
    availability[:, 3] = 0
    return features, probabilities, availability


def test_pointwise_gate_ignores_a_neighbor_that_the_spatial_gate_uses():
    features, probabilities, availability = _inputs()
    spatial = SpatialContextGate(4, width=1, experts=4)
    pointwise = PointwiseMatchedGate(4, width=1, experts=4)
    with torch.no_grad():
        for module in (spatial.reduce, spatial.project, pointwise.network[0], pointwise.network[2]):
            module.weight.zero_()
            module.bias.zero_()
        spatial.reduce.weight[0, 0] = 1
        spatial.context.weight.zero_()
        spatial.context.bias.zero_()
        spatial.context.weight[0, 0, 0, 0, 0] = 1
        spatial.project.weight[0, 0] = 1
        pointwise.network[0].weight[0, 0] = 1
        pointwise.network[2].weight[0, 0] = 1
    center = spatial(features, probabilities, availability)["weights"][0, 0, 1, 1, 1]
    features[0, 0, 0, 0, 0] = 4
    moved = spatial(features, probabilities, availability)["weights"][0, 0, 1, 1, 1]
    pointwise_before = pointwise(torch.zeros_like(features), probabilities, availability)["weights"][0, 0, 1, 1, 1]
    pointwise_after = pointwise(features, probabilities, availability)["weights"][0, 0, 1, 1, 1]
    assert not torch.allclose(center, moved)
    assert torch.allclose(pointwise_before, pointwise_after)


def test_both_gates_mask_unavailable_experts_and_normalize():
    features, probabilities, availability = _inputs()
    features.normal_()
    for gate in (PointwiseMatchedGate(4, width=11), SpatialContextGate(4, width=8)):
        output = gate(features, probabilities, availability)
        assert torch.allclose(output["weights"].sum(1), torch.ones(1, 1, 3, 3, 3), atol=1e-5)
        assert float(output["weights"][:, 3].detach().max()) < 1e-3
        assert torch.allclose(output["probabilities"].sum(1), torch.ones(1, 1, 3, 3, 3), atol=1e-4)


def test_spatial_capacity_stays_near_the_matched_pointwise_gate():
    spatial = parameter_count(SpatialContextGate(65, width=8, experts=4))
    pointwise = parameter_count(PointwiseMatchedGate(65, width=11, experts=4))
    beta1 = parameter_count(GatingNetwork(65, width=8, experts=4))
    assert beta1 < spatial
    assert abs(spatial - pointwise) / spatial < 0.05
