import torch

from brats_debate.controller.spatial_gate import PointwiseMatchedGate
from brats_debate.training.geometric_loss import (
    DISTANCE_CAP_MM,
    geometric_loss,
    geometric_region_loss,
    reference_distances,
    region_probability,
)


def _ball(shape=(9, 9, 9), center=(4, 4, 4), radius=2):
    grid = torch.stack(torch.meshgrid(*(torch.arange(size) for size in shape), indexing="ij"), dim=-1).float()
    return (grid - torch.tensor(center).float()).norm(dim=-1) <= radius


def _loss(probability, reference, spacing=(1, 1, 1)):
    distance_to_set, distance_to_boundary = reference_distances(reference.numpy(), spacing)
    return geometric_region_loss(
        probability[None],
        reference.float()[None],
        torch.from_numpy(distance_to_set)[None],
        torch.from_numpy(distance_to_boundary)[None],
    )


def test_perfect_binary_prediction_has_zero_geometric_loss():
    reference = _ball()
    loss = _loss(reference.float(), reference)
    assert float(loss) < 1e-6


def test_shifted_boundary_and_false_positive_island_are_penalized():
    reference = _ball()
    shifted = _ball(center=(5, 4, 4))
    near = reference.float().clone()
    near[4, 4, 7] = 1
    far = reference.float().clone()
    far[0, 0, 0] = 1
    perfect = _loss(reference.float(), reference)
    assert float(_loss(shifted.float(), reference)) > float(perfect)
    assert float(_loss(far, reference)) > float(_loss(near, reference))


def test_missing_extension_and_interior_false_negative_are_penalized():
    reference = _ball(radius=3)
    missing_side = reference.clone()
    missing_side[:, :, 5:] = False
    prediction = missing_side.float()
    center = reference.clone()
    center[4, 4, 4] = False
    assert float(_loss(prediction, reference)) > 0
    assert float(_loss(center.float(), reference)) > 0
    empty = torch.zeros_like(reference, dtype=torch.float32)
    assert float(_loss(empty, reference)) > float(_loss(reference.float(), reference))


def test_empty_ground_truth_and_one_voxel_surface_behavior():
    shape = (7, 7, 7)
    empty_reference = torch.zeros(shape, dtype=torch.bool)
    zero = torch.zeros(shape)
    blob = torch.zeros(shape)
    blob[1:3, 1:3, 1:3] = 1
    assert float(_loss(zero, empty_reference)) < 1e-6
    assert float(_loss(blob, empty_reference)) > 0
    single = torch.zeros(shape, dtype=torch.bool)
    single[3, 3, 3] = True
    # The single voxel is the surface, so its distance to the surface is zero.
    # The second term therefore does not charge a miss that lands exactly on that voxel.
    missed = _loss(zero, single)
    assert float(missed) < 1e-6


def test_nonunit_spacing_changes_the_physical_penalty_and_cap_is_finite():
    reference = _ball(shape=(11, 9, 9), center=(5, 4, 4), radius=1)
    prediction = reference.float().clone()
    prediction[0, 4, 4] = 1
    unit = _loss(prediction, reference, spacing=(1, 1, 1))
    stretched = _loss(prediction, reference, spacing=(2, 1, 1))
    assert float(stretched) > float(unit)
    distance_to_set, distance_to_boundary = reference_distances(reference.numpy(), (1, 1, 1))
    assert float(distance_to_set.max()) <= DISTANCE_CAP_MM
    assert float(distance_to_boundary.max()) <= DISTANCE_CAP_MM


def test_full_volume_distance_is_not_recomputed_on_a_patch_edge():
    mask = torch.zeros(12, 8, 8, dtype=torch.bool)
    mask[2:10, 2:6, 2:6] = True
    full_set, full_boundary = reference_distances(mask.numpy(), (1, 1, 1))
    window = mask[6:10, 2:6, 2:6]
    _local_set, local_boundary = reference_distances(window.numpy(), (1, 1, 1))
    cropped = full_boundary[6:10, 2:6, 2:6]
    assert not torch.allclose(torch.from_numpy(cropped), torch.from_numpy(local_boundary))


def test_forward_and_backward_are_finite_without_an_argmax():
    reference = _ball()
    leaf = (reference.float() + 0.1).detach().requires_grad_(True)
    loss = _loss(leaf, reference)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(leaf.grad).all()
    assert float(leaf.grad.abs().sum()) > 0


def test_nested_region_probabilities_and_three_region_loss():
    logits = torch.tensor([0.1, 0.2, 0.3, 2.0])
    probabilities = torch.softmax(logits, dim=0)[None, :, None, None, None]
    assert torch.allclose(probabilities.sum(1), torch.ones(1, 1, 1, 1), atol=1e-5)
    whole = region_probability(probabilities, (1, 2, 3))
    core = region_probability(probabilities, (1, 3))
    enhancing = region_probability(probabilities, (3,))
    assert torch.all(whole >= core) and torch.all(core >= enhancing)
    target = torch.zeros(1, 1, 1, 1, dtype=torch.long)
    target[0, 0, 0, 0] = 3
    distance = torch.zeros(1, 3, 1, 1, 1)
    loss = geometric_loss(probabilities, target, distance, distance)
    assert torch.isfinite(loss)


def test_masked_softmax_fusion_renormalizes_without_the_unavailable_expert():
    gate = PointwiseMatchedGate(2, width=4, experts=4)
    features = torch.zeros(1, 2, 1, 1, 1)
    probabilities = torch.zeros(1, 4, 4, 1, 1, 1)
    probabilities[:, 0, 1] = 1
    probabilities[:, 1:, 0] = 1
    availability = torch.tensor([1, 1, 1, 0], dtype=torch.float32).view(1, 4, 1, 1, 1)
    fused = gate(features, probabilities, availability)["probabilities"]
    weights = gate(features, probabilities, availability)["weights"]
    assert torch.allclose(fused.sum(1), torch.ones(1, 1, 1, 1), atol=1e-4)
    assert float(weights[0, 3].detach()) < 1e-3
    assert torch.isfinite(fused).all()
