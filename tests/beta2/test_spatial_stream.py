import torch

from brats_debate.config import load_config
from brats_debate.controller.gating_network import GatingNetwork
from brats_debate.controller.spatial_gate import PointwiseMatchedGate, SpatialContextGate
from brats_debate.controller.spatial_stream import (
    build_gate_inputs,
    interior_prediction,
    real_window,
    stream_gates,
    window_fits,
)

ROOT_SHAPE = (6, 8, 8)


def _cfg():
    from pathlib import Path

    return load_config(Path(__file__).resolve().parents[2] / "configs" / "brats_controller.yaml")


def _outputs(shape=ROOT_SHAPE):
    names = ("cnn", "transformer", "boundary", "highres")
    outputs = {}
    for index, name in enumerate(names):
        probabilities = torch.zeros(1, 4, *shape)
        probabilities[:, index % 4] = 1
        availability = torch.ones(1, 1, *shape)
        if name == "highres":
            availability[:, :, :1] = 0
            probabilities[:, :, :1] = 0.25
        extra = {"margin": torch.full((1, 1, *shape), 0.2), "availability": availability}
        if name == "boundary":
            extra["boundary_probability"] = torch.full((1, 1, *shape), 0.3)
            extra["distance_to_boundary"] = torch.arange(shape[0]).float().view(1, 1, -1, 1, 1).expand(1, 1, *shape)
        outputs[name] = {
            "probabilities": probabilities,
            "uncertainty": torch.full((1, 1, *shape), 0.1 * (index + 1)),
            "confidence": probabilities.amax(1, keepdim=True),
            "features": torch.full((1, 4, *shape), float(index + 1)),
            "extra": extra,
        }
    image = torch.zeros(1, 4, *shape)
    image[:, :, :, 2:5, 2:5] = 1
    return image, outputs


def test_full_halo_window_matches_the_volume_features():
    image, outputs = _outputs()
    full_features, full_probabilities, full_availability = build_gate_inputs(image, outputs)
    origin = (2, 2, 2)
    features, probabilities, availability = real_window(image, outputs, origin, interior=2, halo=1)
    sl = tuple(slice(origin[axis] - 1, origin[axis] + 3) for axis in range(3))
    assert torch.allclose(features, full_features[(..., *sl)])
    assert torch.allclose(probabilities, full_probabilities[(..., *sl)])
    assert torch.allclose(availability, full_availability[(..., *sl)])
    assert not window_fits((0, 2, 2), image.shape[-3:], 2, 1)


def test_halo_streaming_matches_a_full_spatial_forward_and_a_cut_slab_does_not():
    cfg = _cfg()
    image, outputs = _outputs()
    features, probabilities, availability = build_gate_inputs(image, outputs)
    spatial = SpatialContextGate(features.shape[1], width=2, experts=4)
    pointwise = PointwiseMatchedGate(features.shape[1], width=3, experts=4)
    frozen = GatingNetwork(features.shape[1], width=2, experts=4)
    with torch.no_grad():
        for module in (spatial, pointwise, frozen):
            for parameter in module.parameters():
                parameter.normal_(0, 0.2)
        spatial.context.weight.zero_()
        spatial.context.weight[:, :, 0, 1, 1] = 1
    full = {
        "spatial": interior_prediction(spatial, features, probabilities, availability, halo=0),
        "pointwise": interior_prediction(pointwise, features, probabilities, availability, halo=0),
        "frozen": interior_prediction(frozen, features, probabilities, availability, halo=0),
    }
    labels, _stats = stream_gates(
        {"spatial": spatial, "pointwise": pointwise, "frozen": frozen},
        image,
        outputs,
        cfg,
        torch.device("cpu"),
        slab_depth=2,
        halo=1,
    )
    for name in full:
        assert torch.equal(labels[name], full[name]["probabilities"].argmax(1)[0])

    broken, _stats = stream_gates(
        {"spatial": spatial},
        image,
        outputs,
        cfg,
        torch.device("cpu"),
        slab_depth=2,
        halo=0,
    )
    assert not torch.equal(broken["spatial"], labels["spatial"])
