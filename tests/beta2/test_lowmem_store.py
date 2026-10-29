import numpy as np
import torch

from brats_debate.config import load_config
from brats_debate.controller.lowmem_store import (
    expert_complete,
    load_depth_range,
    proposal_from_experts,
    save_expert_compact,
    stream_stored_gates,
)
from brats_debate.controller.spatial_gate import PointwiseMatchedGate, SpatialContextGate
from brats_debate.controller.spatial_stream import build_gate_inputs, stream_gates
from pathlib import Path


def _cfg():
    return load_config(Path(__file__).resolve().parents[2] / "configs" / "brats_controller.yaml")


def _outputs(shape=(6, 8, 8)):
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


def test_depth_shards_round_trip_and_do_not_overwrite(tmp_path):
    image, outputs = _outputs()
    for name, output in outputs.items():
        save_expert_compact(tmp_path / name, output)
        assert expert_complete(tmp_path / name)
        restored = load_depth_range(tmp_path / name / "probabilities", 1, 4)
        assert torch.allclose(torch.from_numpy(restored), output["probabilities"][..., 1:4, :, :])
    proposal = proposal_from_experts(tmp_path)
    votes = torch.stack([output["probabilities"].argmax(1) > 0 for output in outputs.values() if True][:3])
    assert proposal.shape[1] == 1
    assert proposal.shape[-3:] == image.shape[-3:]
    del votes


def test_stored_slabs_match_the_in_memory_gate(tmp_path):
    cfg = _cfg()
    image, outputs = _outputs()
    for name, output in outputs.items():
        save_expert_compact(tmp_path / name, output)
    features, probabilities, availability = build_gate_inputs(image, outputs)
    pointwise = PointwiseMatchedGate(features.shape[1], width=3, experts=4)
    spatial = SpatialContextGate(features.shape[1], width=2, experts=4)
    with torch.no_grad():
        for module in (pointwise, spatial):
            for parameter in module.parameters():
                parameter.normal_(0, 0.2)
    modules = {"pointwise": pointwise, "spatial": spatial}
    memory_labels, _stats = stream_gates(modules, image, outputs, cfg, torch.device("cpu"), slab_depth=2, halo=1)
    stored_labels, stats = stream_stored_gates(
        modules, image, tmp_path, cfg, slab_depth=2, halo=1, capture_flat=[0, 100, 200]
    )
    coords = np.unravel_index([0, 100, 200], image.shape[-3:])
    for name in modules:
        assert torch.equal(stored_labels[name], memory_labels[name])
        assert stats["max_probability_sum_error"][name] < 1e-4
        captured = stats["captured_probabilities"][name]
        assert captured.shape == (3, 4)
        for index in range(3):
            assert int(captured[index].argmax()) == int(
                stored_labels[name][coords[0][index], coords[1][index], coords[2][index]]
            )
