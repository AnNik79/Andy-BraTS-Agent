import pytest
import torch
from brats_debate.experts import build_expert
from brats_debate.experts.base import pack_output, uncertainty
from brats_debate.debate.disagreement import analyze_disagreement
from brats_debate.debate.features import controller_features, feature_channels, stack_evidence
from brats_debate.controller.gating_network import GatingNetwork
from brats_debate.inference.patches import sliding_predict
from brats_debate.training.train_expert import expert_loss


@pytest.mark.parametrize("name", ["cnn", "transformer", "boundary", "highres"])
def test_expert_shape_and_trainable(name, cfg):
    model = build_expert(name, cfg)
    output = model(torch.randn(1, 4, 8, 8, 8))
    assert output["probabilities"].shape == (1, 4, 8, 8, 8)
    assert output["features"].shape == (1, 2, 8, 8, 8)
    assert output["uncertainty"].shape == (1, 1, 8, 8, 8)
    torch.testing.assert_close(output["probabilities"].sum(1), torch.ones(1, 8, 8, 8))
    expert_loss(output, torch.randint(4, (1, 8, 8, 8))).backward()
    assert model.head.weight.grad is not None
    if name == "boundary":
        assert model.boundary_head.weight.grad.abs().sum() > 0
        assert model.distance_head.weight.grad.abs().sum() > 0


def test_uncertainty_and_disagreement():
    uniform = torch.ones(1, 4, 2, 2, 2) / 4
    stats = uncertainty(uniform)
    torch.testing.assert_close(stats["entropy"], torch.ones_like(stats["entropy"]))
    assert stats["margin"].max() == 0
    a = pack_output(uniform.log(), uniform)
    d = analyze_disagreement({"cnn": a, "transformer": a})
    assert d["map"].abs().max() < 1e-6
    p = torch.zeros_like(uniform)
    p[:, 1] = 1
    q = torch.zeros_like(uniform)
    q[:, 2] = 1
    d = analyze_disagreement({"cnn": pack_output(p.clamp_min(1e-8).log(), p),
                              "transformer": pack_output(q.clamp_min(1e-8).log(), q)})
    assert d["map"].min() > .7
    assert d["pairwise"]["cnn__transformer"]["disagreement"].min() == 1


def test_controller_mask_weights_and_gradient(cfg):
    image = torch.randn(1, 4, 8, 8, 8)
    outputs = {name: build_expert(name, cfg)(image) for name in ("cnn", "transformer", "boundary", "highres")}
    outputs["highres"]["extra"]["availability"] = torch.zeros_like(outputs["highres"]["confidence"])
    debate = analyze_disagreement(outputs)
    features = controller_features(image, outputs, debate).detach()
    assert features.shape[1] == feature_channels(cfg)
    p, availability = stack_evidence(outputs)
    gate = GatingNetwork(features.shape[1], 4)
    result = gate(features, p.detach(), availability)
    torch.testing.assert_close(result["weights"].sum(1), torch.ones_like(result["weights"][:, 0]))
    assert result["weights"][:, 3].max() == 0
    result["probabilities"][:, 1].mean().backward()
    assert gate.network[0].weight.grad.abs().sum() > 0


def test_patch_reconstruction_and_abstention():
    class IdentityExpert:
        def predict(self, image, mc_samples):
            return pack_output(image, image[:, :2])
    image = torch.randn(1, 4, 11, 9, 7)
    output = sliding_predict(IdentityExpert(), image, (8, 8, 8), "cpu", .5)
    torch.testing.assert_close(output["probabilities"], image.softmax(1), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(output["features"], image[:, :2])
    assert output["extra"]["availability"].min() == 1
    proposal = torch.zeros(1, 1, 11, 9, 7)
    proposal[:, :, 0] = 1
    limited = sliding_predict(IdentityExpert(), image, (4, 4, 4), "cpu", .5, proposal=proposal, max_patches=1)
    mask = limited["extra"]["availability"] > 0
    assert 0 < mask.sum() < mask.numel()
    torch.testing.assert_close(limited["probabilities"].sum(1), torch.ones(1, 11, 9, 7))


def test_mc_dropout_restores_mode(cfg):
    model = build_expert("cnn", cfg).train()
    output = model.predict(torch.randn(1, 4, 8, 8, 8), mc_samples=3)
    assert model.training
    assert output["extra"]["mutual_information"].min() >= 0
    assert output["extra"]["mutual_information"].max() > 0
