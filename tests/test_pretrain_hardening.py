import json
import re
from pathlib import Path
import pytest
import torch
from brats_debate.config import (
    EXPERTS, REASONING_EXPERT, SEGMENTATION_EXPERTS, SYSTEM_EXPERTS, load_config,
)
from brats_debate.controller.gating_network import GatingNetwork
from brats_debate.debate.disagreement import analyze_disagreement
from brats_debate.debate.features import controller_features, feature_channels, stack_evidence
from brats_debate.experts import build_expert
from brats_debate.experts.base import ResidualUNet
from brats_debate.experts.boundary_expert import BoundaryExpert
from brats_debate.experts.cnn_expert import CNNExpert
from brats_debate.experts.highres_expert import HighResolutionExpert
from brats_debate.experts.transformer_expert import TransformerExpert
from brats_debate.reasoning.reasoning_engine import LLMReasoningExpert
from brats_debate.inference.patches import sliding_predict
from brats_debate.training.train_expert import expert_loss


SPLIT_KEYS = ("expert_train", "controller_train", "validation", "test")
SUBJECT = re.compile(r"^(BraTS-GLI-\d+)-\d+$")
REPO = Path(__file__).resolve().parents[1]


def test_system_has_four_segmentation_experts_and_one_llm():
    assert SEGMENTATION_EXPERTS == ("cnn", "transformer", "boundary", "highres")
    assert EXPERTS == SEGMENTATION_EXPERTS
    assert REASONING_EXPERT == "llm"
    assert SYSTEM_EXPERTS == SEGMENTATION_EXPERTS + ("llm",)
    assert "llm" not in EXPERTS
    assert len(SEGMENTATION_EXPERTS) == 4
    assert SYSTEM_EXPERTS.count("llm") == 1


@pytest.mark.parametrize("name", SEGMENTATION_EXPERTS)
def test_segmentation_experts_four_class_forward_backward(name, cfg):
    model = build_expert(name, cfg)
    image = torch.randn(1, 4, 8, 8, 8)
    target = torch.randint(0, 4, (1, 8, 8, 8))
    output = model(image)
    assert output["logits"].shape == (1, 4, 8, 8, 8)
    assert output["probabilities"].shape == (1, 4, 8, 8, 8)
    assert output["segmentation"].shape == (1, 8, 8, 8)
    torch.testing.assert_close(output["probabilities"].sum(1), torch.ones(1, 8, 8, 8))
    expert_loss(output, target).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_experts_are_role_distinct_implementations(cfg):
    cnn, trans, boundary, highres = (build_expert(name, cfg) for name in SEGMENTATION_EXPERTS)
    assert type(cnn) is CNNExpert
    assert type(trans) is TransformerExpert
    assert type(boundary) is BoundaryExpert
    assert type(highres) is HighResolutionExpert
    assert not isinstance(boundary, CNNExpert)
    assert isinstance(cnn.backbone, ResidualUNet)
    assert isinstance(boundary.backbone, ResidualUNet)
    assert cnn.backbone.depth == 3
    assert hasattr(trans, "mid_attn") and hasattr(trans, "bot_attn")
    assert not any(isinstance(m, (torch.nn.MaxPool3d, torch.nn.AvgPool3d)) for m in highres.modules())
    image = torch.randn(1, 4, 8, 8, 8)
    extra = boundary(image)["extra"]
    assert {"boundary_logits", "boundary_probability", "distance_to_boundary"} <= extra.keys()
    loss = extra["boundary_probability"].mean() + extra["distance_to_boundary"].mean()
    loss.backward()
    assert boundary.geometry[0].block[0].weight.grad.abs().sum() > 0


def test_controller_is_four_way_voxel_softmax_fusion(cfg):
    image = torch.randn(1, 4, 8, 8, 8)
    outputs = {name: build_expert(name, cfg)(image) for name in SEGMENTATION_EXPERTS}
    debate = analyze_disagreement(outputs)
    features = controller_features(image, outputs, debate)
    assert features.shape[1] == feature_channels(cfg)
    p, availability = stack_evidence(outputs)
    assert p.shape[:2] == (1, 4)
    gate = GatingNetwork(features.shape[1], 4)
    result = gate(features, p, availability)
    assert result["weights"].shape == (1, 4, 8, 8, 8)
    torch.testing.assert_close(result["weights"].sum(1), torch.ones(1, 8, 8, 8), atol=1e-6, rtol=1e-5)
    fused = (p * result["weights"].unsqueeze(2)).sum(1)
    torch.testing.assert_close(result["probabilities"], fused, atol=1e-6, rtol=1e-5)
    average = p.mean(1)
    assert not torch.allclose(result["probabilities"], average)


def test_llm_cannot_modify_or_replace_segmentation(cfg):
    expert = LLMReasoningExpert.from_config(cfg)
    result = expert.reason({
        "patient_id": "synthetic_000",
        "disagreement_regions": [],
        "interpretation_limit": "not medical advice",
    })
    assert result.can_modify_segmentation is False
    payload = result.to_dict()
    assert payload["can_modify_segmentation"] is False
    assert not {"logits", "probabilities", "segmentation", "mask"} & payload.keys()
    with pytest.raises(ValueError, match="reasoning expert"):
        build_expert("llm", cfg)


def test_disagreement_tensor_shapes(cfg):
    image = torch.randn(1, 4, 8, 8, 8)
    outputs = {name: build_expert(name, cfg)(image) for name in SEGMENTATION_EXPERTS}
    debate = analyze_disagreement(outputs)
    spatial = (1, 1, 8, 8, 8)
    assert debate["map"].shape == spatial
    assert debate["js_divergence"].shape == spatial
    assert debate["prediction_entropy"].shape == spatial
    assert debate["variance"].shape == (1, 4, 8, 8, 8)
    assert debate["average_probabilities"].shape == (1, 4, 8, 8, 8)
    assert debate["majority_segmentation"].shape == (1, 8, 8, 8)
    assert debate["available_experts"].shape == (1, 1, 8, 8, 8)
    assert len(debate["pairwise"]) == 6
    for pair in debate["pairwise"].values():
        assert pair["disagreement"].shape == spatial
        assert pair["probability_distance"].shape == spatial
        assert pair["valid"].shape == spatial


def test_canonical_case_splits_are_subject_disjoint_and_untouched():
    cases_path = REPO / "configs/splits/brats2023_gli_cases.json"
    roles_path = REPO / "configs/splits/brats2023_gli_roles.json"
    cases = json.loads(cases_path.read_text())
    roles = json.loads(roles_path.read_text())
    assert tuple(cases) == SPLIT_KEYS
    assert tuple(roles) == SPLIT_KEYS
    case_subjects = {}
    for split, ids in cases.items():
        assert ids == sorted(ids)
        assert ids
        for pid in ids:
            match = SUBJECT.fullmatch(pid)
            assert match, pid
            subject = match.group(1)
            assert subject not in case_subjects or case_subjects[subject] == split
            case_subjects[subject] = split
    role_subjects = {}
    for split, ids in roles.items():
        for subject in ids:
            assert subject not in role_subjects or role_subjects[subject] == split
            role_subjects[subject] = split
    assert set(case_subjects) == set(role_subjects)
    for subject, split in case_subjects.items():
        assert role_subjects[subject] == split
    flat = [pid for split in SPLIT_KEYS for pid in cases[split]]
    assert len(flat) == len(set(flat))


def test_highres_network_is_independent_of_other_experts(cfg):
    model = build_expert("highres", cfg)
    image = torch.randn(1, 4, 8, 8, 8)
    output = model(image)
    assert output["probabilities"].shape == (1, 4, 8, 8, 8)
    full = sliding_predict(model, image, (8, 8, 8), "cpu")
    assert full["extra"]["availability"].min() == 1
    expert_loss(output, torch.randint(0, 4, (1, 8, 8, 8))).backward()


def test_production_config_keeps_four_experts_and_brats_regions():
    cfg = load_config(REPO / "configs/brats.yaml")
    assert cfg["model"]["transformer_backend"] == "hierarchical"
    assert cfg["model"]["unet_depth"] == 3
    assert {"WT", "TC", "ET"} <= set(cfg["regions"])
    assert cfg["evaluation"]["hd95"] is True
    assert cfg["crop_size"] == [64, 64, 64]
    assert cfg["highres_patch_size"] == [32, 32, 32]
    assert cfg["batch_size"] == 1
    assert cfg["validation_every_epochs"] == 5
    assert cfg["reasoning"]["mode"] == "deterministic"
    cnn = build_expert("cnn", cfg)
    trans = build_expert("transformer", cfg)
    assert isinstance(cnn.backbone, ResidualUNet)
    assert trans.mid_attn.window == 4
    assert feature_channels(cfg) == 65
