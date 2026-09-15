import json
import os
import torch
import pytest
from brats_debate.config import REASONING_DEFAULTS
from brats_debate.debate.disagreement import analyze_disagreement
from brats_debate.inference.pipeline import load_experts
from brats_debate.experts import build_expert
from brats_debate.reasoning.providers import (
    LLMConfigurationError, LLMProviderError, OpenAICompatibleProvider, parse_model_json,
)
from brats_debate.reasoning.reasoning_engine import LLMReasoningExpert, LLMReasoningResult, ReasoningEngine


def _evidence():
    return {
        "patient_id": "synthetic_000",
        "expert_roles": {"cnn": "local", "transformer": "context", "boundary": "Sum Medical", "highres": "detail"},
        "expert_coverage": {"cnn": 1.0, "transformer": 1.0, "boundary": 1.0, "highres": 0.5},
        "uncertainty": {"cnn": {"overall": 0.4, "disagreement_region": 0.6}},
        "controller_weights": {"cnn": {"disagreement_region": 0.7}, "transformer": {"disagreement_region": 0.3}},
        "expert_pair_with_largest_disagreement": "cnn__transformer",
        "mean_disagreement": 0.25,
        "boundary_disagreement": 0.4,
        "disagreement_regions": [{
            "id": "disagreement_region_1", "voxels": 8, "controller_weights": {"cnn": 0.8, "transformer": 0.2},
        }],
        "unusual_or_difficult_flags": [{"code": "high_disagreement", "label": "unusual / difficult regions identified by expert disagreement and uncertainty"}],
        "interpretation_limit": "not medical advice",
    }


def test_deterministic_mode_is_labeled_not_llm(cfg):
    expert = LLMReasoningExpert.from_config(cfg)
    original = json.loads(json.dumps(_evidence()))
    result = expert.reason(_evidence())
    assert result.mode == "deterministic"
    assert result.expert == "llm"
    assert result.can_modify_segmentation is False
    assert "not an LLM" in result.explanation
    assert "REAL LLM" not in result.explanation
    assert "not medical advice" in result.explanation
    assert result.disagreement_regions[0]["id"] == "disagreement_region_1"
    assert _evidence() == original


def test_llm_mode_without_key_fails(cfg):
    cfg["reasoning"] = {**REASONING_DEFAULTS, "mode": "llm", "model": "test-model"}
    os.environ.pop("BRATS_LLM_API_KEY", None)
    with pytest.raises(LLMConfigurationError, match="API key"):
        LLMReasoningExpert.from_config(cfg)


def test_llm_mode_uses_provider_and_cannot_edit_voxels():
    class Fake:
        name = "fake_llm"

        def complete(self, evidence):
            return {
                "overall_assessment": "conflict at a boundary blob",
                "expert_assessments": {"cnn": "locally consistent"},
                "likely_reliable_experts_by_region": {"disagreement_region_1": "cnn"},
                "uncertainty_summary": "moderate",
                "boundary_assessment": "interface conflict",
                "unusual_region_flags": ["high_disagreement"],
                "reasoning": "CNN is more consistent in the largest blob.",
                "limitations": "research only",
            }

    result = LLMReasoningExpert(Fake(), "llm").reason(_evidence())
    assert result.mode == "llm"
    assert "REAL LLM REASONING" in result.explanation
    assert result.can_modify_segmentation is False
    assert "segmentation" not in result.to_dict() or result.to_dict()["can_modify_segmentation"] is False
    assert not any(key in result.to_dict() for key in ("logits", "probabilities", "mask"))


def test_failed_llm_provider_does_not_pretend_to_be_an_llm():
    class Boom:
        name = "boom"

        def complete(self, evidence):
            raise LLMProviderError("offline")

    with pytest.raises(LLMProviderError, match="offline"):
        LLMReasoningExpert(Boom(), "llm").reason(_evidence())


def test_callback_must_return_structured_object():
    def narrative(_evidence):
        return "a free-form paragraph"

    with pytest.raises(TypeError, match="structured"):
        ReasoningEngine(narrative).explain(_evidence())


def test_parse_model_json_rejects_plain_text():
    with pytest.raises(LLMProviderError, match="valid JSON"):
        parse_model_json("just a story")


def test_openai_provider_requires_key_and_model():
    with pytest.raises(LLMConfigurationError, match="API key"):
        OpenAICompatibleProvider("", "gpt")
    with pytest.raises(LLMConfigurationError, match="model"):
        OpenAICompatibleProvider("sk-test", "")


def test_build_expert_still_rejects_llm(cfg):
    with pytest.raises(ValueError, match="reasoning expert"):
        build_expert("llm", cfg)


def test_load_config_defaults_reasoning(cfg):
    assert cfg["reasoning"]["mode"] == "deterministic"
    assert cfg["reasoning"]["api_key_env"] == "BRATS_LLM_API_KEY"


def test_result_dataclass_is_json_safe():
    result = LLMReasoningResult(explanation="ok")
    json.dumps(result.to_dict())


def test_case_summary_keeps_spatial_map_and_region_blobs(cfg):
    from brats_debate.reasoning.reasoning_engine import case_summary
    image = torch.zeros(1, 4, 8, 8, 8)
    image[:, :, 2:6, 2:6, 2:6] = 1
    outputs = {name: build_expert(name, cfg)(image) for name in ("cnn", "transformer", "boundary", "highres")}
    debate = analyze_disagreement(outputs)
    weights = torch.ones(1, 4, 8, 8, 8) / 4
    patient = {"patient_id": "t", "affine": torch.eye(4).numpy()}
    summary = case_summary(patient, outputs, debate, outputs["cnn"]["probabilities"], weights, cfg)
    assert summary["spatial_disagreement_map_retained"] is True
    assert summary["system_experts"]["reasoning"] == "llm"
    assert summary["expert_roles"]["boundary"].startswith("Sum Medical")
    assert "unusual / difficult" in summary["unusual_region_definition"]
    assert debate["map"].shape[-3:] == (8, 8, 8)


def test_load_experts_rejects_llm(cfg):
    with pytest.raises(ValueError, match="reasoning expert"):
        load_experts(cfg, {}, names=("llm",))
