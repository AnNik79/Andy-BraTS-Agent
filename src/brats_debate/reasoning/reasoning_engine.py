"""Fifth system expert: LLM reasoning over debate evidence. Not a voxel model."""
from dataclasses import asdict, dataclass, field
import json
import numpy as np
import torch
from scipy import ndimage
from ..config import REASONING_EXPERT, SEGMENTATION_EXPERTS, SEGMENTATION_EXPERT_ROLES
from ..data.preprocessing import boundary_target
from ..evaluation.disagreement_analysis import masked_mean
from .providers import DeterministicProvider, provider_from_config


def as_volume(tensor):
    return tensor.detach().cpu().numpy().squeeze(0).squeeze(0) if tensor.ndim == 5 else tensor.detach().cpu().numpy()[0]


def _bbox(coords):
    if len(coords) == 0:
        return None
    lo, hi = coords.min(0), coords.max(0)
    return {"min_voxel": lo.tolist(), "max_voxel": hi.tolist()}


def _world(affine, voxel):
    return (affine @ np.r_[voxel, 1])[:3].tolist() if voxel is not None else None


def _region_record(index, mask, disagreement, outputs, weights, boundary, affine, voxel_ml, names):
    coords = np.argwhere(mask)
    centroid = coords.mean(0) if len(coords) else None
    assessments = {}
    for name, out in outputs.items():
        availability = out["extra"].get("availability", torch.ones_like(out["confidence"]))
        available = (as_volume(availability) > 0) & mask
        assessments[name] = {
            "confidence": masked_mean(as_volume(out["confidence"]), available),
            "uncertainty": masked_mean(as_volume(out["uncertainty"]), available),
            "coverage_in_region": float(available.mean()) if mask.any() else 0.0,
        }
    controller = {}
    for i, name in enumerate(names):
        controller[name] = masked_mean(weights[0, i].cpu().numpy(), mask)
    return {
        "id": f"disagreement_region_{index}",
        "voxels": int(mask.sum()),
        "volume_ml": float(mask.sum() * voxel_ml),
        "centroid_voxel": centroid.tolist() if centroid is not None else None,
        "centroid_world_mm": _world(affine, centroid),
        "bbox_voxel": _bbox(coords),
        "mean_disagreement": masked_mean(disagreement, mask),
        "max_disagreement": float(disagreement[mask].max()) if mask.any() else None,
        "boundary_involvement": masked_mean(boundary, mask),
        "expert_assessments": assessments,
        "controller_weights": controller,
        "note": "unusual / difficult region identified by expert disagreement and uncertainty",
    }


def case_summary(patient, outputs, debate, final, weights, cfg):
    disagreement = as_volume(debate["map"])
    segmentation = as_volume(final.argmax(1))
    threshold = cfg["evaluation"]["high_disagreement_threshold"]
    max_regions = int((cfg.get("reasoning") or {}).get("max_disagreement_regions", 8))
    high = disagreement >= threshold
    components, count = ndimage.label(high)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    ranked = [int(i) for i in np.argsort(-sizes) if sizes[i] > 0][:max_regions]
    voxel_ml = abs(float(np.linalg.det(patient["affine"][:3, :3]))) / 1000
    boundary = boundary_target(segmentation)
    names = [name for name in SEGMENTATION_EXPERTS if name in outputs]
    regions = [_region_record(index, components == label, disagreement, outputs, weights, boundary,
                              patient["affine"], voxel_ml, names)
               for index, label in enumerate(ranked, 1)]
    largest = regions[0] if regions else {"voxels": 0, "volume_ml": 0.0, "centroid_voxel": None,
                                          "centroid_world_mm": None}
    confidence, entropy, margin, coverage = {}, {}, {}, {}
    for name, out in outputs.items():
        available = as_volume(out["extra"].get("availability", torch.ones_like(out["confidence"]))) > 0
        confidence[name] = {"overall": masked_mean(as_volume(out["confidence"]), available),
                            "disagreement_region": masked_mean(as_volume(out["confidence"]), high & available)}
        entropy[name] = {"overall": masked_mean(as_volume(out["uncertainty"]), available),
                         "disagreement_region": masked_mean(as_volume(out["uncertainty"]), high & available)}
        margin[name] = masked_mean(as_volume(out["extra"]["margin"]), available)
        coverage[name] = float(available.mean())
    pair_stats = {}
    for name, values in debate["pairwise"].items():
        valid = as_volume(values["valid"]) > 0
        pair_stats[name] = {"hard_disagreement_fraction": masked_mean(as_volume(values["disagreement"]), valid),
                            "probability_distance": masked_mean(as_volume(values["probability_distance"]), valid)}
    eligible = {k: v["probability_distance"] for k, v in pair_stats.items() if v["probability_distance"] is not None}
    region_stats = {}
    for name, ids in cfg["regions"].items():
        mask = np.isin(segmentation, ids)
        region_stats[name] = {"mean_disagreement": masked_mean(disagreement, mask),
                              "fraction_above_threshold": masked_mean(high, mask)}
    weight_stats = {}
    for i, name in enumerate(names):
        w = weights[0, i].cpu().numpy()
        weight_stats[name] = {"overall": float(w.mean()), "disagreement_region": masked_mean(w, high),
                              "largest_disagreement_region": masked_mean(w, components == ranked[0]) if ranked else None}
    highres = outputs.get("highres", {})
    highres_findings = {
        "coverage": coverage.get("highres"),
        "confidence_in_coverage": (confidence.get("highres") or {}).get("overall"),
        "uncertainty_in_coverage": (entropy.get("highres") or {}).get("overall"),
        "note": "High-resolution specialist may abstain outside proposal-selected patches.",
    }
    if "boundary_probability" in outputs.get("boundary", {}).get("extra", {}):
        high_boundary = masked_mean(as_volume(outputs["boundary"]["extra"]["boundary_probability"]), high)
    else:
        high_boundary = None
    flags = []
    if high.any():
        flags.append({"code": "high_disagreement",
                      "label": "unusual / difficult regions identified by expert disagreement and uncertainty"})
    if any((entropy[name]["overall"] or 0) > 0.8 for name in entropy if entropy[name]["overall"] is not None):
        flags.append({"code": "high_uncertainty",
                      "label": "high predictive entropy among one or more segmentation experts"})
    if (high_boundary or 0) > 0.3 or (masked_mean(disagreement, boundary) or 0) > threshold:
        flags.append({"code": "boundary_conflict",
                      "label": "disagreement concentrated at predicted class interfaces"})
    if eligible and max(eligible.values()) > 0.2:
        flags.append({"code": "expert_conflict",
                      "label": "large pairwise probability distance between segmentation experts"})
    if coverage.get("highres") is not None and coverage["highres"] < 1:
        flags.append({"code": "limited_highres_coverage",
                      "label": "high-resolution specialist abstained on part of the volume"})
    return {
        "patient_id": patient["patient_id"],
        "research_only": True,
        "system_experts": {
            "segmentation": list(SEGMENTATION_EXPERTS),
            "reasoning": REASONING_EXPERT,
            "roles": dict(SEGMENTATION_EXPERT_ROLES),
        },
        "confidence": confidence, "uncertainty": entropy, "probability_margin": margin,
        "expert_roles": dict(SEGMENTATION_EXPERT_ROLES),
        "expert_coverage": coverage, "mean_disagreement": float(disagreement.mean()),
        "disagreement_threshold": threshold, "high_disagreement_fraction": float(high.mean()),
        "boundary_disagreement": masked_mean(disagreement, boundary),
        "region_disagreement": region_stats, "pairwise": pair_stats,
        "expert_pair_with_largest_disagreement": max(eligible, key=eligible.get) if eligible else None,
        "disagreement_regions": regions,
        "largest_disagreement_region": {"voxels": largest.get("voxels", 0),
                                        "volume_ml": largest.get("volume_ml", 0.0),
                                        "centroid_voxel": largest.get("centroid_voxel"),
                                        "centroid_world_mm": largest.get("centroid_world_mm")},
        "unusual_or_difficult_flags": flags,
        "unusual_region_definition": (
            "Flags mark unusual / difficult regions identified by expert disagreement and uncertainty. "
            "This is not an anatomical abnormality detector or medical anatomy prior."
        ),
        "highres_findings": highres_findings,
        "controller_weights": weight_stats,
        "predicted_tumor_volumes_ml": {name: float(np.isin(segmentation, ids).sum() * voxel_ml)
                                         for name, ids in cfg["regions"].items()},
        "spatial_disagreement_map_retained": True,
        "debate_baselines": "average and majority vote are experimental baselines, not the debate mechanism",
        "interpretation_limit": "Weights describe learned allocation, not a causal explanation or calibrated medical uncertainty.",
    }


@dataclass
class LLMReasoningResult:
    expert: str = REASONING_EXPERT
    mode: str = "deterministic"
    provider: str = "deterministic"
    overall_assessment: str = ""
    disagreement_regions: list = field(default_factory=list)
    expert_assessments: dict = field(default_factory=dict)
    likely_reliable_experts_by_region: dict = field(default_factory=dict)
    uncertainty_summary: object = None
    boundary_assessment: str = ""
    unusual_region_flags: list = field(default_factory=list)
    reasoning: str = ""
    limitations: str = ""
    explanation: str = ""
    can_modify_segmentation: bool = False

    def to_dict(self):
        return asdict(self)


def _explanation(mode, provider, payload):
    header = ("MODE: REAL LLM REASONING" if mode == "llm"
              else "MODE: DETERMINISTIC FALLBACK / TEST MODE — this is not an LLM")
    return (
        f"{header}\nprovider: {provider}\n"
        f"{payload['overall_assessment']}\n"
        f"{payload['reasoning']}\n"
        f"{payload['limitations']}\n"
        "The LLM reasoning expert cannot modify the segmentation mask.\n"
        "Research prototype only. This explanation is not medical advice.\n"
    )


class LLMReasoningExpert:
    """Professor's fifth expert: reasoning over debate evidence. Never emits voxels."""

    name = REASONING_EXPERT

    def __init__(self, provider, mode):
        if mode not in ("llm", "deterministic"):
            raise ValueError("LLMReasoningExpert mode must be 'llm' or 'deterministic'")
        self.provider = provider
        self.mode = mode

    @classmethod
    def from_config(cls, cfg):
        provider, mode = provider_from_config(cfg)
        return cls(provider, mode)

    def reason(self, evidence):
        payload = self.provider.complete(json.loads(json.dumps(evidence)))
        return LLMReasoningResult(
            expert=REASONING_EXPERT,
            mode=self.mode,
            provider=getattr(self.provider, "name", type(self.provider).__name__),
            overall_assessment=payload["overall_assessment"],
            disagreement_regions=list(evidence.get("disagreement_regions") or []),
            expert_assessments=payload["expert_assessments"],
            likely_reliable_experts_by_region=payload["likely_reliable_experts_by_region"],
            uncertainty_summary=payload["uncertainty_summary"],
            boundary_assessment=payload["boundary_assessment"],
            unusual_region_flags=payload["unusual_region_flags"],
            reasoning=payload["reasoning"],
            limitations=payload["limitations"],
            explanation=_explanation(self.mode, getattr(self.provider, "name", "unknown"), payload),
            can_modify_segmentation=False,
        )

    def explain(self, summary):
        return self.reason(summary).explanation


class ReasoningEngine(LLMReasoningExpert):
    """Backward-compatible name. Prefer LLMReasoningExpert.from_config(cfg)."""

    def __init__(self, llm_callback=None, mode="deterministic"):
        if llm_callback is not None:
            class _Callback(DeterministicProvider):
                name = "callback"

                def complete(self, evidence):
                    result = llm_callback(json.loads(json.dumps(evidence)))
                    if isinstance(result, dict):
                        from .providers import _validated
                        return _validated(result)
                    raise TypeError("Callback providers must return the structured LLM JSON object")
            super().__init__(_Callback(), "llm")
        else:
            super().__init__(DeterministicProvider(), mode)
