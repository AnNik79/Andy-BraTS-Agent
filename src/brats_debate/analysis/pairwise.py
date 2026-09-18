"""Analysis-friendly pairwise expert comparisons.

Production disagreement (JS mixture, pairwise TV, entropy, variance, spatial
map) remains in debate.disagreement.analyze_disagreement and is not modified.
This module consumes those tensors and adds BraTS-region / error-conditioned views.
"""
from itertools import combinations
import math
import numpy as np
import torch
from ..config import SEGMENTATION_EXPERTS
from ..debate.disagreement import analyze_disagreement
from ..evaluation.metrics import BRATS_REGIONS
from .error_maps import DEFAULT_REGIONS, _as_int_volume, _binary_boundary, _region_mask


SEGMENTATION_PAIRS = tuple(combinations(SEGMENTATION_EXPERTS, 2))


def _pair_key(a, b):
    return f"{a}__{b}"


def pairwise_js_divergence(p_i, p_j):
    """Pairwise JS for analysis. Not a replacement for production mixture JS."""
    p_i, p_j = p_i.float().clamp_min(1e-8), p_j.float().clamp_min(1e-8)
    mix = 0.5 * (p_i + p_j)
    entropy = lambda p: -(p * p.log()).sum(1, keepdim=True)
    js = (entropy(mix) - 0.5 * entropy(p_i) - 0.5 * entropy(p_j)) / math.log(p_i.shape[1])
    return js.clamp(0, 1)


def _numpy(volume):
    if torch.is_tensor(volume):
        volume = volume.detach().cpu().numpy()
    array = np.asarray(volume)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    return array


def _hard(output):
    if "segmentation" in output:
        return _as_int_volume(_numpy(output["segmentation"]))
    return _as_int_volume(_numpy(output["probabilities"]).argmax(0 if np.asarray(output["probabilities"]).ndim == 4 else 1))


def pairwise_expert_analysis(outputs, debate=None, target=None, regions=None,
                             error_reference=None):
    """Voxel maps and patient summaries for each required expert pair.

    error_reference, if provided, is a segmentation-error dict from
    segmentation_error_maps for conditioning disagreement on FP/FN/boundary.
    Ground truth is optional; unlabeled disagreement does not require it.
    """
    regions = DEFAULT_REGIONS if regions is None else regions
    debate = analyze_disagreement(outputs) if debate is None else debate
    names = [name for name in SEGMENTATION_EXPERTS if name in outputs]
    result = {
        "definition": "expert disagreement = disagreement among experts; not ground-truth error",
        "production": {k: debate[k] for k in ("map", "js_divergence", "variance",
                                              "prediction_entropy", "pairwise") if k in debate},
        "pairs": {},
    }
    hards = {name: _hard(outputs[name]) for name in names}
    probs = {name: outputs[name]["probabilities"] for name in names}
    truth = None if target is None else _as_int_volume(target)
    tumor_boundary = None if truth is None else _binary_boundary(_region_mask(truth, regions.get("WT", [1, 2, 3])))
    for a, b in combinations(names, 2):
        key = _pair_key(a, b)
        production_pair = debate["pairwise"].get(key) or debate["pairwise"].get(_pair_key(b, a))
        p_a, p_b = probs[a], probs[b]
        hard_a, hard_b = hards[a], hards[b]
        hard_disagree = hard_a != hard_b
        tv = (p_a.float() - p_b.float()).abs().sum(1, keepdim=True) / 2
        js = pairwise_js_divergence(p_a, p_b)
        pair = {
            "probability_difference": (p_a.float() - p_b.float()),
            "hard_disagreement": hard_disagree,
            "pairwise_tv": tv,
            "pairwise_js": js,
            "production_hard_disagreement": None if production_pair is None else production_pair["disagreement"],
            "production_tv": None if production_pair is None else production_pair["probability_distance"],
            "region_hard_disagreement": {},
        }
        for name, ids in regions.items():
            if name not in BRATS_REGIONS:
                continue
            pair["region_hard_disagreement"][name] = _region_mask(hard_a, ids) != _region_mask(hard_b, ids)
        if tumor_boundary is not None:
            pair["boundary_hard_disagreement"] = hard_disagree & tumor_boundary
        if error_reference is not None:
            pair["disagreement_in_false_positive"] = hard_disagree & error_reference["false_positive"]
            pair["disagreement_in_false_negative"] = hard_disagree & error_reference["false_negative"]
        pair["patient_summary"] = {
            "hard_disagreement_fraction": float(hard_disagree.mean()),
            "mean_pairwise_tv": float(tv.detach().mean()),
            "mean_pairwise_js": float(js.detach().mean()),
        }
        for name, mask in pair["region_hard_disagreement"].items():
            pair["patient_summary"][f"{name}_hard_disagreement_fraction"] = float(np.asarray(mask).mean())
        result["pairs"][key] = pair
    return result
