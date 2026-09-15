from itertools import combinations
import math
import torch
import torch.nn.functional as F


def analyze_disagreement(outputs):
    names = list(outputs)
    p = torch.stack([outputs[n]["probabilities"] for n in names], 1).float()
    masks = torch.stack([outputs[n]["extra"].get("availability", torch.ones_like(outputs[n]["confidence"]))
                         for n in names], 1)
    count = masks.sum(1).clamp_min(1)
    mean = (p * masks).sum(1) / count
    variance = (((p - mean[:, None]) ** 2) * masks).sum(1) / count
    expert_entropy = -(p.clamp_min(1e-8) * p.clamp_min(1e-8).log()).sum(2, keepdim=True)
    entropy = -(mean.clamp_min(1e-8) * mean.clamp_min(1e-8).log()).sum(1, keepdim=True)
    js = ((entropy - (expert_entropy * masks).sum(1) / count) / math.log(p.shape[2])).clamp(0, 1)
    votes = (F.one_hot(p.argmax(2), p.shape[2]).movedim(-1, 2).float() * masks).sum(1)
    pairs, distances, valid_pairs = {}, [], []
    for i, j in combinations(range(len(names)), 2):
        valid = masks[:, i] * masks[:, j]
        distance = (p[:, i] - p[:, j]).abs().sum(1, keepdim=True) / 2
        disagree = (p[:, i].argmax(1, keepdim=True) != p[:, j].argmax(1, keepdim=True)).float()
        pairs[f"{names[i]}__{names[j]}"] = {"disagreement": disagree * valid,
                                            "probability_distance": distance * valid, "valid": valid}
        distances.append(distance * valid)
        valid_pairs.append(valid)
    pair_distance = torch.stack(distances).sum(0) / torch.stack(valid_pairs).sum(0).clamp_min(1)
    # average_probabilities and majority_segmentation are experimental baselines only.
    return {"map": (js + pair_distance) / 2, "js_divergence": js, "variance": variance,
            "prediction_entropy": entropy / math.log(p.shape[2]), "votes": votes,
            "pairwise": pairs, "average_probabilities": mean,
            "majority_segmentation": votes.argmax(1), "available_experts": count}
