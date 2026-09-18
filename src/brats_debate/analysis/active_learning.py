"""Unlabeled active-learning candidate scores. No ground truth allowed."""
import numpy as np


def unlabeled_region_priority(*, disagreement, uncertainty, novelty,
                              predicted_error_probability=None, region_diversity=None,
                              ground_truth=None, labels=None, target=None):
    """Rank regions for future annotation. Refuses any ground-truth argument."""
    if ground_truth is not None or labels is not None or target is not None:
        raise ValueError("Active-learning ranking must not use ground truth or held-out labels")
    parts = [np.asarray(disagreement, dtype=np.float64).ravel(),
             np.asarray(uncertainty, dtype=np.float64).ravel(),
             np.asarray(novelty, dtype=np.float64).ravel()]
    if predicted_error_probability is not None:
        parts.append(np.asarray(predicted_error_probability, dtype=np.float64).ravel())
    if region_diversity is not None:
        parts.append(np.asarray(region_diversity, dtype=np.float64).ravel())
    stacked = np.stack(parts, 0)
    ranked = np.empty_like(stacked)
    for index, row in enumerate(stacked):
        order = row.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.linspace(0, 1, row.size)
        ranked[index] = ranks
    score = ranked.mean(0)
    return {
        "priority": score,
        "uses_ground_truth": False,
        "signals": ["disagreement", "uncertainty", "representation_novelty"]
        + (["predicted_error_probability"] if predicted_error_probability is not None else [])
        + (["region_diversity"] if region_diversity is not None else []),
    }
