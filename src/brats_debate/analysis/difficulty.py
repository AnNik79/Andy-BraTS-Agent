"""Preserve enough information to test whether disagreement/novelty predict error.

Low disagreement is not treated as safe: all experts can share the same mistake.
High Mahalanobis with low disagreement remains an explicit, studyable case.
"""
import numpy as np


QUADRANT_A = "low_disagreement_low_novelty"
QUADRANT_B = "high_disagreement_low_novelty"
QUADRANT_C = "high_disagreement_high_novelty"
QUADRANT_D = "low_disagreement_high_novelty"


def novelty_disagreement_quadrant(disagreement, novelty, disagreement_threshold, novelty_threshold):
    """Label regions/voxels into disagreement × novelty quadrants without safety claims."""
    d = np.asarray(disagreement) >= disagreement_threshold
    n = np.asarray(novelty) >= novelty_threshold
    labels = np.empty(np.broadcast(d, n).shape, dtype=object)
    labels[~d & ~n] = QUADRANT_A
    labels[d & ~n] = QUADRANT_B
    labels[d & n] = QUADRANT_C
    labels[~d & n] = QUADRANT_D
    return {
        "quadrant": labels,
        "note": "Low disagreement does not imply the region is safe or correctly segmented.",
        "thresholds": {"disagreement": float(disagreement_threshold), "novelty": float(novelty_threshold)},
        "counts": {
            QUADRANT_A: int((~d & ~n).sum()),
            QUADRANT_B: int((d & ~n).sum()),
            QUADRANT_C: int((d & n).sum()),
            QUADRANT_D: int((~d & n).sum()),
        },
    }
