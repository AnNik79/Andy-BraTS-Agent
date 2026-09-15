import numpy as np
from scipy.stats import spearmanr
from scipy import ndimage
from sklearn.metrics import roc_auc_score


def masked_mean(values, mask):
    return float(values[mask].mean()) if mask.any() else None


def case_error_analysis(disagreement, prediction, target, uncertainty, threshold=.25, brain_mask=None):
    error = prediction != target
    valid = np.ones(error.shape, bool) if brain_mask is None else brain_mask.astype(bool)
    high, low = valid & (disagreement >= threshold), valid & (disagreement < threshold)
    values, truth = disagreement[valid], error[valid]
    auc = float(roc_auc_score(truth, values)) if len(np.unique(truth)) == 2 else None
    near_errors = ndimage.binary_dilation(error, iterations=1) & valid
    return {"average_disagreement": masked_mean(disagreement, valid),
            "disagreement_at_errors": masked_mean(disagreement, valid & error),
            "disagreement_at_correct_voxels": masked_mean(disagreement, valid & ~error),
            "disagreement_within_one_voxel_of_error": masked_mean(disagreement, near_errors),
            "disagreement_away_from_errors": masked_mean(disagreement, valid & ~near_errors),
            "segmentation_error_rate": masked_mean(error, valid),
            "mean_uncertainty": masked_mean(uncertainty, valid),
            "high_disagreement_error_rate": masked_mean(error, high),
            "low_disagreement_error_rate": masked_mean(error, low),
            "error_detection_auroc": auc, "high_disagreement_threshold": threshold,
            "evaluated_voxels": int(valid.sum()), "error_voxels": int((error & valid).sum())}


def cohort_correlation(rows):
    pairs = [(r["average_disagreement"], r["segmentation_error_rate"]) for r in rows
             if r["average_disagreement"] is not None and r["segmentation_error_rate"] is not None]
    if len(pairs) < 3 or len(set(x for x, _ in pairs)) < 2 or len(set(y for _, y in pairs)) < 2:
        return {"spearman_r": None, "p_value": None, "patients": len(pairs)}
    r, p = spearmanr(*zip(*pairs))
    return {"spearman_r": float(r), "p_value": float(p), "patients": len(pairs)}
