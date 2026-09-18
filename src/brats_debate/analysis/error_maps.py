"""Segmentation error = prediction disagrees with ground truth.

This is not expert disagreement (experts disagreeing with each other).
"""
import numpy as np
from scipy import ndimage
from ..evaluation.metrics import BRATS_REGIONS


DEFAULT_REGIONS = {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3]}


def _as_int_volume(volume):
    array = np.asarray(volume)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3:
        raise ValueError("Prediction and target must be 3D label volumes")
    return array.astype(np.int16, copy=False)


def _region_mask(labels, ids):
    return np.isin(labels, list(ids))


def _binary_boundary(mask):
    if not mask.any():
        return mask.copy()
    eroded = ndimage.binary_erosion(mask, iterations=1, border_value=0)
    return mask ^ eroded


def segmentation_error_maps(prediction, target, regions=None, classes=4):
    """Voxel maps and summaries of disagreement with ground truth."""
    pred, truth = _as_int_volume(prediction), _as_int_volume(target)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and target shapes must match")
    regions = DEFAULT_REGIONS if regions is None else regions
    error = pred != truth
    correct = ~error
    false_positive = (pred != 0) & (truth == 0)
    false_negative = (pred == 0) & (truth != 0)
    class_fp = {int(c): (pred == c) & (truth != c) for c in range(1, classes)}
    class_fn = {int(c): (truth == c) & (pred != c) for c in range(1, classes)}
    class_error = {c: class_fp[c] | class_fn[c] for c in class_fp}
    region_error = {name: _region_mask(pred, ids) != _region_mask(truth, ids)
                    for name, ids in regions.items() if name in BRATS_REGIONS or name in regions}
    tumor = _region_mask(truth, regions.get("WT", [1, 2, 3]))
    boundary = _binary_boundary(tumor)
    boundary_error = error & boundary
    voxel_count = int(pred.size)
    summaries = {
        "error_voxels": int(error.sum()),
        "error_fraction": float(error.mean()),
        "false_positive_voxels": int(false_positive.sum()),
        "false_negative_voxels": int(false_negative.sum()),
        "false_positive_fraction": float(false_positive.mean()),
        "false_negative_fraction": float(false_negative.mean()),
        "boundary_error_voxels": int(boundary_error.sum()),
        "voxel_count": voxel_count,
    }
    for name in BRATS_REGIONS:
        if name in region_error:
            summaries[f"{name}_error_voxels"] = int(region_error[name].sum())
            summaries[f"{name}_error_fraction"] = float(region_error[name].mean())
    for c, mask in class_error.items():
        summaries[f"class_{c}_error_voxels"] = int(mask.sum())
    return {
        "definition": "segmentation error = prediction disagrees with ground truth",
        "correct": correct,
        "error": error,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "class_false_positive": class_fp,
        "class_false_negative": class_fn,
        "class_error": class_error,
        "region_error": {k: region_error[k] for k in region_error if k in BRATS_REGIONS},
        "boundary": boundary,
        "boundary_error": boundary_error,
        "patient_summary": summaries,
        "region_summary": summaries,
    }
