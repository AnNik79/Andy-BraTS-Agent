import numpy as np
from scipy import ndimage

BRATS_REGIONS = ("WT", "TC", "ET")


def binary_metrics(pred, target, spacing=(1, 1, 1), hd95=False):
    pred, target = pred.astype(bool), target.astype(bool)
    tp, p, t = int((pred & target).sum()), int(pred.sum()), int(target.sum())
    result = {"dice": 2 * tp / (p + t) if p + t else 1.,
              "sensitivity": tp / t if t else (1. if not p else 0.),
              "precision": tp / p if p else (1. if not t else 0.)}
    if hd95:
        if not p and not t:
            result["hd95_mm"] = 0.
        elif not p or not t:
            result["hd95_mm"] = None  # Undefined/infinite; never silently replace by zero.
        else:
            a = pred ^ ndimage.binary_erosion(pred)
            b = target ^ ndimage.binary_erosion(target)
            distances = np.r_[ndimage.distance_transform_edt(~b, sampling=spacing)[a],
                              ndimage.distance_transform_edt(~a, sampling=spacing)[b]]
            result["hd95_mm"] = float(np.percentile(distances, 95))
    return result


def segmentation_metrics(pred, target, regions, classes, spacing=(1, 1, 1), hd95=False):
    groups = {**{f"class_{i}": [i] for i in range(1, classes)}, **regions}
    metrics = {name: binary_metrics(np.isin(pred, ids), np.isin(target, ids), spacing, hd95)
               for name, ids in groups.items()}
    # Mean Dice is the unweighted mean of configured tumor regions, not background.
    metrics["mean_dice"] = float(np.mean([metrics[name]["dice"] for name in regions]))
    # Official BraTS composite: Enhancing Tumor, Tumor Core, Whole Tumor only.
    missing = [name for name in BRATS_REGIONS if name not in metrics]
    if missing:
        raise ValueError(f"BraTS region metrics require {BRATS_REGIONS}; missing {missing}")
    metrics["brats_mean_dice"] = float(np.mean([metrics[name]["dice"] for name in BRATS_REGIONS]))
    return metrics
