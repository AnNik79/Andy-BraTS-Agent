"""Offline pairwise error-correlation vs prediction-similarity.

Error correlation asks whether two experts fail in the same places.
Prediction similarity asks whether their hard labels match.
They are reported separately and must not be substituted for each other.
"""
from itertools import combinations
import csv
import json
from pathlib import Path
import numpy as np
from ..config import SEGMENTATION_EXPERTS
from ..evaluation.metrics import BRATS_REGIONS
from .error_maps import segmentation_error_maps
from .pairwise import SEGMENTATION_PAIRS, _hard, _pair_key


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def _phi(a, b):
    a, b = np.asarray(a, dtype=np.float64).ravel(), np.asarray(b, dtype=np.float64).ravel()
    if a.size < 2 or a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _jaccard(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    union = int((a | b).sum())
    if union == 0:
        return 1.0
    return float((a & b).sum() / union)


def _dice(a, b):
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    denom = int(a.sum() + b.sum())
    if denom == 0:
        return 1.0
    return float(2 * (a & b).sum() / denom)


def error_correlation_matrix(predictions, target, names=SEGMENTATION_EXPERTS, regions=None):
    """Symmetric matrices of error-map correlation and error-region overlap."""
    names = tuple(name for name in names if name in predictions)
    errors = {name: segmentation_error_maps(_hard(predictions[name]), target, regions) for name in names}
    empty = {a: {b: None for b in names} for a in names}
    matrices = {
        "error_phi": {a: dict(empty[a]) for a in names},
        "error_jaccard": {a: dict(empty[a]) for a in names},
        "false_positive_jaccard": {a: dict(empty[a]) for a in names},
        "false_negative_jaccard": {a: dict(empty[a]) for a in names},
        **{f"{region}_error_jaccard": {a: dict(empty[a]) for a in names} for region in BRATS_REGIONS},
    }
    long_form = []
    for a, b in combinations(names, 2):
        ea, eb = errors[a], errors[b]
        values = {
            "error_phi": _phi(ea["error"], eb["error"]),
            "error_jaccard": _jaccard(ea["error"], eb["error"]),
            "false_positive_jaccard": _jaccard(ea["false_positive"], eb["false_positive"]),
            "false_negative_jaccard": _jaccard(ea["false_negative"], eb["false_negative"]),
        }
        for region in BRATS_REGIONS:
            values[f"{region}_error_jaccard"] = _jaccard(ea["region_error"][region], eb["region_error"][region])
        for metric, value in values.items():
            matrices[metric][a][b] = matrices[metric][b][a] = value
        long_form.append({"pair": _pair_key(a, b), "kind": "error_correlation", **values})
    for name in names:
        for metric in matrices:
            matrices[metric][name][name] = 1.0
    return {"kind": "error_correlation", "experts": names, "matrices": matrices, "pairs": long_form,
            "per_expert_error": {name: errors[name]["patient_summary"] for name in names}}


def prediction_similarity_matrix(predictions, names=SEGMENTATION_EXPERTS):
    """Hard-prediction agreement. Not an error-correlation matrix."""
    names = tuple(name for name in names if name in predictions)
    hards = {name: _hard(predictions[name]) for name in names}
    dice = {a: {b: None for b in names} for a in names}
    agree = {a: {b: None for b in names} for a in names}
    long_form = []
    for a, b in combinations(names, 2):
        same = hards[a] == hards[b]
        values = {"hard_agreement_fraction": float(same.mean()),
                  "hard_dice": _dice(hards[a] != 0, hards[b] != 0)}
        dice[a][b] = dice[b][a] = values["hard_dice"]
        agree[a][b] = agree[b][a] = values["hard_agreement_fraction"]
        long_form.append({"pair": _pair_key(a, b), "kind": "prediction_similarity", **values})
    for name in names:
        dice[name][name] = agree[name][name] = 1.0
    return {"kind": "prediction_similarity", "experts": names,
            "matrices": {"hard_dice": dice, "hard_agreement_fraction": agree},
            "pairs": long_form}


def export_matrices(payload, directory, stem):
    """Write matrix CSV + JSON and long-form JSON. Callers choose when to persist."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.json"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False, default=_json_default))
    experts = list(payload["experts"])
    for name, matrix in payload["matrices"].items():
        csv_path = directory / f"{stem}_{name}.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["expert", *experts])
            for row_name in experts:
                writer.writerow([row_name, *[matrix[row_name][col] for col in experts]])
    long_path = directory / f"{stem}_pairs.json"
    long_path.write_text(json.dumps(payload["pairs"], indent=2, allow_nan=False, default=_json_default))
    return {"json": json_path, "long_form": long_path}
