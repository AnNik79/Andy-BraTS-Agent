"""Region-level analysis record schema.

Records are not populated for the full cohort in this instrumentation pass.
Optional future fields stay None until a later study fills them.
"""

REGION_RECORD_FIELDS = (
    "patient_id", "region_id",
    "x_start", "x_end", "y_start", "y_end", "z_start", "z_end",
    "centroid_x", "centroid_y", "centroid_z",
    "voxel_count", "volume_ml",
    "ground_truth_available", "ground_truth_error",
    "false_positive_fraction", "false_negative_fraction",
    "WT_error", "TC_error", "ET_error", "boundary_error",
    "CNN_class_probs", "Transformer_class_probs", "Boundary_class_probs", "HighRes_class_probs",
    "CNN_confidence", "Transformer_confidence", "Boundary_confidence", "HighRes_confidence",
    "CNN_entropy", "Transformer_entropy", "Boundary_entropy", "HighRes_entropy",
    "JS_disagreement", "pairwise_TV", "probability_variance", "vote_disagreement",
    "CNN_vs_Transformer", "CNN_vs_Boundary", "CNN_vs_HighRes",
    "Transformer_vs_Boundary", "Transformer_vs_HighRes", "Boundary_vs_HighRes",
    "CNN_mahalanobis", "Transformer_mahalanobis", "Boundary_mahalanobis", "HighRes_mahalanobis",
    "boundary_probability", "distance_to_boundary",
    "T1n_region_statistics", "T1c_region_statistics", "T2w_region_statistics", "T2f_region_statistics",
    "predicted_error_probability", "difficulty_score", "active_learning_priority",
    "expert_identity", "mri_modality_identity", "spatial_coordinates",
)


def empty_region_record(patient_id=None, region_id=None, **values):
    record = {field: None for field in REGION_RECORD_FIELDS}
    record["patient_id"] = patient_id
    record["region_id"] = region_id
    record["ground_truth_available"] = False
    record["spatial_coordinates"] = None
    record.update(values)
    unknown = set(values) - set(REGION_RECORD_FIELDS)
    if unknown:
        raise ValueError(f"Unknown region-record fields: {sorted(unknown)}")
    return record


def attach_spatial_identity(record, *, origin, shape, spacing=(1.0, 1.0, 1.0)):
    z0, y0, x0 = (int(v) for v in origin)
    dz, dy, dx = (int(v) for v in shape)
    record["z_start"], record["y_start"], record["x_start"] = z0, y0, x0
    record["z_end"], record["y_end"], record["x_end"] = z0 + dz, y0 + dy, x0 + dx
    record["centroid_z"] = z0 + (dz - 1) / 2
    record["centroid_y"] = y0 + (dy - 1) / 2
    record["centroid_x"] = x0 + (dx - 1) / 2
    record["voxel_count"] = dz * dy * dx
    record["volume_ml"] = float(record["voxel_count"] * spacing[0] * spacing[1] * spacing[2] / 1000.0)
    record["spatial_coordinates"] = {
        "origin_zyx": [z0, y0, x0],
        "shape_zyx": [dz, dy, dx],
        "spacing_zyx": list(spacing),
    }
    return record


def modality_statistics(image, mask, modality_names=("t1n", "t1c", "t2w", "t2f")):
    """Keep MRI modality identity; do not collapse channels anonymously."""
    import numpy as np
    volume = np.asarray(image)
    mask = np.asarray(mask, dtype=bool)
    if volume.ndim == 4 and volume.shape[0] == len(modality_names):
        channels = volume
    elif volume.ndim == 5 and volume.shape[1] == len(modality_names):
        channels = volume[0]
    else:
        raise ValueError("Image must have a dedicated modality channel axis")
    stats = {}
    for index, name in enumerate(modality_names):
        values = channels[index][mask] if mask.any() else channels[index].ravel()
        stats[name] = {
            "mean": float(values.mean()) if values.size else None,
            "std": float(values.std()) if values.size else None,
            "min": float(values.min()) if values.size else None,
            "max": float(values.max()) if values.size else None,
        }
    return stats
