"""Metadata layout for possible later tensor decomposition.

No tensor library is added. Axes keep patient, region, modality, expert, and
feature identity instead of flattening them away.
"""

ANALYSIS_AXES = (
    "patient",
    "spatial_region",
    "mri_modality",
    "expert",
    "representation_feature",
)


def tagged_feature_vector(values, *, patient_id, region_id, expert, modality=None, feature_names=None):
    names = list(feature_names) if feature_names is not None else [f"dim_{i}" for i in range(len(values))]
    return {
        "axes": ANALYSIS_AXES,
        "patient": patient_id,
        "spatial_region": region_id,
        "mri_modality": modality,
        "expert": expert,
        "representation_feature": names,
        "values": list(values),
    }
