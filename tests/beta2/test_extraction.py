from pathlib import Path

import numpy as np
import pytest
import torch

from brats_debate.config import SEGMENTATION_EXPERTS
from brats_debate.debate.disagreement import analyze_disagreement
from brats_debate.experts.base import pack_output
from brats_debate.uncertainty.extraction import (
    VoxelSamplingPolicy,
    case_artifact_is_complete,
    case_provenance,
    extract_sampled_case,
    load_case_artifact,
    sample_voxels,
    save_case_artifact,
)


def test_stratified_sampling_is_bounded_deterministic_and_correctable():
    shape = (5, 5, 8)
    domain = np.ones(shape, dtype=bool)
    error = np.zeros(shape, dtype=bool)
    error.reshape(-1)[:20] = True
    boundary = np.zeros(shape, dtype=bool)
    boundary.reshape(-1)[20:60] = True
    foreground = np.zeros(shape, dtype=bool)
    foreground.reshape(-1)[60:120] = True
    policy = VoxelSamplingPolicy(
        seed=7, error=5, boundary_correct=6, foreground_correct=7, other_correct=8
    )
    first = sample_voxels(
        patient_id="case",
        error=error,
        boundary=boundary,
        foreground=foreground,
        domain=domain,
        policy=policy,
    )
    second = sample_voxels(
        patient_id="case",
        error=error,
        boundary=boundary,
        foreground=foreground,
        domain=domain,
        policy=policy,
    )
    np.testing.assert_array_equal(first["flat_index"], second["flat_index"])
    assert first["flat_index"].size == policy.maximum_voxels
    assert (
        (first["inclusion_probability"] > 0)
        & (first["inclusion_probability"] <= 1)
    ).all()
    with pytest.raises(ValueError, match="contain every"):
        sample_voxels(
            patient_id="case",
            error=error,
            boundary=boundary,
            foreground=foreground,
            domain=domain & ~error,
            policy=policy,
        )


def _fake_case(cfg):
    shape = (6, 6, 6)
    image = np.ones((4, *shape), dtype=np.float32)
    target = np.zeros(shape, dtype=np.int64)
    target[2:5, 2:5, 2:5] = 1
    probabilities = {}
    for index, name in enumerate(SEGMENTATION_EXPERTS):
        p = torch.full((1, 4, *shape), 0.02)
        labels = torch.from_numpy(target)[None]
        p.scatter_(1, labels[:, None], 0.94)
        if index == 1:
            p[:, 0, 0, 0, 0], p[:, 1, 0, 0, 0] = 0.45, 0.49
        p /= p.sum(1, keepdim=True)
        extra = {"availability": torch.ones(1, 1, *shape)}
        if name == "boundary":
            extra.update(
                {
                    "boundary_probability": torch.full((1, 1, *shape), 0.25),
                    "distance_to_boundary": torch.full((1, 1, *shape), 4.0),
                }
            )
        probabilities[name] = pack_output(
            p.log(),
            torch.zeros(1, cfg["model"]["feature_channels"], *shape),
            extra=extra,
            probabilities=p,
        )
    debate = analyze_disagreement(probabilities)
    final = torch.zeros(1, 4, *shape)
    final.scatter_(1, torch.from_numpy(target)[None, None], 1.0)
    final[:, :, 0, 0, 0] = 0
    final[:, 1, 0, 0, 0] = 1
    patient = {
        "patient_id": "validation_case",
        "image": image,
        "label": target,
    }
    return patient, probabilities, debate, final


def test_case_extraction_alignment_features_and_weighted_prevalence(cfg):
    patient, outputs, debate, final = _fake_case(cfg)
    policy = VoxelSamplingPolicy(
        seed=8, error=5, boundary_correct=10, foreground_correct=12, other_correct=15
    )
    artifact = extract_sampled_case(
        patient,
        outputs,
        debate,
        final,
        cfg,
        subject_id="validation_subject",
        policy=policy,
    )
    arrays, metadata = artifact["arrays"], artifact["metadata"]
    assert arrays["X"].shape[1] == 8
    assert arrays["X"].shape[0] <= policy.maximum_voxels
    assert np.isfinite(arrays["X"]).all()
    assert np.isfinite(arrays["sample_weight"]).all()
    np.testing.assert_allclose(
        arrays["sample_weight"], 1 / arrays["inclusion_probability"], rtol=1e-6
    )
    assert metadata["patient_id"] == "validation_case"
    assert metadata["source_split"] == "validation"
    assert metadata["full_error_voxels"] == 1
    assert metadata["js_identity_max_residual"] < 1e-5
    assert metadata["weighted_sample_error_prevalence"] == pytest.approx(
        metadata["full_error_prevalence"]
    )
    assert set(metadata["target_labels"]) == {0, 1}
    assert set(metadata["controller_prediction_labels"]) == {0, 1}


def test_atomic_case_artifact_provenance_and_resume(cfg, tmp_path):
    patient, outputs, debate, final = _fake_case(cfg)
    policy = VoxelSamplingPolicy(
        seed=9, error=2, boundary_correct=2, foreground_correct=2, other_correct=2
    )
    artifact = extract_sampled_case(
        patient,
        outputs,
        debate,
        final,
        cfg,
        subject_id="validation_subject",
        policy=policy,
    )
    splits = {
        "expert_train": ["expert"],
        "controller_train": ["controller"],
        "validation": ["validation_case"],
        "test": ["test_case"],
    }
    provenance = case_provenance(
        cfg,
        splits,
        expert_hashes={name: f"{name}_hash" for name in SEGMENTATION_EXPERTS},
        controller_hash="controller_hash",
        policy=policy,
        code_commit="commit",
    )
    path = tmp_path / "case.npz"
    save_case_artifact(path, artifact, provenance=provenance)
    loaded = load_case_artifact(path, expected_provenance=provenance)
    assert loaded["metadata"]["patient_id"] == "validation_case"
    assert set(loaded["arrays"]) == {
        "X",
        "y",
        "flat_index",
        "inclusion_probability",
        "sample_weight",
        "stratum",
    }
    assert case_artifact_is_complete(
        path,
        patient_id="validation_case",
        expected_provenance=provenance,
    )
    stale = {**provenance, "code_commit": "other"}
    assert not case_artifact_is_complete(
        path,
        patient_id="validation_case",
        expected_provenance=stale,
    )
    assert not list(Path(tmp_path).glob("*.partial*"))

    malformed = {
        "arrays": artifact["arrays"],
        "metadata": {**artifact["metadata"], "sample_count": 999},
    }
    malformed_path = tmp_path / "malformed.npz"
    save_case_artifact(malformed_path, malformed, provenance=provenance)
    with pytest.raises(ValueError, match="metadata"):
        load_case_artifact(malformed_path, expected_provenance=provenance)
