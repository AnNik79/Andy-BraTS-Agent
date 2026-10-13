import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "beta2_stability_benchmark.py"
SPEC = importlib.util.spec_from_file_location("beta2_stability_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_approved_case_selection_is_validation_only_and_subject_grouped():
    splits = {
        "validation": list(benchmark.APPROVED_CASES),
        "test": ["test-case"],
    }
    subjects = ["subject-6", "subject-25", "subject-31"]
    benchmark.validate_case_selection(
        benchmark.APPROVED_CASES, splits, subjects
    )
    with pytest.raises(ValueError, match="historical test"):
        contaminated = {
            **splits,
            "test": [benchmark.APPROVED_CASES[-1]],
        }
        benchmark.validate_case_selection(
            benchmark.APPROVED_CASES, contaminated, subjects
        )
    with pytest.raises(ValueError, match="different subjects"):
        benchmark.validate_case_selection(
            benchmark.APPROVED_CASES,
            splits,
            ["subject-6", "subject-25", "subject-25"],
        )


def _artifact():
    arrays = {
        "X": np.arange(16, dtype=np.float32).reshape(2, 8),
        "y": np.array([0, 1], dtype=np.uint8),
        "flat_index": np.array([2, 9], dtype=np.int64),
        "inclusion_probability": np.array([0.5, 1.0], dtype=np.float32),
        "sample_weight": np.array([2.0, 1.0], dtype=np.float32),
        "stratum": np.array([1, 2], dtype=np.uint8),
    }
    metadata = {
        "feature_names": list(benchmark.FEATURE_NAMES),
        "spatial_shape": [2, 2, 2],
        "class_order": [0, 1, 2, 3],
        "sample_count": 2,
        "full_domain_voxels": 8,
        "full_error_voxels": 1,
        "controller_prediction_labels": [0, 1],
        "target_labels": [0, 1],
    }
    return {"arrays": arrays, "metadata": metadata}


def test_artifact_validation_and_equivalence_diagnostics():
    original = _artifact()
    benchmark.validate_artifact(original)
    optimized = _artifact()
    exact = benchmark.compare_artifacts(original, optimized)
    assert all(item["exact"] for item in exact["arrays"].values())
    assert all(item["exact"] for item in exact["metadata"].values())

    optimized["arrays"]["X"][0, 0] += 0.25
    changed = benchmark.compare_artifacts(original, optimized)
    assert not changed["arrays"]["X"]["exact"]
    assert changed["arrays"]["X"]["maximum_absolute_difference"] == 0.25
    assert changed["arrays"]["X"]["changed_values"] == 1
