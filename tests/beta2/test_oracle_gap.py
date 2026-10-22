import numpy as np
import pytest
import torch

from brats_debate.config import SEGMENTATION_EXPERTS
from brats_debate.debate.disagreement import analyze_disagreement
from brats_debate.experts.base import pack_output
from brats_debate.uncertainty.disagreement import analyze_uncertainty
from brats_debate.uncertainty.extraction import (
    VoxelSamplingPolicy,
    compare_reused_sample,
    extract_reused_sample,
    extract_sampled_case,
)
from brats_debate.uncertainty.oracle_gap import case_oracle_summary


def _fake_case(cfg, shape=(6, 6, 6)):
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
    return {
        "patient_id": "validation_case",
        "image": image,
        "label": target,
    }, probabilities, analyze_disagreement(probabilities), _final(target)


def _final(target):
    final = torch.zeros(1, 4, *target.shape)
    final.scatter_(1, torch.from_numpy(target)[None, None], 1.0)
    final[:, :, 0, 0, 0] = 0
    final[:, 1, 0, 0, 0] = 1
    return final


def _uncertainty_maps(outputs):
    uncertainty = analyze_uncertainty(outputs)
    return {
        "js_disagreement": uncertainty["js_divergence"],
        "predictive_entropy": uncertainty["mean_predictive_entropy"],
    }


def test_reused_sample_keeps_coordinates_and_adds_expert_records(cfg):
    patient, outputs, debate, final = _fake_case(cfg)
    policy = VoxelSamplingPolicy(
        seed=8, error=5, boundary_correct=4, foreground_correct=4, other_correct=4
    )
    original = extract_sampled_case(
        patient,
        outputs,
        debate,
        final,
        cfg,
        subject_id="validation_subject",
        policy=policy,
    )
    weights = torch.full((1, 4, *patient["label"].shape), 0.25)
    rich = extract_reused_sample(
        patient,
        outputs,
        final,
        weights,
        _uncertainty_maps(outputs),
        cfg,
        subject_id="validation_subject",
        original=original["arrays"],
    )
    comparison = compare_reused_sample(
        original["arrays"],
        rich["arrays"],
        original["metadata"]["feature_names"],
    )
    assert comparison["integer_fields_equal"] is True
    assert comparison["max_abs"] == pytest.approx(0)
    assert rich["arrays"]["expert_probabilities"].shape[1:] == (4, 4)
    assert rich["arrays"]["controller_weights"].shape[1] == len(SEGMENTATION_EXPERTS)
    assert rich["metadata"]["highres_full_volume_used"] is False
    assert rich["metadata"]["ground_truth_used_for_inference"] is False
    np.testing.assert_array_equal(
        rich["arrays"]["controller_correct"], 1 - rich["arrays"]["y"]
    )
    changed = dict(original["arrays"])
    changed["y"] = 1 - changed["y"]
    with pytest.raises(ValueError, match="y"):
        compare_reused_sample(changed, rich["arrays"], original["metadata"]["feature_names"])


def test_oracle_gap_separates_selection_from_mixture_and_weights():
    rows = 2
    arrays = {
        "expert_correct": np.asarray([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.uint8),
        "expert_available": np.ones((rows, 4), dtype=np.uint8),
        "controller_weights": np.asarray(
            [[0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1]], dtype=np.float32
        ),
        "controller_correct": np.asarray([1, 0], dtype=np.uint8),
        "sample_weight": np.asarray([1.0, 4.0], dtype=np.float32),
        "target_class": np.asarray([3, 1], dtype=np.uint8),
        "X": np.asarray(
            [
                [0.1, 0.2, 0.1, 0, 0, 0, 0, 0],
                [0.8, 0.9, 0.7, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
    }
    names = [
        "js_disagreement",
        "predictive_entropy",
        "boundary_probability",
        "boundary_distance",
        "mri_t1n",
        "mri_t1c",
        "mri_t2w",
        "mri_t2f",
    ]
    summary = case_oracle_summary(arrays, names)
    headroom = summary["selection_headroom_when_controller_wrong"]
    assert headroom["condition_count"] == 1
    assert headroom["sampled_fraction"] == pytest.approx(1)
    mixture = summary["mixture_correct_when_every_hard_expert_wrong"]
    assert mixture["sampled_fraction"] == pytest.approx(1)
    unresolved = summary["hard_selection_cannot_repair"]
    assert unresolved["event_count"] == 0
    assert summary["mean_weight_on_correct_experts"]["sampled_mean"] == pytest.approx(0.35)
    high = summary["by_disagreement_selection_headroom_when_controller_wrong"]["high"]
    assert high["condition_count"] == 1


def test_mixture_rescue_is_outside_the_controller_error_census():
    arrays = {
        "expert_correct": np.zeros((2, 4), dtype=np.uint8),
        "expert_available": np.ones((2, 4), dtype=np.uint8),
        "controller_weights": np.full((2, 4), 0.25, dtype=np.float32),
        "controller_correct": np.asarray([1, 0], dtype=np.uint8),
        "sample_weight": np.asarray([2.0, 1.0], dtype=np.float32),
        "target_class": np.asarray([2, 2], dtype=np.uint8),
        "X": np.zeros((2, 8), dtype=np.float32),
    }
    names = [
        "js_disagreement",
        "predictive_entropy",
        "boundary_probability",
        "boundary_distance",
        "mri_t1n",
        "mri_t1c",
        "mri_t2w",
        "mri_t2f",
    ]
    summary = case_oracle_summary(arrays, names)
    census = summary["controller_error_census"]
    rescue = summary["controller_correct_when_all_hard_experts_wrong"]
    assert census["error_rows"] == 1
    assert census["no_correct_hard_expert_rows"] == 1
    assert census["selection_correctable_rows"] == 0
    assert census["complete"] is True
    assert rescue["sample_event_count"] == 1
    assert rescue["disjoint_from_controller_errors"] is True
    assert rescue["full_volume_count_known"] is False
    mixture = summary["mixture_correct_when_every_hard_expert_wrong"]
    assert mixture["event_count"] == 1
    assert mixture["condition_count"] == 2
