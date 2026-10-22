import numpy as np
import torch

from brats_debate.controller.adaptive_gate import ResidualAdaptiveGate, assemble_features, feature_names
from brats_debate.uncertainty.adaptive_pilot import (
    PILOT_EVAL_SUBJECTS,
    PILOT_TRAIN_SUBJECTS,
    apply_residual_volume,
    pilot_subjects,
    subject_equal_weights,
    rows_to_tensors,
    train_residual_gate,
    voxel_change_counts,
)


def test_recorded_split_follows_manifest_order_and_keeps_subject_scans_together():
    fit = [
        "BraTS-GLI-00006-000",
        "BraTS-GLI-00025-000",
        "BraTS-GLI-00049-000",
        "BraTS-GLI-00056-000",
        "BraTS-GLI-00056-001",
        "BraTS-GLI-00101-000",
        "BraTS-GLI-00104-000",
        "BraTS-GLI-00112-000",
        "BraTS-GLI-00133-000",
        "BraTS-GLI-00167-000",
    ]
    subjects = {case_id: case_id.rsplit("-", 1)[0] for case_id in fit}
    split = pilot_subjects(fit, fit, subjects)
    assert split["train_subjects"] == PILOT_TRAIN_SUBJECTS
    assert split["eval_subjects"] == PILOT_EVAL_SUBJECTS
    assert "BraTS-GLI-00056" in split["train_subjects"]
    assert set(split["train_subjects"]).isdisjoint(split["eval_subjects"])


def test_features_exclude_labels_and_zero_residual_reproduces_frozen_weights():
    names = feature_names()
    assert not any(token in name for name in names for token in ("target", "correct", "oracle", "label"))
    voxels, experts, classes = 5, 4, 4
    probabilities = torch.zeros(voxels, experts, classes)
    probabilities[:, :, 0] = 0.7
    probabilities[:, :, 1] = 0.3
    probabilities[0, 3] = torch.tensor([0.25, 0.25, 0.25, 0.25])
    entropy = torch.rand(voxels, experts)
    availability = torch.ones(voxels, experts)
    availability[0, 3] = 0
    frozen = torch.tensor([0.4, 0.3, 0.3, 0.0])[None].repeat(voxels, 1)
    frozen[1:] = torch.tensor([0.2, 0.2, 0.2, 0.4])
    base = torch.randn(voxels, 8)
    features = assemble_features(probabilities, entropy, availability, frozen, base)
    model = ResidualAdaptiveGate(features.shape[1])
    output = model(features, probabilities, availability, frozen)
    assert output["weights"][0, 3].item() == 0
    assert torch.allclose(output["weights"].sum(1), torch.ones(voxels), atol=1e-6)
    assert torch.allclose(output["weights"], frozen, atol=1e-5)
    assert output["residual_logits"].abs().max().item() == 0


def test_subject_weights_are_one_per_subject():
    sample_weight = np.array([1, 1, 1, 10, 10], dtype=np.float32)
    subject_index = np.array([0, 0, 0, 1, 1])
    balanced = subject_equal_weights(sample_weight, subject_index, 2)
    assert abs(float(balanced[:3].sum()) - 1) < 1e-6
    assert abs(float(balanced[3:].sum()) - 1) < 1e-6


def test_training_moves_weight_toward_the_correct_available_expert():
    rows = 64
    probabilities = np.full((rows, 4, 4), 0.05, dtype=np.float32)
    probabilities[:, 0, 1] = 0.85
    probabilities[:, 1:, 0] = 0.85
    probabilities /= probabilities.sum(2, keepdims=True)
    rows_map = {
        "expert_probabilities": probabilities,
        "expert_entropy": np.full((rows, 4), 0.2, dtype=np.float32),
        "expert_available": np.ones((rows, 4), dtype=np.float32),
        "controller_weights": np.full((rows, 4), 0.25, dtype=np.float32),
        "X": np.zeros((rows, 8), dtype=np.float32),
        "feature_names": [
            "js_disagreement",
            "predictive_entropy",
            "boundary_probability",
            "boundary_distance",
            "mri_t1n",
            "mri_t1c",
            "mri_t2w",
            "mri_t2f",
        ],
        "target_class": np.ones(rows, dtype=np.uint8),
        "sample_weight": np.ones(rows, dtype=np.float32),
    }
    model, report = train_residual_gate(
        rows_map,
        np.zeros(rows, dtype=np.int64),
        protocol={"epochs": 25, "learning_rate": 0.05, "weight_decay": 0},
    )
    assert report["final_loss"] < report["initial_loss"]
    tensors = rows_to_tensors(rows_map)
    with torch.no_grad():
        trained_weights = model(
            tensors["features"],
            tensors["expert_probabilities"],
            tensors["availability"],
            tensors["frozen_weights"],
        )["weights"]
    assert float(trained_weights[:, 0].mean()) > float(trained_weights[:, 1].mean())


def test_volume_application_keeps_unavailable_weight_at_zero():
    shape = (2, 2, 2)
    probabilities = torch.zeros(1, 4, *shape)
    probabilities[:, 1] = 1
    entropy = torch.full((1, 1, *shape), 0.2)
    availability = torch.ones(1, 1, *shape)
    outputs = {}
    for index, name in enumerate(("cnn", "transformer", "boundary", "highres")):
        extra = {"availability": availability.clone(), "boundary_probability": entropy.clone(), "distance_to_boundary": torch.ones(1, 1, *shape)}
        if name == "highres":
            extra["availability"] = torch.zeros_like(availability)
        outputs[name] = {"probabilities": probabilities.clone(), "uncertainty": entropy.clone(), "extra": extra}
    frozen = torch.zeros(1, 4, *shape)
    frozen[:, :3] = 1 / 3
    image = torch.zeros(1, 4, *shape)
    model = ResidualAdaptiveGate(len(feature_names()))
    result = apply_residual_volume(model, image, outputs, frozen, entropy, entropy, slab_depth=1)
    assert torch.equal(result["weights"][3], torch.zeros(shape))
    assert torch.allclose(result["weights"].sum(0), torch.ones(shape), atol=1e-5)
    counts = voxel_change_counts(
        np.zeros(shape, dtype=np.uint8),
        np.ones(shape, dtype=np.uint8),
        np.ones(shape, dtype=np.uint8),
        np.full(shape, 0.8),
        np.ones(shape, dtype=bool),
    )
    assert counts["all"]["corrected"] == 8
    assert counts["all"]["introduced"] == 0
    assert counts["high_entropy"]["corrected"] == 8
