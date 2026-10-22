import numpy as np
import torch

from brats_debate.controller.adaptive_gate import ResidualAdaptiveGate, feature_names
from brats_debate.uncertainty.adaptive_pilot import rows_to_tensors
from brats_debate.uncertainty.stage2_gate import (
    disagreement_loss_weights,
    hard_disagreement,
    train_revised_gate,
)


def _rows():
    count = 20
    probabilities = np.full((count, 4, 4), 0.05, dtype=np.float32)
    probabilities[:, :, 0] = 0.85
    probabilities[:4, 0, :] = 0.05
    probabilities[:4, 0, 1] = 0.85
    probabilities /= probabilities.sum(2, keepdims=True)
    availability = np.ones((count, 4), dtype=np.float32)
    availability[0, 3] = 0
    frozen = np.full((count, 4), 0.25, dtype=np.float32)
    frozen[0] = np.array([1 / 3, 1 / 3, 1 / 3, 0], dtype=np.float32)
    return {
        "expert_probabilities": probabilities,
        "expert_entropy": np.full((count, 4), 0.2, dtype=np.float32),
        "expert_available": availability,
        "controller_weights": frozen,
        "X": np.zeros((count, 8), dtype=np.float32),
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
        "target_class": np.zeros(count, dtype=np.uint8),
        "sample_weight": np.ones(count, dtype=np.float32),
    }


def test_feature_names_and_disagreement_do_not_use_labels():
    assert not any(token in name for name in feature_names() for token in ("target", "correct", "oracle", "label"))
    rows = _rows()
    tensors = rows_to_tensors(rows)
    mask = hard_disagreement(tensors["expert_probabilities"], tensors["availability"])
    assert int(mask.sum()) == 4
    assert mask[0].item() == 1


def test_disagreement_emphasis_reaches_the_recorded_share():
    weight = np.ones(10, dtype=np.float64)
    subject = np.zeros(10, dtype=np.int64)
    disagree = np.zeros(10, dtype=bool)
    disagree[0] = True
    balanced, multipliers = disagreement_loss_weights(weight, subject, disagree, 1, 0.20)
    assert abs(float(balanced[0]) - 0.20) < 1e-6
    assert abs(float(balanced.sum()) - 1) < 1e-6
    assert multipliers[0] > 1


def test_agreement_voxels_keep_the_frozen_weights():
    rows = _rows()
    rows["target_class"][:4] = 1
    model, report = train_revised_gate(
        rows,
        np.zeros(20, dtype=np.int64),
        protocol={"epochs": 15, "learning_rate": 0.05, "weight_decay": 0, "frozen_weight_l1_penalty": 0},
    )
    tensors = rows_to_tensors(rows)
    disagree = hard_disagreement(tensors["expert_probabilities"], tensors["availability"])
    with torch.no_grad():
        weights = model(
            tensors["features"],
            tensors["expert_probabilities"],
            tensors["availability"],
            tensors["frozen_weights"],
            disagree,
        )["weights"]
    agreed = disagree <= 0
    assert torch.allclose(weights[agreed], tensors["frozen_weights"][agreed], atol=1e-5)
    assert weights[0, 3].item() == 0
    assert report["final_loss"] < report["initial_loss"]
    assert report["max_abs_weight_change_on_agreement"] <= 1e-5


def test_zero_mask_matches_a_shifted_frozen_gate():
    features = torch.zeros(2, len(feature_names()))
    probabilities = torch.full((2, 4, 4), 0.25)
    availability = torch.ones(2, 4)
    frozen = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1]])
    model = ResidualAdaptiveGate(features.shape[1])
    with torch.no_grad():
        model.network[-1].bias.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        masked = model(features, probabilities, availability, frozen, torch.tensor([0.0, 1.0]))
    assert torch.allclose(masked["weights"][0], frozen[0], atol=1e-5)
    assert not torch.allclose(masked["weights"][1], frozen[1], atol=1e-3)
