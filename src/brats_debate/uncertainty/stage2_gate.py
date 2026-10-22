"""Revised residual gate for the Stage 2 fit-only study.

The correction is allowed only where available experts already disagree.
Each subject's disagreement voxels are given 20 percent of that subject's
loss weight when their natural sampling weight is smaller than that.
A fixed penalty keeps the weights near the frozen controller.
"""

from typing import Mapping

import numpy as np
import torch

from ..controller.adaptive_gate import EXPERT_ORDER, ResidualAdaptiveGate
from .adaptive_pilot import rows_to_tensors, subject_equal_weights, weighted_nll


REVISED_PROTOCOL = {
    "model": "residual_adaptive_gate_v2",
    "hidden": 32,
    "epochs": 80,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "grad_clip": 1.0,
    "optimizer": "adam",
    "batching": "full_batch",
    "seed": 20261020,
    "device": "cpu",
    "residual_initialization": "zero",
    "eligible_region": "available_expert_hard_disagreement",
    "disagreement_loss_share_target": 0.20,
    "frozen_weight_l1_penalty": 0.02,
    "held_out_used_for_training_or_tuning": False,
    "capped_error_samples": "inverse_inclusion_weights_not_treated_as_a_census",
}


def hard_disagreement(expert_probabilities: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
    """One where available experts do not share a hard class. Labels are unused."""

    hard = expert_probabilities.argmax(2)
    highest = torch.where(availability > 0, hard, torch.full_like(hard, -1)).max(1).values
    lowest = torch.where(availability > 0, hard, torch.full_like(hard, 99)).min(1).values
    return (highest != lowest).to(dtype=expert_probabilities.dtype)


def disagreement_loss_weights(
    sample_weight: np.ndarray,
    subject_index: np.ndarray,
    disagree: np.ndarray,
    subject_count: int,
    target_share: float = 0.20,
) -> tuple[np.ndarray, list[float]]:
    """Raise disagreement weight to the target share, then equalize subjects."""

    if not 0 < target_share < 0.5:
        raise ValueError("the disagreement loss share must stay below one half")
    weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    subject_index = np.asarray(subject_index, dtype=np.int64).reshape(-1)
    disagree = np.asarray(disagree, dtype=bool).reshape(-1)
    boosted = weights.copy()
    multipliers = []
    for subject in range(subject_count):
        selected = subject_index == subject
        total = weights[selected].sum()
        share = weights[selected & disagree].sum() / total
        if share <= 0 or share >= target_share:
            multiplier = 1.0
        else:
            multiplier = (target_share * (1.0 - share)) / (share * (1.0 - target_share))
        boosted[selected & disagree] *= multiplier
        multipliers.append(float(multiplier))
    return subject_equal_weights(boosted, subject_index, subject_count), multipliers


def train_revised_gate(
    rows: Mapping[str, np.ndarray],
    subject_index: np.ndarray,
    *,
    protocol: Mapping[str, object] | None = None,
) -> tuple[ResidualAdaptiveGate, dict[str, object]]:
    settings = {**REVISED_PROTOCOL, **(protocol or {})}
    if settings["device"] != "cpu":
        raise ValueError("this study trains the small gate on CPU")
    tensors = rows_to_tensors(rows)
    subject_count = int(np.asarray(subject_index).max()) + 1
    disagree = hard_disagreement(tensors["expert_probabilities"], tensors["availability"])
    loss_weights, multipliers = disagreement_loss_weights(
        rows["sample_weight"],
        subject_index,
        disagree.numpy(),
        subject_count,
        float(settings["disagreement_loss_share_target"]),
    )
    penalty_weights = torch.as_tensor(subject_equal_weights(rows["sample_weight"], subject_index, subject_count))
    loss_weights = torch.as_tensor(loss_weights)
    torch.manual_seed(int(settings["seed"]))
    model = ResidualAdaptiveGate(tensors["features"].shape[1], int(settings["hidden"]), len(EXPERT_ORDER))

    def predict():
        return model(
            tensors["features"],
            tensors["expert_probabilities"],
            tensors["availability"],
            tensors["frozen_weights"],
            disagree,
        )

    model.train()
    with torch.no_grad():
        initial = predict()
        frozen_probabilities = (tensors["expert_probabilities"] * tensors["frozen_weights"].unsqueeze(2)).sum(1)
        initial_loss = float(weighted_nll(initial["probabilities"], tensors["target"], loss_weights))
        frozen_loss = float(weighted_nll(frozen_probabilities, tensors["target"], loss_weights))
        max_weight_difference = float((initial["weights"] - tensors["frozen_weights"]).abs().max())
        if max_weight_difference > 1e-5 or abs(initial_loss - frozen_loss) > 1e-5:
            raise RuntimeError("zero residual does not reproduce the frozen gate")
        agreed = disagree <= 0
        if agreed.any() and float((initial["weights"][agreed] - tensors["frozen_weights"][agreed]).abs().max()) > 1e-5:
            raise RuntimeError("the gate changed a voxel where the experts agree")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    history = []
    penalty_scale = float(settings["frozen_weight_l1_penalty"])
    for _ in range(int(settings["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        output = predict()
        deviation = (output["weights"] - tensors["frozen_weights"]).abs().sum(1)
        penalty = (deviation * penalty_weights).sum() / penalty_weights.sum()
        loss = weighted_nll(output["probabilities"], tensors["target"], loss_weights) + penalty_scale * penalty
        if not torch.isfinite(loss):
            raise RuntimeError("training loss became non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["grad_clip"]))
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("training gradient became non-finite")
        optimizer.step()
        history.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        final = predict()
        agreed = disagree <= 0
    report = {
        "protocol": dict(settings),
        "rows": int(tensors["target"].shape[0]),
        "subjects": subject_count,
        "disagreement_multipliers": multipliers,
        "frozen_loss": frozen_loss,
        "initial_loss": initial_loss,
        "final_loss": history[-1],
        "loss_history": history,
        "max_init_weight_abs_difference": max_weight_difference,
        "mean_abs_weight_change": float((final["weights"] - tensors["frozen_weights"]).abs().mean()),
        "mean_abs_weight_change_on_disagreement": float(
            (final["weights"][disagree > 0] - tensors["frozen_weights"][disagree > 0]).abs().mean()
        ) if bool((disagree > 0).any()) else 0.0,
        "max_abs_weight_change_on_agreement": float(
            (final["weights"][agreed] - tensors["frozen_weights"][agreed]).abs().max()
        ) if bool(agreed.any()) else 0.0,
        "max_unavailable_weight": float(final["weights"][tensors["availability"] <= 0].abs().max())
        if bool((tensors["availability"] <= 0).any()) else 0.0,
    }
    if report["max_unavailable_weight"] != 0 or report["max_abs_weight_change_on_agreement"] > 1e-5:
        raise RuntimeError("the revised gate left the frozen prediction outside its eligible region")
    return model, report
