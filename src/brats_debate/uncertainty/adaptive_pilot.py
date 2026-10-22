"""Fit-only training of the residual adaptive gate.

Held-out subjects are not an input to this module. Inverse-inclusion
weights estimate each subject's volume, and each subject then receives
the same total weight.
"""

from typing import Mapping, Sequence

import numpy as np
import torch

from ..controller.adaptive_gate import (
    BASE_FEATURE_NAMES,
    CLASS_COUNT,
    EXPERT_ORDER,
    ResidualAdaptiveGate,
    assemble_features,
    feature_names,
)
from ..evaluation.metrics import segmentation_metrics


TRAINING_PROTOCOL = {
    "model": "residual_adaptive_gate_v1",
    "hidden": 32,
    "epochs": 40,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "grad_clip": 1.0,
    "optimizer": "adam",
    "batching": "full_batch",
    "seed": 20261019,
    "loss": "subject_equal_inverse_inclusion_weighted_negative_log_likelihood",
    "device": "cpu",
    "residual_initialization": "zero",
    "unavailable_weight": "exact_zero_then_renormalize",
    "held_out_used_for_training_or_tuning": False,
    "entropy_threshold": 0.5,
}
PILOT_TRAIN_SUBJECTS = (
    "BraTS-GLI-00006",
    "BraTS-GLI-00025",
    "BraTS-GLI-00049",
    "BraTS-GLI-00056",
    "BraTS-GLI-00101",
    "BraTS-GLI-00104",
)
PILOT_EVAL_SUBJECTS = (
    "BraTS-GLI-00112",
    "BraTS-GLI-00133",
    "BraTS-GLI-00167",
)


def pilot_subjects(fit_case_ids: Sequence[str], rich_case_ids: Sequence[str], case_to_subject: Mapping[str, str]):
    """First six rich fit subjects train. The last three are held out.

    Order is the immutable fit manifest. Metrics do not enter the choice.
    """

    rich = set(rich_case_ids)
    ordered = []
    for case_id in fit_case_ids:
        if case_id not in rich:
            continue
        subject = case_to_subject[case_id]
        if subject not in ordered:
            ordered.append(subject)
    subjects = tuple(ordered)
    if subjects[:6] != PILOT_TRAIN_SUBJECTS or subjects[6:] != PILOT_EVAL_SUBJECTS:
        raise ValueError("rich fit subjects do not match the recorded pilot split")
    return {"train_subjects": PILOT_TRAIN_SUBJECTS, "eval_subjects": PILOT_EVAL_SUBJECTS}


def subject_equal_weights(sample_weight: np.ndarray, subject_index: np.ndarray, subject_count: int) -> np.ndarray:
    """Give every subject total weight 1 after inverse-inclusion weighting."""

    weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1)
    subject_index = np.asarray(subject_index, dtype=np.int64).reshape(-1)
    if weights.shape != subject_index.shape:
        raise ValueError("weights and subject ids are misaligned")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("inverse-inclusion weights must be finite and positive")
    balanced = weights.copy()
    for subject in range(subject_count):
        selected = subject_index == subject
        if not selected.any():
            raise ValueError("every training subject must contribute rows")
        total = balanced[selected].sum()
        balanced[selected] /= total
    if not np.allclose(balanced.reshape(-1).sum(), subject_count, rtol=0, atol=1e-6):
        raise ValueError("subject weights do not total one per subject")
    return balanced.astype(np.float32)


def rows_to_tensors(rows: Mapping[str, np.ndarray]) -> dict[str, torch.Tensor]:
    probabilities = torch.as_tensor(rows["expert_probabilities"], dtype=torch.float32)
    entropy = torch.as_tensor(rows["expert_entropy"], dtype=torch.float32)
    availability = torch.as_tensor(rows["expert_available"], dtype=torch.float32)
    frozen = torch.as_tensor(rows["controller_weights"], dtype=torch.float32)
    base = torch.as_tensor(rows["X"], dtype=torch.float32)
    if list(rows["feature_names"]) != list(BASE_FEATURE_NAMES):
        raise ValueError("base feature order does not match the recorded schema")
    features = assemble_features(probabilities, entropy, availability, frozen, base)
    if features.shape[1] != len(feature_names()):
        raise ValueError("assembled feature width changed")
    target = torch.as_tensor(rows["target_class"], dtype=torch.long)
    if target.min() < 0 or target.max() >= CLASS_COUNT:
        raise ValueError("training labels are outside the four-class mapping")
    return {
        "features": features,
        "expert_probabilities": probabilities,
        "availability": availability,
        "frozen_weights": frozen,
        "target": target,
    }


def _max_abs(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    return float(values.abs().max())


def weighted_nll(probabilities: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    chosen = probabilities.gather(1, target[:, None]).clamp_min(1e-8)
    loss = -chosen.log().squeeze(1)
    return (loss * weights).sum() / weights.sum()


def train_residual_gate(
    rows: Mapping[str, np.ndarray],
    subject_index: np.ndarray,
    *,
    protocol: Mapping[str, object] | None = None,
) -> tuple[ResidualAdaptiveGate, dict[str, object]]:
    """Train on CPU. The protocol is fixed; there is no held-out selection."""

    settings = {**TRAINING_PROTOCOL, **(protocol or {})}
    if settings["device"] != "cpu":
        raise ValueError("this pilot trains the small gate on CPU")
    tensors = rows_to_tensors(rows)
    weights = torch.as_tensor(
        subject_equal_weights(rows["sample_weight"], subject_index, int(subject_index.max()) + 1)
    )
    torch.manual_seed(int(settings["seed"]))
    model = ResidualAdaptiveGate(tensors["features"].shape[1], int(settings["hidden"]), EXPERT_ORDER.__len__())
    model.train()
    with torch.no_grad():
        initial = model(
            tensors["features"],
            tensors["expert_probabilities"],
            tensors["availability"],
            tensors["frozen_weights"],
        )
        frozen_probabilities = (
            tensors["expert_probabilities"] * tensors["frozen_weights"].unsqueeze(2)
        ).sum(1)
        initial_loss = float(weighted_nll(initial["probabilities"], tensors["target"], weights))
        frozen_loss = float(weighted_nll(frozen_probabilities, tensors["target"], weights))
        max_weight_difference = float((initial["weights"] - tensors["frozen_weights"]).abs().max())
        if max_weight_difference > 1e-5 or abs(initial_loss - frozen_loss) > 1e-5:
            raise RuntimeError("zero residual does not reproduce the frozen gate")
        if _max_abs(initial["weights"][tensors["availability"] <= 0]) != 0:
            raise RuntimeError("an unavailable expert received weight before training")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    history = []
    for _ in range(int(settings["epochs"])):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            tensors["features"],
            tensors["expert_probabilities"],
            tensors["availability"],
            tensors["frozen_weights"],
        )
        loss = weighted_nll(output["probabilities"], tensors["target"], weights)
        if not torch.isfinite(loss):
            raise RuntimeError("training loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(settings["grad_clip"]))
        optimizer.step()
        history.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        final = model(
            tensors["features"],
            tensors["expert_probabilities"],
            tensors["availability"],
            tensors["frozen_weights"],
        )
    report = {
        "protocol": dict(settings),
        "rows": int(tensors["target"].shape[0]),
        "subjects": int(subject_index.max()) + 1,
        "feature_names": list(feature_names()),
        "frozen_loss": frozen_loss,
        "initial_loss": initial_loss,
        "final_loss": history[-1],
        "loss_history": history,
        "max_init_weight_abs_difference": max_weight_difference,
        "mean_abs_weight_change": float((final["weights"] - tensors["frozen_weights"]).abs().mean()),
        "max_unavailable_weight": _max_abs(final["weights"][tensors["availability"] <= 0]),
        "min_weight_sum_residual": float((final["weights"].sum(1) - 1).abs().max()),
    }
    if report["max_unavailable_weight"] != 0 or report["min_weight_sum_residual"] > 1e-5:
        raise RuntimeError("trained gate weights failed the availability or simplex check")
    return model, report


def _spatial(tensor: torch.Tensor, channels: int) -> torch.Tensor:
    value = tensor.detach().cpu().float()
    while value.ndim > 3 and value.shape[0] == 1:
        value = value[0]
    if channels == 1:
        while value.ndim > 3:
            if value.shape[0] != 1:
                raise ValueError(f"expected one channel, got {tuple(value.shape)}")
            value = value[0]
        if value.ndim != 3:
            raise ValueError(f"expected a spatial map, got {tuple(value.shape)}")
        return value
    if value.ndim != 4 or value.shape[0] != channels:
        raise ValueError(f"expected {[channels]} leading channels, got {tuple(value.shape)}")
    return value


@torch.inference_mode()
def apply_residual_volume(
    model: ResidualAdaptiveGate,
    image: torch.Tensor,
    outputs: Mapping[str, Mapping[str, object]],
    frozen_weights: torch.Tensor,
    js_disagreement: torch.Tensor,
    predictive_entropy: torch.Tensor,
    *,
    slab_depth: int = 8,
    restrict_to_disagreement: bool = False,
) -> dict[str, torch.Tensor]:
    """Apply the trained residual on proposal-limited expert outputs."""

    model.eval()
    probabilities = torch.stack(
        [_spatial(outputs[name]["probabilities"], CLASS_COUNT) for name in EXPERT_ORDER],
        dim=0,
    )
    entropy = torch.stack(
        [_spatial(outputs[name]["uncertainty"], 1) for name in EXPERT_ORDER],
        dim=0,
    )
    availability = torch.stack(
        [_spatial(outputs[name]["extra"]["availability"], 1) for name in EXPERT_ORDER],
        dim=0,
    )
    frozen = _spatial(frozen_weights, len(EXPERT_ORDER))
    image_map = _spatial(image, 4)
    disagreement = _spatial(js_disagreement, 1)
    fused_entropy = _spatial(predictive_entropy, 1)
    boundary_probability = _spatial(outputs["boundary"]["extra"]["boundary_probability"], 1)
    boundary_distance = _spatial(outputs["boundary"]["extra"]["distance_to_boundary"], 1) / 16.0
    shape = tuple(image_map.shape[1:])
    if any(tuple(volume.shape[-3:]) != shape for volume in (probabilities[0], frozen, disagreement)):
        raise ValueError("expert, gate, and MRI grids do not match")
    fused = torch.empty((CLASS_COUNT, *shape), dtype=torch.float32)
    weights = torch.empty((len(EXPERT_ORDER), *shape), dtype=torch.float32)
    hard_disagreement = torch.empty(shape, dtype=torch.bool)
    cnn = torch.empty(shape, dtype=torch.uint8)
    majority = torch.empty(shape, dtype=torch.uint8)
    for start in range(0, shape[0], slab_depth):
        stop = min(start + slab_depth, shape[0])
        expert_probability = probabilities[:, :, start:stop].permute(2, 3, 4, 0, 1).reshape(-1, len(EXPERT_ORDER), CLASS_COUNT)
        expert_entropy = entropy[:, start:stop].permute(1, 2, 3, 0).reshape(-1, len(EXPERT_ORDER))
        expert_available = availability[:, start:stop].permute(1, 2, 3, 0).reshape(-1, len(EXPERT_ORDER))
        frozen_slab = frozen[:, start:stop].permute(1, 2, 3, 0).reshape(-1, len(EXPERT_ORDER))
        base = torch.stack(
            (
                disagreement[start:stop].reshape(-1),
                fused_entropy[start:stop].reshape(-1),
                boundary_probability[start:stop].reshape(-1),
                boundary_distance[start:stop].reshape(-1),
                image_map[0, start:stop].reshape(-1),
                image_map[1, start:stop].reshape(-1),
                image_map[2, start:stop].reshape(-1),
                image_map[3, start:stop].reshape(-1),
            ),
            dim=1,
        )
        features = assemble_features(expert_probability, expert_entropy, expert_available, frozen_slab, base)
        hard = expert_probability.argmax(2)
        visible_high = torch.where(expert_available > 0, hard, torch.full_like(hard, -1)).max(1).values
        visible_low = torch.where(expert_available > 0, hard, torch.full_like(hard, 99)).min(1).values
        slab_disagreement = visible_high != visible_low
        output = model(
            features,
            expert_probability,
            expert_available,
            frozen_slab,
            slab_disagreement.to(expert_probability.dtype) if restrict_to_disagreement else None,
        )
        depth, height, width = stop - start, shape[1], shape[2]
        fused[:, start:stop] = output["probabilities"].reshape(depth, height, width, CLASS_COUNT).permute(3, 0, 1, 2)
        weights[:, start:stop] = output["weights"].reshape(depth, height, width, len(EXPERT_ORDER)).permute(3, 0, 1, 2)
        hard_disagreement[start:stop] = slab_disagreement.reshape(depth, height, width)
        cnn[start:stop] = expert_probability[:, 0].argmax(1).reshape(depth, height, width).to(torch.uint8)
        votes = torch.zeros(expert_probability.shape[0], CLASS_COUNT)
        votes.scatter_add_(1, hard, expert_available)
        majority[start:stop] = votes.argmax(1).reshape(depth, height, width).to(torch.uint8)
    return {
        "probabilities": fused,
        "weights": weights,
        "cnn": cnn,
        "majority": majority,
        "expert_disagreement": hard_disagreement,
    }


def score_methods(predictions: Mapping[str, np.ndarray], target: np.ndarray, spacing, regions) -> dict[str, dict]:
    target = np.asarray(target)
    return {
        name: segmentation_metrics(
            np.asarray(prediction),
            target,
            regions,
            CLASS_COUNT,
            spacing=spacing,
            hd95=True,
        )
        for name, prediction in predictions.items()
    }


def voxel_change_counts(
    frozen: np.ndarray,
    adaptive: np.ndarray,
    target: np.ndarray,
    entropy: np.ndarray,
    disagreement: np.ndarray,
    *,
    entropy_threshold: float = 0.5,
) -> dict[str, dict[str, int]]:
    """Count full-volume label disagreements. This is not a Dice estimate."""

    frozen = np.asarray(frozen).reshape(-1)
    adaptive = np.asarray(adaptive).reshape(-1)
    target = np.asarray(target).reshape(-1)
    entropy = np.asarray(entropy).reshape(-1)
    disagreement = np.asarray(disagreement, dtype=bool).reshape(-1)
    frozen_wrong = frozen != target
    adaptive_wrong = adaptive != target
    corrected = frozen_wrong & ~adaptive_wrong
    introduced = ~frozen_wrong & adaptive_wrong

    def pack(mask: np.ndarray) -> dict[str, int]:
        selected = np.asarray(mask, dtype=bool).reshape(-1)
        return {
            "voxels": int(selected.sum()),
            "frozen_errors": int((selected & frozen_wrong).sum()),
            "adaptive_errors": int((selected & adaptive_wrong).sum()),
            "corrected": int((selected & corrected).sum()),
            "introduced": int((selected & introduced).sum()),
        }

    high_entropy = entropy >= entropy_threshold
    return {
        "all": pack(np.ones(target.shape[0], dtype=bool)),
        "high_entropy": pack(high_entropy),
        "low_entropy": pack(~high_entropy),
        "expert_disagreement": pack(disagreement),
        "expert_agreement": pack(~disagreement),
        "high_entropy_and_disagreement": pack(high_entropy & disagreement),
    }
