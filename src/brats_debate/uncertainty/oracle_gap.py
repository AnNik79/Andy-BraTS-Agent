"""Retrospective expert-selection headroom for the frozen controller.

Ground truth is used only to score already computed predictions. It is not a
controller input, and the resulting oracle is not a deployable gate.
"""

from typing import Mapping, Sequence

import numpy as np

from ..config import SEGMENTATION_EXPERTS


EXPERT_ORDER = tuple(SEGMENTATION_EXPERTS)
REGION_LABELS = {"ET": (3,), "TC": (1, 3), "WT": (1, 2, 3)}


def _weighted_mean(event: np.ndarray, weight: np.ndarray) -> float | None:
    event = np.asarray(event, dtype=np.float64).reshape(-1)
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    if event.size == 0 or not np.isfinite(weight).all() or weight.sum() <= 0:
        return None
    return float(np.average(event, weights=weight))


def _rates(mask: np.ndarray, weight: np.ndarray) -> dict[str, float | int | None]:
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    return {
        "sampled_count": int(mask.sum()),
        "sampled_fraction_of_rows": float(mask.mean()) if mask.size else None,
        "inclusion_weighted_fraction": _weighted_mean(mask, weight),
    }


def case_oracle_summary(
    arrays: Mapping[str, np.ndarray],
    feature_names: Sequence[str],
    *,
    expert_order: Sequence[str] = EXPERT_ORDER,
    high_disagreement: float = 0.25,
    high_entropy: float = 0.5,
    boundary_probability: float = 0.5,
) -> dict[str, object]:
    """Score one reused sample. Raw rates are sample-conditional.

    Within the controller-error stratum every row shares one inclusion
    probability, so the conditional oracle gap is a simple random sample of
    that case's controller-error voxels. Rates that mix strata use the stored
    inverse-inclusion weights. Neither estimate treats voxels as patients.
    """

    correct = np.asarray(arrays["expert_correct"], dtype=bool)
    available = np.asarray(arrays["expert_available"], dtype=bool)
    weights = np.asarray(arrays["controller_weights"], dtype=np.float64)
    controller_correct = np.asarray(arrays["controller_correct"], dtype=bool).reshape(-1)
    sample_weight = np.asarray(arrays["sample_weight"], dtype=np.float64).reshape(-1)
    target = np.asarray(arrays["target_class"]).reshape(-1)
    features = {
        name: np.asarray(arrays["X"])[:, index]
        for index, name in enumerate(feature_names)
    }
    if correct.shape != available.shape or weights.shape != correct.shape:
        raise ValueError("expert correctness, availability, and weights are misaligned")
    if correct.shape[1] != len(expert_order):
        raise ValueError("expert axis does not match the frozen expert order")
    if np.any(correct & ~available):
        raise ValueError("an unavailable expert was marked correct")

    oracle = correct.any(1)
    all_hard_wrong = ~oracle
    controller_wrong = ~controller_correct
    any_expert_wrong = (available & ~correct).any(1)
    correct_mass = (weights * correct).sum(1)
    incorrect_mass = (weights * (available & ~correct)).sum(1)
    hard_oracle_class_available = oracle
    mixture_only = controller_correct & all_hard_wrong
    needs_new_evidence = controller_wrong & all_hard_wrong
    selection_headroom = controller_wrong & oracle

    def conditional(event: np.ndarray, condition: np.ndarray) -> dict[str, object]:
        selected = np.asarray(condition, dtype=bool).reshape(-1)
        event = np.asarray(event, dtype=bool).reshape(-1)[selected]
        weight = sample_weight[selected]
        unique_weights = np.unique(np.round(weight, 6)) if weight.size else np.asarray([])
        return {
            "condition_count": int(selected.sum()),
            "event_count": int(event.sum()) if event.size else 0,
            "sampled_fraction": float(event.mean()) if event.size else None,
            "inclusion_weighted_fraction": _weighted_mean(event, weight),
            "constant_inclusion_weight": bool(unique_weights.size <= 1),
            "estimand": (
                "simple random sample of this condition"
                if unique_weights.size <= 1
                else "inverse-inclusion weighted across sampling strata"
            ),
        }

    regions = {
        name: conditional(selection_headroom, np.isin(target, labels) & controller_wrong)
        for name, labels in REGION_LABELS.items()
    }
    disagreement = features["js_disagreement"] >= high_disagreement
    entropy = features["predictive_entropy"] >= high_entropy
    boundary = features["boundary_probability"] >= boundary_probability
    return {
        "definition": (
            "An available expert is correct only when its hard class matches "
            "ground truth. The oracle uses labels retrospectively and is not "
            "an inference input or a trained controller."
        ),
        "rows": int(controller_correct.size),
        "controller_correct": _rates(controller_correct, sample_weight),
        "selection_headroom_when_controller_wrong": conditional(
            selection_headroom, controller_wrong
        ),
        "controller_corrects_any_available_expert": conditional(
            controller_correct & any_expert_wrong, np.ones_like(controller_correct)
        ),
        "controller_overrides_available_correct_expert": conditional(
            selection_headroom, np.ones_like(controller_correct)
        ),
        "mean_weight_on_correct_experts": {
            "sampled_mean": float(correct_mass.mean()),
            "inclusion_weighted_mean": _weighted_mean(correct_mass, sample_weight),
        },
        "mean_weight_on_incorrect_available_experts": {
            "sampled_mean": float(incorrect_mass.mean()),
            "inclusion_weighted_mean": _weighted_mean(incorrect_mass, sample_weight),
        },
        "hard_selection_cannot_repair": conditional(
            needs_new_evidence, controller_wrong
        ),
        "mixture_correct_when_every_hard_expert_wrong": conditional(
            mixture_only, all_hard_wrong
        ),
        "controller_error_census": {
            "definition": (
                "Controller-error rows. The census is complete when every such "
                "row has inclusion probability 1, which is true when the error "
                "stratum is below its sampling cap. Selection-correctable rows "
                "and rows with no correct hard expert partition this census."
            ),
            "error_rows": int(controller_wrong.sum()),
            "selection_correctable_rows": int(selection_headroom.sum()),
            "no_correct_hard_expert_rows": int(needs_new_evidence.sum()),
            "complete": bool(
                controller_wrong.any()
                and np.allclose(
                    sample_weight[controller_wrong], 1.0, rtol=0, atol=1e-5
                )
            ),
        },
        "controller_correct_when_all_hard_experts_wrong": {
            "definition": (
                "The controller argmax matches ground truth while every "
                "available expert hard class does not. These rows are outside "
                "the controller-error census. The sample event count is a "
                "full-volume total only when every controller-correct row has "
                "inclusion probability 1."
            ),
            "sample_event_count": int(mixture_only.sum()),
            "disjoint_from_controller_errors": True,
            "full_volume_count_known": bool(
                controller_correct.any()
                and np.allclose(
                    sample_weight[controller_correct], 1.0, rtol=0, atol=1e-5
                )
            ),
            "inclusion_weighted_fraction_of_sampled_rows": _weighted_mean(
                mixture_only, sample_weight
            ),
        },
        "by_region_selection_headroom_when_controller_wrong": regions,
        "by_disagreement_selection_headroom_when_controller_wrong": {
            "high": conditional(selection_headroom, controller_wrong & disagreement),
            "low": conditional(selection_headroom, controller_wrong & ~disagreement),
        },
        "by_entropy_selection_headroom_when_controller_wrong": {
            "high": conditional(selection_headroom, controller_wrong & entropy),
            "low": conditional(selection_headroom, controller_wrong & ~entropy),
        },
        "by_boundary_selection_headroom_when_controller_wrong": {
            "boundary": conditional(selection_headroom, controller_wrong & boundary),
            "non_boundary": conditional(
                selection_headroom, controller_wrong & ~boundary
            ),
        },
        "weight_exceeds_incorrect_mass": {
            "sampled_fraction": float((correct_mass > incorrect_mass).mean()),
            "inclusion_weighted_fraction": _weighted_mean(
                correct_mass > incorrect_mass, sample_weight
            ),
        },
        "sampling_note": (
            "Raw sampled error prevalence is inflated by stratified sampling. "
            "Conditional rates inside the controller-error sample use equal "
            "inclusion probability. Cross-stratum rates use inverse-inclusion "
            "weights. Voxels are not independent patients."
        ),
        "expert_order": list(expert_order),
        "hard_oracle_available": _rates(hard_oracle_class_available, sample_weight),
    }


def pooled_case_mean(case_summaries: Sequence[Mapping[str, object]], key: tuple[str, ...]):
    values = []
    for summary in case_summaries:
        value = summary
        for part in key:
            value = value[part]
        if value is not None:
            values.append(float(value))
    if not values:
        return None
    return {
        "cases": len(values),
        "unweighted_case_mean": float(np.mean(values)),
        "case_min": float(np.min(values)),
        "case_max": float(np.max(values)),
    }
