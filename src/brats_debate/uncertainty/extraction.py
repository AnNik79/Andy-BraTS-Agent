"""Bounded, resumable sampled-voxel extraction for Stage B."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping

import numpy as np
import torch

from ..config import SEGMENTATION_EXPERTS, fingerprint, protocol
from ..debate.features import controller_features, stack_evidence
from ..inference.patches import sliding_predict
from .disagreement import analyze_uncertainty
from .error_prediction import construct_error_target, inverse_probability_weights


CASE_ARTIFACT_FORMAT = "beta2_error_prediction_case_v1"
FEATURE_SCHEMA_VERSION = "beta2_error_features_v1"
STRATUM_NAMES = (
    "error",
    "boundary_correct",
    "foreground_correct",
    "other_correct",
)


@dataclass(frozen=True)
class VoxelSamplingPolicy:
    """Maximum sampled voxels per mutually exclusive stratum."""

    seed: int = 20261012
    error: int = 20000
    boundary_correct: int = 10000
    foreground_correct: int = 20000
    other_correct: int = 30000

    def __post_init__(self):
        if min(
            self.error,
            self.boundary_correct,
            self.foreground_correct,
            self.other_correct,
        ) < 1:
            raise ValueError("every voxel sampling cap must be positive")

    @property
    def maximum_voxels(self) -> int:
        return (
            self.error
            + self.boundary_correct
            + self.foreground_correct
            + self.other_correct
        )


def _patient_seed(seed: int, patient_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{patient_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def sample_voxels(
    *,
    patient_id: str,
    error: np.ndarray,
    boundary: np.ndarray,
    foreground: np.ndarray,
    domain: np.ndarray,
    policy: VoxelSamplingPolicy,
) -> dict[str, np.ndarray | dict[str, object]]:
    """Sample disjoint strata and retain exact inclusion probabilities."""

    arrays = [
        np.asarray(value, dtype=bool)
        for value in (error, boundary, foreground, domain)
    ]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("sampling masks must share one voxel grid")
    error, boundary, foreground, domain = arrays
    if not domain.any():
        raise ValueError("sampling domain is empty")
    if (error & ~domain).any():
        raise ValueError("sampling domain must contain every evaluated error voxel")

    masks = (
        domain & error,
        domain & ~error & boundary,
        domain & ~error & ~boundary & foreground,
        domain & ~error & ~boundary & ~foreground,
    )
    caps = (
        policy.error,
        policy.boundary_correct,
        policy.foreground_correct,
        policy.other_correct,
    )
    rng = np.random.default_rng(_patient_seed(policy.seed, patient_id))
    selected, probabilities, strata, summary = [], [], [], {}
    for stratum_id, (name, mask, cap) in enumerate(
        zip(STRATUM_NAMES, masks, caps)
    ):
        candidates = np.flatnonzero(mask.reshape(-1))
        count = int(candidates.size)
        take = min(count, int(cap))
        if take:
            chosen = (
                candidates
                if take == count
                else rng.choice(candidates, size=take, replace=False)
            )
            selected.append(np.asarray(chosen, dtype=np.int64))
            probability = take / count
            probabilities.append(np.full(take, probability, dtype=np.float64))
            strata.append(np.full(take, stratum_id, dtype=np.uint8))
        summary[name] = {
            "population_voxels": count,
            "sampled_voxels": take,
            "inclusion_probability": (take / count if count else None),
        }
    if not selected:
        raise ValueError("sampling policy selected no voxels")
    flat_index = np.concatenate(selected)
    inclusion_probability = np.concatenate(probabilities)
    stratum = np.concatenate(strata)
    order = rng.permutation(flat_index.size)
    return {
        "flat_index": flat_index[order],
        "inclusion_probability": inclusion_probability[order],
        "stratum": stratum[order],
        "summary": summary,
    }


def _scalar_volume(value, shape, name: str) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    array = np.asarray(value, dtype=np.float32)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.shape != tuple(shape):
        raise ValueError(f"{name} does not align with the patient voxel grid")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    from scipy import ndimage

    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask.copy()
    return mask ^ ndimage.binary_erosion(mask, iterations=1, border_value=0)


def _compact_expert_output(name, output):
    extra = output.get("extra") or {}
    required_extra = ("margin", "availability")
    missing = [key for key in required_extra if key not in extra]
    if name == "boundary":
        missing.extend(
            key
            for key in ("boundary_probability", "distance_to_boundary")
            if key not in extra
        )
    if missing:
        raise ValueError(f"expert {name} is missing Stage B tensors {missing}")
    kept_extra = {key: extra[key] for key in required_extra}
    if name == "boundary":
        kept_extra.update(
            boundary_probability=extra["boundary_probability"],
            distance_to_boundary=extra["distance_to_boundary"],
        )
    compact = {
        key: output[key]
        for key in ("probabilities", "uncertainty", "confidence", "features")
    }
    compact["extra"] = kept_extra
    return compact


def compact_expert_outputs(
    outputs: Mapping[str, Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    """Keep only exact tensors needed by the Stage B gate and feature schema."""

    if tuple(outputs) != tuple(SEGMENTATION_EXPERTS):
        raise ValueError("all four frozen experts in canonical order are required")
    return {
        name: _compact_expert_output(name, outputs[name])
        for name in SEGMENTATION_EXPERTS
    }


@torch.inference_mode()
def predict_experts_compact(patient, models, cfg, device):
    """Production-equivalent expert inference with Stage B-only early release.

    Hard segmentations for the first three experts remain live only until the
    unchanged HighRes proposal has been formed. Rich logits and unused extras
    are released immediately after each expert.
    """

    if tuple(models) != tuple(SEGMENTATION_EXPERTS):
        raise ValueError("compact inference requires canonical four-expert order")
    image = torch.from_numpy(patient["image"])[None]
    outputs, proposal_segmentations = {}, []
    inference = cfg["inference"]
    for name, model in models.items():
        proposal = None
        if name == "highres":
            if len(outputs) != 3 or len(proposal_segmentations) != 3:
                raise ValueError(
                    "HighRes proposal inference requires CNN, transformer, boundary first"
                )
            votes = torch.stack(
                [(segmentation > 0) for segmentation in proposal_segmentations]
            ).any(0)[:, None]
            uncertain = (
                torch.stack(
                    [output["uncertainty"] for output in outputs.values()]
                ).mean(0)
                > inference["uncertainty_threshold"]
            )
            proposal = (votes | uncertain).float()
            proposal_segmentations.clear()
        model.to(device)
        try:
            rich = sliding_predict(
                model,
                image,
                (
                    cfg["highres_patch_size"]
                    if name == "highres"
                    else cfg["crop_size"]
                ),
                device,
                inference["overlap"],
                inference["mc_samples"],
                proposal,
                (
                    inference["highres_max_patches"]
                    if name == "highres"
                    else None
                ),
            )
        finally:
            model.cpu()
        if name != "highres":
            proposal_segmentations.append(rich["segmentation"])
        outputs[name] = _compact_expert_output(name, rich)
        del rich
    return outputs


def _slice_outputs(outputs, spatial_slices):
    return {
        name: {
            **{
                key: value[(..., *spatial_slices)]
                for key, value in output.items()
                if key != "extra"
            },
            "extra": {
                key: value[(..., *spatial_slices)]
                for key, value in output["extra"].items()
            },
        }
        for name, output in outputs.items()
    }


@torch.inference_mode()
def stream_controller_and_uncertainty(
    controller,
    image: torch.Tensor,
    outputs: Mapping[str, Mapping[str, object]],
    cfg: Mapping[str, object],
    device: torch.device,
    *,
    slab_depth: int = 8,
    return_weights: bool = False,
) -> tuple[torch.Tensor, dict[str, object], torch.Tensor | None]:
    """Gate independent slabs and retain only final, JS, and entropy volumes.

    The existing Stage A formulas and frozen controller are called on each
    slab. All reductions are over expert/class axes, so slab partitioning does
    not change the voxel-local mathematics.
    """

    if image.ndim != 5 or image.shape[0] != 1 or slab_depth < 1:
        raise ValueError("streaming requires one [1,C,D,H,W] image and a positive slab")
    compact = compact_expert_outputs(outputs)
    shape = tuple(image.shape[2:])
    classes = len(cfg["label_mapping"])
    final = torch.empty((1, classes, *shape), dtype=torch.float32)
    js = torch.empty((1, 1, *shape), dtype=torch.float32)
    entropy = torch.empty((1, 1, *shape), dtype=torch.float32)
    weights = (
        torch.empty((1, len(SEGMENTATION_EXPERTS), *shape), dtype=torch.float32)
        if return_weights
        else None
    )
    maximum_residual = 0.0
    maximum_production_js_difference = 0.0
    maximum_controller_reconstruction_residual = 0.0
    maximum_probability_sum_residual = 0.0
    debate_seconds = controller_seconds = 0.0
    controller.to(device)
    try:
        for start in range(0, shape[0], slab_depth):
            spatial_slices = (
                slice(start, min(start + slab_depth, shape[0])),
                slice(None),
                slice(None),
            )
            local = _slice_outputs(compact, spatial_slices)
            debate_start = time.perf_counter()
            uncertainty = analyze_uncertainty(local)
            debate = uncertainty["production"]
            debate_seconds += time.perf_counter() - debate_start

            controller_start = time.perf_counter()
            features = controller_features(
                image[(..., *spatial_slices)], local, debate
            )
            probabilities, availability = stack_evidence(local)
            device_features = features.to(device)
            device_probabilities = probabilities.to(device)
            device_availability = availability.to(device)
            result = controller(
                device_features,
                device_probabilities,
                device_availability,
            )
            reconstructed = (
                device_probabilities * result["weights"].unsqueeze(2)
            ).sum(1)
            final[(..., *spatial_slices)] = result["probabilities"].cpu()
            if weights is not None:
                weights[(..., *spatial_slices)] = result["weights"].cpu()
            controller_seconds += time.perf_counter() - controller_start

            js[(..., *spatial_slices)] = uncertainty["js_divergence"]
            entropy[(..., *spatial_slices)] = uncertainty[
                "mean_predictive_entropy"
            ]
            maximum_residual = max(
                maximum_residual,
                float(
                    uncertainty["js_identity_residual"].max().detach().cpu()
                ),
            )
            maximum_production_js_difference = max(
                maximum_production_js_difference,
                float(
                    (
                        uncertainty["js_divergence"]
                        - debate["js_divergence"]
                    )
                    .abs()
                    .max()
                    .detach()
                    .cpu()
                ),
            )
            maximum_controller_reconstruction_residual = max(
                maximum_controller_reconstruction_residual,
                float(
                    (result["probabilities"] - reconstructed)
                    .abs()
                    .max()
                    .detach()
                    .cpu()
                ),
            )
            maximum_probability_sum_residual = max(
                maximum_probability_sum_residual,
                float(
                    (result["probabilities"].sum(1) - 1)
                    .abs()
                    .max()
                    .detach()
                    .cpu()
                ),
            )
            del (
                reconstructed,
                result,
                probabilities,
                availability,
                features,
                device_availability,
                device_probabilities,
                device_features,
                debate,
                uncertainty,
                local,
            )
    finally:
        controller.cpu()
    return (
        final,
        {
            "js_disagreement": js,
            "predictive_entropy": entropy,
            "js_identity_max_residual": maximum_residual,
            "equivalence": {
                "stage_a_vs_production_js_max_abs_difference": (
                    maximum_production_js_difference
                ),
                "controller_reconstruction_max_abs_difference": (
                    maximum_controller_reconstruction_residual
                ),
                "probability_sum_max_abs_residual": (
                    maximum_probability_sum_residual
                ),
            },
            "host_timing_seconds": {
                "debate_and_uncertainty": debate_seconds,
                "controller": controller_seconds,
            },
        },
        weights,
    )


def extract_sampled_case(
    patient: Mapping[str, object],
    outputs: Mapping[str, Mapping[str, object]],
    debate: Mapping[str, object] | None,
    final_probabilities,
    cfg: Mapping[str, object],
    *,
    subject_id: str,
    policy: VoxelSamplingPolicy,
    uncertainty_maps: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create a compact sampled representation and discard no provenance."""

    image = np.asarray(patient["image"], dtype=np.float32)
    target = np.asarray(patient["label"])
    if image.ndim != 4 or target.shape != image.shape[1:]:
        raise ValueError("MRI and ground truth are not aligned")
    if tuple(outputs) != tuple(SEGMENTATION_EXPERTS):
        raise ValueError("all four frozen experts in canonical order are required")

    try:
        import torch

        final = (
            final_probabilities.detach().cpu().numpy()
            if torch.is_tensor(final_probabilities)
            else np.asarray(final_probabilities)
        )
    except ImportError:
        final = np.asarray(final_probabilities)
    if final.shape != (1, len(cfg["label_mapping"]), *target.shape):
        raise ValueError("frozen controller probabilities are not grid-aligned")
    if not np.isfinite(final).all() or not np.allclose(
        final.sum(1), 1, atol=1e-4, rtol=0
    ):
        raise ValueError("frozen controller probabilities are not a finite simplex")
    prediction = final.argmax(1)[0].astype(np.int16)
    error = construct_error_target(prediction, target).astype(bool)

    if uncertainty_maps is None:
        if debate is None:
            raise ValueError("legacy extraction requires the production debate")
        uncertainty = analyze_uncertainty(outputs)
        js_source = uncertainty["js_divergence"]
        entropy_source = uncertainty["mean_predictive_entropy"]
        maximum_residual = float(
            uncertainty["js_identity_residual"].max().detach().cpu()
        )
        production_js = _scalar_volume(
            debate["js_divergence"], target.shape, "production JS disagreement"
        )
    else:
        js_source = uncertainty_maps["js_disagreement"]
        entropy_source = uncertainty_maps["predictive_entropy"]
        maximum_residual = float(
            uncertainty_maps["js_identity_max_residual"]
        )
        production_js = None
    js = _scalar_volume(js_source, target.shape, "JS disagreement")
    entropy = _scalar_volume(
        entropy_source, target.shape, "predictive entropy"
    )
    if production_js is not None and not np.allclose(
        js, production_js, atol=1e-6, rtol=0
    ):
        raise ValueError("Stage A and production JS disagreement do not match")
    if (js < -1e-6).any() or (js > 1 + 1e-6).any():
        raise ValueError("normalized JS disagreement is outside [0,1]")
    if (entropy < -1e-6).any() or (entropy > 1 + 1e-6).any():
        raise ValueError("normalized predictive entropy is outside [0,1]")

    boundary_probability = _scalar_volume(
        outputs["boundary"]["extra"]["boundary_probability"],
        target.shape,
        "boundary probability",
    )
    boundary_distance = (
        _scalar_volume(
            outputs["boundary"]["extra"]["distance_to_boundary"],
            target.shape,
            "boundary distance",
        )
        / 16.0
    )
    feature_names = (
        "js_disagreement",
        "predictive_entropy",
        "boundary_probability",
        "boundary_distance",
        *(f"mri_{name}" for name in cfg["modalities"]),
    )
    feature_volumes = (
        js,
        entropy,
        boundary_probability,
        boundary_distance,
        *(image[index] for index in range(image.shape[0])),
    )
    if any(
        volume.shape != target.shape or not np.isfinite(volume).all()
        for volume in feature_volumes
    ):
        raise ValueError("Stage B feature maps are not finite and aligned")

    whole_tumor = np.isin(target, cfg["regions"]["WT"])
    target_boundary = _binary_boundary(whole_tumor)
    foreground = (target > 0) | (prediction > 0)
    domain = (np.abs(image).sum(0) > 0) | foreground
    sampled = sample_voxels(
        patient_id=str(patient["patient_id"]),
        error=error,
        boundary=target_boundary,
        foreground=foreground,
        domain=domain,
        policy=policy,
    )
    flat_index = sampled["flat_index"]
    X = np.stack(
        [volume.reshape(-1)[flat_index] for volume in feature_volumes], axis=1
    ).astype(np.float32)
    y = error.reshape(-1)[flat_index].astype(np.uint8)
    inclusion_probability = np.asarray(
        sampled["inclusion_probability"], dtype=np.float32
    )
    sample_weight = inverse_probability_weights(
        inclusion_probability, normalize=False
    ).astype(np.float32)
    if not np.isfinite(X).all() or not np.isfinite(sample_weight).all():
        raise ValueError("sampled Stage B data contain non-finite values")

    domain_voxels = int(domain.sum())
    domain_errors = int((error & domain).sum())
    return {
        "arrays": {
            "X": X,
            "y": y,
            "flat_index": np.asarray(flat_index, dtype=np.int64),
            "inclusion_probability": inclusion_probability,
            "sample_weight": sample_weight,
            "stratum": np.asarray(sampled["stratum"], dtype=np.uint8),
        },
        "metadata": {
            "format": CASE_ARTIFACT_FORMAT,
            "complete": True,
            "patient_id": str(patient["patient_id"]),
            "subject_id": str(subject_id),
            "source_split": "validation",
            "exploratory": True,
            "feature_schema": FEATURE_SCHEMA_VERSION,
            "feature_names": list(feature_names),
            "spatial_shape": list(target.shape),
            "class_order": sorted(int(value) for value in cfg["label_mapping"].values()),
            "sample_count": int(X.shape[0]),
            "sampling": {
                "policy": asdict(policy),
                "maximum_voxels": policy.maximum_voxels,
                "strata": sampled["summary"],
                "inverse_probability_weighted": True,
            },
            "full_domain_voxels": domain_voxels,
            "full_error_voxels": domain_errors,
            "full_error_prevalence": (
                domain_errors / domain_voxels if domain_voxels else None
            ),
            "sample_error_prevalence": float(y.mean()),
            "weighted_sample_error_prevalence": float(
                np.average(y, weights=sample_weight)
            ),
            "js_identity_max_residual": maximum_residual,
            "controller_prediction_labels": sorted(
                int(value) for value in np.unique(prediction)
            ),
            "target_labels": sorted(int(value) for value in np.unique(target)),
        },
    }


def case_provenance(
    cfg: Mapping[str, object],
    splits: Mapping[str, object],
    *,
    expert_hashes: Mapping[str, str],
    controller_hash: str,
    policy: VoxelSamplingPolicy,
    code_commit: str,
) -> dict[str, object]:
    return {
        "format": CASE_ARTIFACT_FORMAT,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "source_split": "validation",
        "exploratory": True,
        "protocol_hash": fingerprint(protocol(cfg)),
        "split_hash": fingerprint(splits),
        "expert_hashes": dict(expert_hashes),
        "controller_hash": str(controller_hash),
        "inference": {
            "overlap": cfg["inference"]["overlap"],
            "mc_samples": cfg["inference"]["mc_samples"],
            "uncertainty_threshold": cfg["inference"]["uncertainty_threshold"],
            "highres_max_patches": cfg["inference"]["highres_max_patches"],
        },
        "sampling_policy": asdict(policy),
        "code_commit": str(code_commit),
    }


def _metadata_from_npz(archive) -> dict[str, object]:
    if "metadata_json" not in archive:
        raise ValueError("Stage B artifact has no metadata")
    return json.loads(str(archive["metadata_json"].item()))


def save_case_artifact(
    path,
    artifact: Mapping[str, object],
    *,
    provenance: Mapping[str, object],
) -> None:
    """Atomically save one compact case; never stores full-volume tensors."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **dict(artifact["metadata"]),
        "provenance": dict(provenance),
        "provenance_hash": fingerprint(provenance),
    }
    arrays = dict(artifact["arrays"])
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".partial.npz", dir=path.parent
    )
    os.close(descriptor)
    try:
        np.savez_compressed(
            temporary_name,
            **arrays,
            metadata_json=np.asarray(
                json.dumps(metadata, allow_nan=False, sort_keys=True)
            ),
        )
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_case_artifact(
    path,
    *,
    expected_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = _metadata_from_npz(archive)
        if (
            metadata.get("format") != CASE_ARTIFACT_FORMAT
            or not metadata.get("complete")
        ):
            raise ValueError("unknown or incomplete Stage B case artifact")
        if metadata.get("provenance_hash") != fingerprint(metadata["provenance"]):
            raise ValueError("Stage B artifact provenance fingerprint is invalid")
        if expected_provenance is not None and metadata["provenance"] != dict(
            expected_provenance
        ):
            raise ValueError("Stage B artifact provenance does not match")
        arrays = {
            name: np.asarray(archive[name])
            for name in (
                "X",
                "y",
                "flat_index",
                "inclusion_probability",
                "sample_weight",
                "stratum",
            )
        }
    if arrays["X"].shape[0] != arrays["y"].size:
        raise ValueError("Stage B artifact arrays are not aligned")
    samples = arrays["y"].size
    if (
        arrays["X"].ndim != 2
        or any(
            arrays[name].reshape(-1).size != samples
            for name in (
                "flat_index",
                "inclusion_probability",
                "sample_weight",
                "stratum",
            )
        )
    ):
        raise ValueError("Stage B artifact arrays have invalid shapes")
    spatial_shape = tuple(int(value) for value in metadata.get("spatial_shape", []))
    feature_names = list(metadata.get("feature_names") or [])
    if (
        len(spatial_shape) != 3
        or min(spatial_shape) < 1
        or arrays["X"].shape[1] != len(feature_names)
        or metadata.get("sample_count") != samples
    ):
        raise ValueError("Stage B artifact metadata does not match its arrays")
    flat_index = arrays["flat_index"].reshape(-1)
    probability = arrays["inclusion_probability"].reshape(-1)
    weight = arrays["sample_weight"].reshape(-1)
    stratum = arrays["stratum"].reshape(-1)
    if (
        not np.isfinite(arrays["X"]).all()
        or not np.isin(arrays["y"], [0, 1]).all()
        or (flat_index < 0).any()
        or (flat_index >= int(np.prod(spatial_shape))).any()
        or np.unique(flat_index).size != samples
        or not np.isfinite(probability).all()
        or (probability <= 0).any()
        or (probability > 1).any()
        or not np.isfinite(weight).all()
        or (weight <= 0).any()
        or not np.allclose(weight, 1 / probability, rtol=1e-5, atol=1e-6)
        or not np.isin(stratum, np.arange(len(STRATUM_NAMES))).all()
    ):
        raise ValueError("Stage B artifact contains invalid sampled values")
    return {"arrays": arrays, "metadata": metadata}


def case_artifact_is_complete(
    path,
    *,
    patient_id: str,
    expected_provenance: Mapping[str, object],
) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    try:
        artifact = load_case_artifact(
            path, expected_provenance=expected_provenance
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return artifact["metadata"].get("patient_id") == patient_id
