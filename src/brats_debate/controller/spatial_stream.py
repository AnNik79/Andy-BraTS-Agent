"""Halo-aware streaming for the spatial controller pilot.

Pointwise slabs can be cut anywhere. A 3x3x3 gate cannot: a slab that
stops at its own edge would replace a real neighbor with the convolution's
zero padding. Streaming therefore reads one extra voxel on each open side
and writes only the voxels whose full neighborhood was inside the read.

Training windows use the same rule. The saved tensor is a real
``interior + 2 * halo`` neighborhood, and the loss is applied to the
interior after the gate's own padding. Out-of-volume voxels are not
invented by zeroing MRI and then rebuilding features.
"""

from __future__ import annotations

import time

import torch

from ..config import SEGMENTATION_EXPERTS
from ..debate.features import controller_features, stack_evidence
from ..uncertainty.disagreement import analyze_uncertainty
from ..uncertainty.extraction import _slice_outputs


def _box(origin, shape, interior, halo):
    origin = tuple(int(value) for value in origin)
    if len(origin) != 3 or interior < 1 or halo < 0:
        raise ValueError("a spatial window needs a 3D origin, a positive interior, and a halo")
    src_start = tuple(max(0, origin[axis] - halo) for axis in range(3))
    src_stop = tuple(min(int(shape[axis]), origin[axis] + interior + halo) for axis in range(3))
    for axis in range(3):
        if not 0 <= origin[axis] and origin[axis] + interior <= shape[axis]:
            raise ValueError("the scored interior must lie inside the volume")
        if src_stop[axis] <= src_start[axis]:
            raise ValueError("the spatial read is empty")
    return src_start, src_stop


def window_fits(origin, shape, interior, halo) -> bool:
    """True when the interior and its entire halo are real voxels."""

    return all(
        int(origin[axis]) >= halo
        and int(origin[axis]) + interior + halo <= int(shape[axis])
        for axis in range(3)
    )


def build_gate_inputs(image, outputs):
    """Voxelwise controller inputs. The image and expert maps must already be the intended crop."""

    debate = analyze_uncertainty(outputs)["production"]
    features = controller_features(image, outputs, debate)
    probabilities, availability = stack_evidence(outputs)
    return features, probabilities, availability


def real_window(image, outputs, origin, interior, halo):
    """Return gate inputs for a neighborhood that is fully inside the volume."""

    shape = tuple(int(size) for size in image.shape[-3:])
    if not window_fits(origin, shape, interior, halo):
        raise ValueError("refusing a training window whose halo would leave the volume")
    src_start, src_stop = _box(origin, shape, interior, halo)
    spatial = tuple(slice(src_start[axis], src_stop[axis]) for axis in range(3))
    features, probabilities, availability = build_gate_inputs(
        image[(..., *spatial)],
        _slice_outputs(outputs, spatial),
    )
    expected = interior + 2 * halo
    if features.shape[-3:] != (expected, expected, expected):
        raise ValueError("the real window does not have the expected halo")
    return features, probabilities, availability


def interior_prediction(module, features, probabilities, availability, halo):
    """Run a gate and drop the halo ring that padding, not anatomy, would affect."""

    result = module(features, probabilities, availability)
    if halo == 0:
        return result
    sl = (slice(halo, -halo), slice(halo, -halo), slice(halo, -halo))
    return {
        "weights": result["weights"][(..., *sl)],
        "probabilities": result["probabilities"][(..., *sl)],
    }


@torch.inference_mode()
def stream_gates(modules, image, outputs, cfg, device, slab_depth=8, halo=1):
    """Apply one or more gates to a volume without storing a 65-channel feature volume.

    ``modules`` maps a name to a gate. Every gate is run on the same slab.
    The read includes ``halo`` voxels beyond each open slab face. Pointwise
    gates ignore that extra context; a 3x3x3 gate needs it.
    """

    if image.ndim != 5 or image.shape[0] != 1:
        raise ValueError("streaming expects one [1, modality, D, H, W] image")
    if slab_depth < 1 or halo < 0:
        raise ValueError("slab depth must be positive and halo cannot be negative")
    if not modules:
        raise ValueError("at least one gate is required")
    shape = tuple(int(size) for size in image.shape[2:])
    depth = shape[0]
    classes = len(cfg["label_mapping"])
    experts = len(SEGMENTATION_EXPERTS)
    labels = {
        name: torch.empty(shape, dtype=torch.uint8) for name in modules
    }
    weight_variation = {name: [] for name in modules}
    timing = {name: 0.0 for name in modules}
    timing["features"] = 0.0
    max_prob_error = {name: 0.0 for name in modules}
    max_weight_error = {name: 0.0 for name in modules}
    max_unavailable_weight = {name: 0.0 for name in modules}
    for module in modules.values():
        module.to(device).eval()
    try:
        start = 0
        while start < depth:
            stop = min(start + slab_depth, depth)
            src_start = max(0, start - halo)
            src_stop = min(depth, stop + halo)
            if stop < depth and src_stop != stop + halo:
                raise RuntimeError("an open slab face is missing its real neighbor")
            if start > 0 and src_start != start - halo:
                raise RuntimeError("an open slab face is missing its real neighbor")
            spatial = (slice(src_start, src_stop), slice(None), slice(None))
            built = time.perf_counter()
            features, probabilities, availability = build_gate_inputs(
                image[(..., *spatial)],
                _slice_outputs(outputs, spatial),
            )
            timing["features"] += time.perf_counter() - built
            offset = start - src_start
            length = stop - start
            dest = (slice(start, stop), slice(None), slice(None))
            src = (slice(offset, offset + length), slice(None), slice(None))
            device_features = features.to(device)
            device_probabilities = probabilities.to(device)
            device_availability = availability.to(device)
            for name, module in modules.items():
                began = time.perf_counter()
                result = module(device_features, device_probabilities, device_availability)
                probs = result["probabilities"].detach().cpu()
                weights = result["weights"].detach().cpu()
                timing[name] += time.perf_counter() - began
                written_probs = probs[(..., *src)]
                written_weights = weights[(..., *src)]
                labels[name][dest] = written_probs.argmax(1)[0].to(torch.uint8)
                max_prob_error[name] = max(
                    max_prob_error[name],
                    float((written_probs.sum(1) - 1).abs().max()),
                )
                max_weight_error[name] = max(
                    max_weight_error[name],
                    float((written_weights.sum(1) - 1).abs().max()),
                )
                unavailable = availability.detach().cpu()[(..., *src)] <= 0
                if bool(unavailable.any()):
                    max_unavailable_weight[name] = max(
                        max_unavailable_weight[name],
                        float(written_weights.masked_select(unavailable).abs().max()),
                    )
                weight_variation[name].append(_neighbor_abs(written_weights))
                del result, probs, weights, written_probs, written_weights
            del features, probabilities, availability, device_features, device_probabilities, device_availability
            start = stop
    finally:
        for module in modules.values():
            module.cpu()
    return labels, {
        "forward_seconds": {name: timing[name] for name in modules},
        "feature_seconds": timing["features"],
        "max_probability_sum_error": max_prob_error,
        "max_weight_sum_error": max_weight_error,
        "max_unavailable_expert_weight": max_unavailable_weight,
        "mean_neighbor_weight_abs_difference": {
            name: float(torch.stack(values).mean()) if values else None
            for name, values in weight_variation.items()
        },
    }


def _neighbor_abs(weights: torch.Tensor) -> torch.Tensor:
    """Mean absolute weight change across face-adjacent voxels in one slab."""

    differences = []
    spatial = weights[0]
    for axis in range(1, 4):
        if spatial.shape[axis] < 2:
            continue
        left = spatial.narrow(axis, 0, spatial.shape[axis] - 1)
        right = spatial.narrow(axis, 1, spatial.shape[axis] - 1)
        differences.append((left - right).abs().mean())
    if not differences:
        return weights.new_zeros(())
    return torch.stack(differences).mean()
