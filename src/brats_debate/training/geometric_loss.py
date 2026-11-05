"""Distance-weighted geometric term for nested BraTS regions.

This is the training surrogate

    L_dist = sum_v p(v) d(v, G)^2
             + sum_v g(v) (1 - p(v)) d(v, boundary(G))^2
             all over sum_v (p(v) + g(v)) + eps

applied separately to WT, TC, and ET and then averaged.

d(v, G) is the Euclidean distance in millimeters to the reference
region. It is zero inside the region. d(v, boundary(G)) is the
distance in millimeters to the reference surface. Both are
precomputed from the full-volume reference mask and then cropped
onto training windows. They do not depend on the prediction, so
the backward pass does not threshold the network output.

An empty reference region has no surface. d(v, G) is then the
fixed cap at every voxel, so any positive prediction is a false
positive, and a zero prediction has zero loss. A distance of zero
on the surface itself means a false negative that lands exactly
on a boundary voxel is not charged by the second term; interior
misses are.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

DISTANCE_CAP_MM = 32.0
EPSILON = 1e-6
REGION_LABELS = (("WT", (1, 2, 3)), ("TC", (1, 3)), ("ET", (3,)))


def reference_distances(mask: np.ndarray, spacing, cap_mm: float = DISTANCE_CAP_MM):
    """Return (distance to the set, distance to the surface), both in millimeters."""

    mask = np.asarray(mask, dtype=bool)
    spacing = tuple(float(value) for value in spacing)
    if len(spacing) != mask.ndim:
        raise ValueError("spacing must have one value per spatial axis")
    if not mask.any():
        capped = np.full(mask.shape, cap_mm, dtype=np.float32)
        return capped, capped.copy()
    outside = ~mask
    distance_outside = ndimage.distance_transform_edt(outside, sampling=spacing)
    distance_to_set = np.where(mask, 0.0, distance_outside).astype(np.float32)
    eroded = ndimage.binary_erosion(mask, border_value=0)
    boundary = mask if not eroded.any() else mask & ~eroded
    distance_to_boundary = ndimage.distance_transform_edt(~boundary, sampling=spacing).astype(np.float32)
    np.minimum(distance_to_set, cap_mm, out=distance_to_set)
    np.minimum(distance_to_boundary, cap_mm, out=distance_to_boundary)
    return distance_to_set, distance_to_boundary


def geometric_region_loss(probability, reference, distance_to_set, distance_to_boundary, eps: float = EPSILON):
    """One region. probability and reference are [N, D, H, W]. Distances are constants."""

    probability = probability.float().clamp(0, 1)
    reference = reference.float()
    distance_to_set = distance_to_set.float()
    distance_to_boundary = distance_to_boundary.float()
    false_positive = probability * distance_to_set.square()
    false_negative = reference * (1 - probability) * distance_to_boundary.square()
    numerator = (false_positive + false_negative).sum(dim=(1, 2, 3))
    denominator = (probability + reference).sum(dim=(1, 2, 3)).clamp_min(eps)
    return (numerator / denominator).mean()


def region_probability(probabilities, labels: tuple[int, ...]):
    """Sum mutually exclusive class probabilities into one nested region probability."""

    if probabilities.shape[1] < 4:
        raise ValueError("region probabilities expect four classes")
    selected = probabilities[:, list(labels)].sum(dim=1)
    return selected


def geometric_loss(probabilities, target, distance_to_set, distance_to_boundary, eps: float = EPSILON):
    """Average the WT, TC, and ET geometric terms.

    distance tensors are [N, 3, D, H, W] in the order WT, TC, ET.
    """

    if distance_to_set.shape[1] != 3 or distance_to_boundary.shape[1] != 3:
        raise ValueError("distance tensors must have three regions")
    terms = []
    for index, (_name, labels) in enumerate(REGION_LABELS):
        probability = region_probability(probabilities, labels)
        reference = torch_region(target, labels)
        terms.append(
            geometric_region_loss(
                probability,
                reference,
                distance_to_set[:, index],
                distance_to_boundary[:, index],
                eps,
            )
        )
    return sum(terms) / len(terms)


def torch_region(target, labels: tuple[int, ...]):
    reference = target == labels[0]
    for label in labels[1:]:
        reference = reference | (target == label)
    return reference.float()
