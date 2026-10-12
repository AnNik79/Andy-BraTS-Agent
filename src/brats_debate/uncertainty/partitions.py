"""Subject-grouped Stage B fit/calibration/evaluation partitions."""

from collections import Counter
from typing import Mapping, Sequence

import numpy as np


PARTITION_FORMAT = "beta2_error_partitions_v1"
PARTITION_ROLES = ("fit", "calibration", "evaluation")


def validate_subject_partitions(
    manifest: Mapping[str, object],
    case_to_subject: Mapping[str, str],
    *,
    forbidden_case_ids: Sequence[str] = (),
    forbidden_subject_ids: Sequence[str] = (),
) -> None:
    if manifest.get("format") != PARTITION_FORMAT:
        raise ValueError("unknown Stage B partition format")
    if manifest.get("source_split") != "validation":
        raise ValueError("Stage B exploratory partitions must come from validation")
    partitions = manifest.get("partitions")
    if not isinstance(partitions, Mapping) or set(partitions) != set(PARTITION_ROLES):
        raise ValueError(f"partitions must contain exactly {PARTITION_ROLES}")
    forbidden = set(forbidden_case_ids)
    forbidden_subjects = set(str(value) for value in forbidden_subject_ids)
    forbidden_subjects.update(
        case_to_subject[case_id]
        for case_id in forbidden
        if case_id in case_to_subject
    )
    all_cases = []
    subject_owner = {}
    for role in PARTITION_ROLES:
        cases = list(partitions[role])
        if not cases:
            raise ValueError(f"{role} partition is empty")
        if set(cases) & forbidden:
            raise RuntimeError("Beta 1 test cases are forbidden in Stage B partitions")
        for case_id in cases:
            if case_id not in case_to_subject:
                raise ValueError(f"unknown case in Stage B partition: {case_id}")
            subject = case_to_subject[case_id]
            if subject in forbidden_subjects:
                raise RuntimeError(
                    f"forbidden test subject {subject} entered Stage B partitions"
                )
            previous = subject_owner.setdefault(subject, role)
            if previous != role:
                raise RuntimeError(
                    f"subject leakage: {subject} occurs in {previous} and {role}"
                )
        all_cases.extend(cases)
    if len(all_cases) != len(set(all_cases)):
        raise RuntimeError("case leakage across Stage B partitions")
    expected = set(manifest.get("allowed_case_ids") or [])
    if expected and set(all_cases) != expected:
        raise ValueError("Stage B partitions do not cover the allowed cases exactly")


def make_subject_partitions(
    case_to_subject: Mapping[str, str],
    allowed_case_ids: Sequence[str],
    *,
    forbidden_case_ids: Sequence[str] = (),
    forbidden_subject_ids: Sequence[str] = (),
    seed: int = 20261012,
    fractions: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Assign complete subjects to exploratory validation-only roles."""

    fractions = (
        {"fit": 0.6, "calibration": 0.2, "evaluation": 0.2}
        if fractions is None
        else dict(fractions)
    )
    if set(fractions) != set(PARTITION_ROLES):
        raise ValueError(f"fractions must contain exactly {PARTITION_ROLES}")
    if any(value <= 0 for value in fractions.values()) or not np.isclose(
        sum(fractions.values()), 1
    ):
        raise ValueError("partition fractions must be positive and sum to one")
    allowed = list(allowed_case_ids)
    if not allowed or len(allowed) != len(set(allowed)):
        raise ValueError("allowed case IDs must be unique and nonempty")
    forbidden = set(forbidden_case_ids)
    forbidden_subjects = set(str(value) for value in forbidden_subject_ids)
    forbidden_subjects.update(
        case_to_subject[case_id]
        for case_id in forbidden
        if case_id in case_to_subject
    )
    if set(allowed) & forbidden:
        raise RuntimeError("Beta 1 test cases cannot enter Stage B partitions")
    missing = [case_id for case_id in allowed if case_id not in case_to_subject]
    if missing:
        raise ValueError(f"missing subject identity for cases including {missing[:3]}")

    subjects = sorted({case_to_subject[case_id] for case_id in allowed})
    overlap = set(subjects) & forbidden_subjects
    if overlap:
        raise RuntimeError(
            f"Beta 1 test subjects cannot enter Stage B partitions: {sorted(overlap)}"
        )
    if len(subjects) < len(PARTITION_ROLES):
        raise ValueError("at least three subjects are required for grouped partitions")
    shuffled = np.random.default_rng(seed).permutation(subjects).tolist()
    raw = np.array([fractions[role] for role in PARTITION_ROLES]) * (
        len(subjects) - len(PARTITION_ROLES)
    )
    counts = np.floor(raw).astype(int) + 1
    remaining = len(subjects) - int(counts.sum())
    for index in np.argsort(-(raw - np.floor(raw)))[:remaining]:
        counts[index] += 1

    partitions = {}
    subject_partitions = {}
    start = 0
    for role, count in zip(PARTITION_ROLES, counts):
        selected = set(shuffled[start : start + int(count)])
        subject_partitions[role] = sorted(selected)
        partitions[role] = sorted(
            case_id for case_id in allowed if case_to_subject[case_id] in selected
        )
        start += int(count)
    manifest = {
        "format": PARTITION_FORMAT,
        "source_split": "validation",
        "exploratory": True,
        "seed": int(seed),
        "fractions": fractions,
        "allowed_case_ids": sorted(allowed),
        "partitions": partitions,
        "subject_partitions": subject_partitions,
        "case_counts": {role: len(partitions[role]) for role in PARTITION_ROLES},
        "subject_counts": {
            role: len(subject_partitions[role]) for role in PARTITION_ROLES
        },
        "limitation": (
            "The frozen Beta 1 controller was selected using the full validation "
            "set; these grouped partitions support exploratory Beta 2 analysis only."
        ),
    }
    validate_subject_partitions(
        manifest,
        case_to_subject,
        forbidden_case_ids=forbidden_case_ids,
        forbidden_subject_ids=forbidden_subject_ids,
    )
    if sum(Counter(case_to_subject[case] for case in allowed).values()) != len(
        allowed
    ):
        raise AssertionError("internal subject accounting failure")
    return manifest
