"""Pointwise versus 3x3x3 controller pilot on the predeclared fit split.

Frozen experts and the frozen Beta 1 controller are read only. Training
patches keep a real one-voxel halo and are the only new volumes stored.
Full-volume 65-channel feature tensors are not written.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import beta2_stability_benchmark as stability
from beta2_adaptive_pilot import load_patient
from beta2_stability_benchmark import (
    BenchmarkStop,
    CaseResourceMonitor,
    checkpoint_hashes,
    memory_free_percent,
    resource_snapshot,
    swap_used_mib,
    write_progress,
)
from brats_debate.config import device_for, load_config
from brats_debate.controller.gating_network import GatingNetwork
from brats_debate.controller.spatial_gate import (
    PointwiseMatchedGate,
    SpatialContextGate,
    parameter_count,
)
from brats_debate.controller.spatial_stream import (
    interior_prediction,
    real_window,
    stream_gates,
    window_fits,
)
from brats_debate.data.brats_dataset import discover_patients, make_splits, subject_id, validate_training_audit
from brats_debate.debate.features import feature_channels as controller_width_channels
from brats_debate.evaluation.metrics import binary_metrics
from brats_debate.inference.pipeline import load_controller, load_experts
from brats_debate.training.train_controller import assert_split_isolation, records_for
from brats_debate.training.train_expert import segmentation_loss
from brats_debate.uncertainty.extraction import predict_experts_compact
from freeze_beta2_partitions import validate_frozen_manifest
from scipy import ndimage


OUTPUT = ROOT / "outputs" / "beta2" / "spatial_controller_pilot" / "run_20261024"
PROTOCOL_PATH = ROOT / "outputs" / "beta2" / "spatial_controller_pilot" / "protocol.json"
MANIFEST_PATH = ROOT / "configs" / "beta2" / "subject_partitions.json"
STAGE2_GATE = ROOT / "outputs" / "beta2" / "stage2_adaptive_20261020" / "revised_gate.pt"
EXPECTED_PROTOCOL_SHA256 = "2928ee3e0115e58c6ba5ad0911aa6fbb656a9e4cb1a32095baa277cad98b8ff4"
EXPECTED_MANIFEST_HASH = "692099fde80794232d97f378a6cb0c391a78f89950a5ad3a6eb8d12e30a389fb"
EXPECTED_CHECKPOINTS = {
    "cnn": "e8fbde5084353dc5aa499626d208fad7c7c4595e23376daa7d98a4890259697c",
    "transformer": "dd77255376fb0831e7b91795dff3dbf1bba647b3e305df9ef5dfa21d6d55b64b",
    "boundary": "dd360d8d6430acd4eff116b181ed3b83b2144fffd16c0bf8c0baaa913f00b79c",
    "highres": "15bd9b270ff3ca160e7a4f07a93809e1e5ad9c7e25b352966df5940d639b83c0",
    "controller": "86b604082a934a1d4f4a68467a88d16d268ae7653c552dcfc6c3087ba1d79fe2",
}
TRAIN_CASES = (
    "BraTS-GLI-00006-000",
    "BraTS-GLI-00025-000",
    "BraTS-GLI-00049-000",
    "BraTS-GLI-00056-000",
    "BraTS-GLI-00056-001",
    "BraTS-GLI-00101-000",
    "BraTS-GLI-00104-000",
    "BraTS-GLI-00112-000",
    "BraTS-GLI-00133-000",
)
EVAL_CASES = (
    "BraTS-GLI-01016-000",
    "BraTS-GLI-01020-000",
    "BraTS-GLI-01022-000",
    "BraTS-GLI-01064-000",
)
PRIOR_EVAL_ROOTS = (
    ROOT / "outputs" / "beta2" / "stage2_adaptive_20261020",
    ROOT / "outputs" / "beta2" / "stage3_mri_20261020",
    ROOT / "outputs" / "beta2" / "stage3_proposal_extension_20261020",
    ROOT / "outputs" / "beta2" / "stage4_frozen_eval_20261020",
    ROOT / "outputs" / "beta2" / "adaptive_pilot_20261019",
)
REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
INTERIOR = 16
HALO = 1
PATCHES_PER_SUBJECT = 36
TUMOR_PATCHES = 24
OTHER_PATCHES = 12
EPOCHS = 40
SEED = 20261024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
SLAB_DEPTH = 8
STOP_FREE_BYTES = 20 * 1024**3
START_FREE_BYTES = 25 * 1024**3
WALL_SECONDS = 6 * 60 * 60
SOFT_STOP_SECONDS = WALL_SECONDS - 20 * 60


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_json(value):
    if isinstance(value, dict):
        return {str(key): as_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return as_json(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def subject_of(case_id: str) -> str:
    return case_id.rsplit("-", 1)[0]


def patches_for_case(case_id: str) -> int:
    """00056's two scans split that subject's patch budget. Every other case gets the full budget."""

    if case_id in {"BraTS-GLI-00056-000", "BraTS-GLI-00056-001"}:
        return PATCHES_PER_SUBJECT // 2
    return PATCHES_PER_SUBJECT


def predicted_storage_bytes() -> int:
    window = INTERIOR + 2 * HALO
    per_patch = window**3 * (65 + 4 * 4 + 4) * 4 + INTERIOR**3 * 8
    training = sum(patches_for_case(case_id) for case_id in TRAIN_CASES) * per_patch
    labels = len(EVAL_CASES) * 5 * 240 * 240 * 155
    return int(training + labels + 32 * 1024**2)


def require_resources(output: Path):
    import shutil

    free = shutil.disk_usage(output if output.exists() else ROOT).free
    needed = predicted_storage_bytes()
    if free < START_FREE_BYTES:
        raise BenchmarkStop(f"internal free space is {free / 1024**3:.2f} GiB, below the 25 GiB start gate")
    if needed > free - STOP_FREE_BYTES:
        raise BenchmarkStop(
            f"predicted new storage is {needed / 1024**3:.2f} GiB, which would cross the 20 GiB stop line"
        )
    pressure = memory_free_percent()
    swap = swap_used_mib()
    if pressure is not None and pressure < 15:
        raise BenchmarkStop(f"memory free percentage is {pressure}%; refusing to start inference")
    if swap is not None and swap > 2048:
        raise BenchmarkStop(f"swap is already {swap:.0f} MiB; refusing to start inference")
    return {"disk_free_bytes": free, "predicted_new_bytes": needed, "memory_free_percent": pressure, "swap_used_mib": swap}


def require_protocol_and_split(cfg, records, splits):
    if sha256_file(PROTOCOL_PATH) != EXPECTED_PROTOCOL_SHA256:
        raise BenchmarkStop("the predeclared spatial protocol file changed")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol["train_subjects_first_scan"] != [
        "BraTS-GLI-00006-000",
        "BraTS-GLI-00025-000",
        "BraTS-GLI-00049-000",
        "BraTS-GLI-00056-000",
        "BraTS-GLI-00101-000",
        "BraTS-GLI-00104-000",
        "BraTS-GLI-00112-000",
        "BraTS-GLI-00133-000",
    ]:
        raise BenchmarkStop("the protocol training list changed")
    if protocol["evaluation_subjects_first_scan"] != list(EVAL_CASES):
        raise BenchmarkStop("the protocol evaluation list changed")
    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest.get("manifest_hash") != EXPECTED_MANIFEST_HASH:
        raise BenchmarkStop("the frozen subject-partition manifest hash changed")
    import yaml

    stage_b = yaml.safe_load((ROOT / "configs" / "beta2" / "foundation.yaml").read_text())["stage_b"]
    validate_frozen_manifest(manifest, cfg=cfg, records=records, splits=splits, stage_b_cfg=stage_b)
    train_subjects = {subject_of(case_id) for case_id in TRAIN_CASES}
    eval_subjects = {subject_of(case_id) for case_id in EVAL_CASES}
    if train_subjects & eval_subjects:
        raise BenchmarkStop("a subject is in both training and evaluation")
    if len(EVAL_CASES) != len(set(EVAL_CASES)) or len(TRAIN_CASES) != len(set(TRAIN_CASES)):
        raise BenchmarkStop("a case id is repeated")
    fit = set(manifest["partitions"]["fit"])
    for case_id in TRAIN_CASES + EVAL_CASES:
        if case_id not in fit:
            raise BenchmarkStop(f"{case_id} is not in the frozen fit partition")
        if case_id in set(splits["test"]):
            raise BenchmarkStop(f"{case_id} is a historical test case")
        if case_id not in set(splits["validation"]):
            raise BenchmarkStop(f"{case_id} is outside the validation pool")
    for case_id in EVAL_CASES:
        for folder in PRIOR_EVAL_ROOTS:
            if folder.exists() and any(folder.rglob(f"*{case_id}*")):
                raise BenchmarkStop(f"{case_id} already appears under {folder.name}")
    wanted = list(TRAIN_CASES + EVAL_CASES)
    selected = records_for(records, wanted, splits["test"])
    identities = {record["patient_id"]: subject_id(record, cfg) for record in selected}
    for case_id in wanted:
        if identities[case_id] != subject_of(case_id):
            raise BenchmarkStop(f"{case_id} does not belong to the expected subject")
    return protocol, {record["patient_id"]: record for record in selected}, identities


def sample_origins(label, shape, rng, tumor_count, other_count):
    tumor = np.isin(label, (1, 2, 3))
    other = label == 0

    def take(mask, count):
        coords = np.argwhere(mask)
        chosen, seen = [], set()
        if coords.size == 0 or count == 0:
            return chosen
        attempts = 0
        while len(chosen) < count and attempts < count * 80:
            attempts += 1
            voxel = coords[int(rng.integers(0, len(coords)))]
            origin = []
            for axis in range(3):
                start = int(voxel[axis]) - INTERIOR // 2
                low = HALO
                high = int(shape[axis]) - INTERIOR - HALO
                if high < low:
                    return chosen
                origin.append(int(min(max(low, start), high)))
            origin = tuple(origin)
            if origin in seen or not window_fits(origin, shape, INTERIOR, HALO):
                continue
            if not all(origin[axis] <= int(voxel[axis]) < origin[axis] + INTERIOR for axis in range(3)):
                continue
            seen.add(origin)
            chosen.append(origin)
        return chosen

    tumor_origins = take(tumor, tumor_count)
    other_origins = take(other, other_count)
    return tumor_origins, other_origins


def extract_case(case_id, record, models, cfg, device, channels):
    import shutil

    if shutil.disk_usage(OUTPUT).free < STOP_FREE_BYTES:
        raise BenchmarkStop("internal free space fell below 20 GiB")
    destination = OUTPUT / "patches" / f"{case_id}.npz"
    sidecar = OUTPUT / "patches" / f"{case_id}.json"
    if destination.is_file() and sidecar.is_file():
        meta = json.loads(sidecar.read_text())
        if meta.get("patient_id") == case_id and meta.get("channels") == channels:
            print(json.dumps({"skipped_existing_patches": case_id}), flush=True)
            return meta
        raise BenchmarkStop(f"existing patch file for {case_id} does not match this protocol")
    quota = patches_for_case(case_id)
    tumor_count = round(quota * TUMOR_PATCHES / PATCHES_PER_SUBJECT)
    other_count = quota - tumor_count
    monitor = CaseResourceMonitor(OUTPUT)
    monitor.start()
    patient = outputs = image = None
    started = time.perf_counter()
    try:
        patient = load_patient(record, cfg)
        if patient["patient_id"] != case_id:
            raise RuntimeError("loaded patient does not match the requested case")
        if patient["label"] is None:
            raise RuntimeError("training case has no label")
        outputs = predict_experts_compact(patient, models, cfg, device)
        image = torch.from_numpy(np.ascontiguousarray(patient["image"]))[None]
        label = np.asarray(patient["label"])
        if tuple(label.shape) != tuple(image.shape[-3:]):
            raise RuntimeError("label and image are not aligned")
        rng = np.random.default_rng(np.random.SeedSequence([SEED, TRAIN_CASES.index(case_id)]))
        tumor_origins, other_origins = sample_origins(label, label.shape, rng, tumor_count, other_count)
        origins = tumor_origins + other_origins
        if len(tumor_origins) < max(4, tumor_count // 2):
            raise BenchmarkStop(f"{case_id} produced only {len(tumor_origins)} tumor windows")
        features, probabilities, availability, targets = [], [], [], []
        for origin in origins:
            window_features, window_probabilities, window_availability = real_window(
                image, outputs, origin, INTERIOR, HALO
            )
            if window_features.shape[1] != channels:
                raise RuntimeError("controller feature width changed")
            features.append(window_features[0].contiguous().numpy())
            probabilities.append(window_probabilities[0].contiguous().numpy())
            availability.append(window_availability[0].contiguous().numpy())
            target = label[
                origin[0] : origin[0] + INTERIOR,
                origin[1] : origin[1] + INTERIOR,
                origin[2] : origin[2] + INTERIOR,
            ]
            targets.append(np.ascontiguousarray(target, dtype=np.int16))
        payload = {
            "features": np.stack(features).astype(np.float32),
            "probabilities": np.stack(probabilities).astype(np.float32),
            "availability": np.stack(availability).astype(np.float32),
            "target": np.stack(targets),
            "origins": np.asarray(origins, dtype=np.int16),
        }
        meta = {
            "patient_id": case_id,
            "subject_id": subject_of(case_id),
            "channels": channels,
            "interior": INTERIOR,
            "halo": HALO,
            "tumor_windows": len(tumor_origins),
            "other_windows": len(other_origins),
            "windows": len(origins),
            "ground_truth_used_for_inference": False,
            "ground_truth_used_for_patch_location": True,
            "ground_truth_used_as_feature": False,
            "highres_full_volume_used": False,
            "seconds": time.perf_counter() - started,
            "resources": monitor.summary(),
        }
        temporary = destination.with_name(destination.stem + ".partial.npz")
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, destination)
        write_progress(sidecar, meta)
        print(json.dumps({"extracted": case_id, "windows": len(origins), "seconds": meta["seconds"]}), flush=True)
        return meta
    finally:
        del patient, outputs, image
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        monitor.stop()


def load_patches(channels):
    pools = {subject_of(case_id): [] for case_id in TRAIN_CASES}
    store = {"features": [], "probabilities": [], "availability": [], "target": []}
    cursor = 0
    for case_id in TRAIN_CASES:
        path = OUTPUT / "patches" / f"{case_id}.npz"
        arrays = np.load(path)
        count = int(arrays["features"].shape[0])
        if arrays["features"].shape[1:] != (channels, INTERIOR + 2 * HALO, INTERIOR + 2 * HALO, INTERIOR + 2 * HALO):
            raise RuntimeError(f"{case_id} feature window has the wrong shape")
        if arrays["target"].shape[1:] != (INTERIOR, INTERIOR, INTERIOR):
            raise RuntimeError(f"{case_id} target window has the wrong shape")
        for key in store:
            store[key].append(arrays[key])
        pools[subject_of(case_id)].extend(range(cursor, cursor + count))
        cursor += count
        arrays.close()
    stacked = {key: np.concatenate(value, axis=0) for key, value in store.items()}
    if any(len(indices) == 0 for indices in pools.values()):
        raise RuntimeError("a training subject has no windows")
    return stacked, pools


def training_schedule(pools):
    subjects = list(dict.fromkeys(subject_of(case_id) for case_id in TRAIN_CASES))
    rng = np.random.default_rng(SEED)
    steps_per_epoch = max(len(indices) for indices in pools.values())
    schedule = []
    for _epoch in range(EPOCHS):
        order = {subject: rng.permutation(indices).tolist() for subject, indices in pools.items()}
        epoch = []
        for step in range(steps_per_epoch):
            epoch.append([order[subject][step % len(order[subject])] for subject in subjects])
        schedule.append(epoch)
    return subjects, schedule


def train_one(kind, stacked, schedule, channels):
    torch.manual_seed(SEED)
    if kind == "pointwise":
        model = PointwiseMatchedGate(channels, width=11, experts=4)
    elif kind == "spatial":
        model = SpatialContextGate(channels, width=8, experts=4)
    else:
        raise ValueError(kind)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses = []
    for epoch, steps in enumerate(schedule, start=1):
        epoch_loss = []
        for batch in steps:
            features = torch.from_numpy(stacked["features"][batch])
            probabilities = torch.from_numpy(stacked["probabilities"][batch])
            availability = torch.from_numpy(stacked["availability"][batch])
            target = torch.from_numpy(stacked["target"][batch].astype(np.int64))
            prediction = interior_prediction(model, features, probabilities, availability, HALO)["probabilities"]
            loss = torch.stack(
                [segmentation_loss(prediction[index : index + 1], target[index : index + 1]) for index in range(prediction.shape[0])]
            ).mean()
            if not torch.isfinite(loss):
                raise BenchmarkStop(f"{kind} training produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
        losses.append(float(np.mean(epoch_loss)))
        print(json.dumps({"train": kind, "epoch": epoch, "loss": losses[-1]}), flush=True)
    model.eval()
    destination = OUTPUT / f"{kind}.pt"
    temporary = destination.with_suffix(".partial.pt")
    torch.save(
        {
            "format": "beta2_spatial_pilot_gate_v1",
            "kind": kind,
            "state_dict": model.state_dict(),
            "in_channels": channels,
            "epochs": EPOCHS,
            "seed": SEED,
            "loss_per_epoch": losses,
            "parameter_count": parameter_count(model),
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "grad_clip": GRAD_CLIP,
            "selection": "final_epoch_only",
            "spatial_regularizer": None,
        },
        temporary,
    )
    os.replace(temporary, destination)
    return model, losses


def load_trained(kind, channels):
    path = OUTPUT / f"{kind}.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["kind"] != kind or saved["epochs"] != EPOCHS or saved["seed"] != SEED:
        raise BenchmarkStop(f"{path.name} does not match the predeclared training budget")
    if kind == "pointwise":
        model = PointwiseMatchedGate(channels, width=11, experts=4)
    else:
        model = SpatialContextGate(channels, width=8, experts=4)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, saved["loss_per_epoch"]


def region_scores(prediction, target, spacing):
    scores = {}
    for name, labels in REGIONS.items():
        metrics = binary_metrics(np.isin(prediction, labels), np.isin(target, labels), spacing, hd95=True)
        scores[name] = {"dice": float(metrics["dice"]), "hd95_mm": metrics["hd95_mm"]}
    scores["mean_dice"] = float(np.mean([scores[name]["dice"] for name in REGIONS]))
    scores["voxel_errors"] = int(np.count_nonzero(prediction != target))
    return scores


def component_story(prediction, target, labels, spacing, hd95_mm):
    pred = np.isin(prediction, labels)
    truth = np.isin(target, labels)
    pred_ids, pred_count = ndimage.label(pred)
    truth_ids, truth_count = ndimage.label(truth)
    islands = []
    without_islands = pred.copy()
    for index in range(1, pred_count + 1):
        component = pred_ids == index
        if not np.any(component & truth):
            islands.append(int(component.sum()))
            without_islands[component] = False
    missed = []
    for index in range(1, truth_count + 1):
        component = truth_ids == index
        if not np.any(component & pred):
            missed.append(int(component.sum()))
    no_island = binary_metrics(without_islands, truth, spacing, hd95=True)
    if pred_count:
        sizes = ndimage.sum(pred, pred_ids, index=np.arange(1, pred_count + 1))
        largest = pred_ids == (int(np.argmax(sizes)) + 1)
        largest_metrics = binary_metrics(largest, truth, spacing, hd95=True)
    else:
        largest_metrics = {"hd95_mm": None}
    return {
        "pred_components": int(pred_count),
        "gt_components": int(truth_count),
        "false_positive_islands": len(islands),
        "false_positive_island_voxels": int(sum(islands)),
        "largest_false_positive_island_voxels": int(max(islands) if islands else 0),
        "missed_gt_components": len(missed),
        "missed_gt_component_voxels": int(sum(missed)),
        "hd95_mm": hd95_mm,
        "hd95_without_false_positive_islands_mm": no_island["hd95_mm"],
        "hd95_largest_pred_component_mm": largest_metrics["hd95_mm"],
    }


def classify_disagreement(candidate, reference):
    """Quantitative flags. A single slice is not used as the cause."""

    notes = []
    dice_up = candidate["dice"] > reference["dice"]
    hd_candidate = candidate["hd95_mm"]
    hd_reference = reference["hd95_mm"]
    hd_worse = (
        hd_candidate is not None
        and hd_reference is not None
        and hd_candidate > hd_reference + 0.05
    )
    if dice_up and hd_worse:
        notes.append("higher_dice_and_worse_hd95")
        stripped = candidate["hd95_without_false_positive_islands_mm"]
        if stripped is not None and hd_candidate - stripped > 1:
            notes.append("false_positive_islands_account_for_more_than_1mm_of_hd95")
        if candidate["missed_gt_components"] > reference["missed_gt_components"]:
            notes.append("more_disconnected_ground_truth_components_missed")
        largest = candidate["hd95_largest_pred_component_mm"]
        if largest is not None and abs(hd_candidate - largest) <= 0.5:
            notes.append("hd95_remains_on_the_largest_component")
    return notes


def save_contour(path, image, prediction, target, affine, spacing, case_id, region, labels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib

    codes = nib.aff2axcodes(affine)
    axial = next((index for index, code in enumerate(codes) if code in ("S", "I")), int(np.argmin(target.shape)))
    truth = np.isin(target, labels)
    counts = truth.sum(axis=tuple(axis for axis in range(3) if axis != axial))
    slice_index = int(np.argmax(counts)) if int(counts.max()) else int(target.shape[axial] // 2)
    flair = np.take(image[3], slice_index, axis=axial)
    finite = flair[np.isfinite(flair)]
    lo, hi = (np.percentile(finite, [1, 99]) if finite.size else (0, 1))

    def plane(volume):
        return np.take(np.isin(volume, labels), slice_index, axis=axial)

    panels = (
        ("Ground truth", plane(target), "#f4d35e"),
        ("Transformer", plane(prediction["transformer"]), "#4cc9f0"),
        ("Frozen Beta 1", plane(prediction["frozen_beta1"]), "#f4845f"),
        ("Pointwise control", plane(prediction["pointwise"]), "#f8f9fa"),
        ("Spatial controller", plane(prediction["spatial"]), "#c77dff"),
    )
    figure, axes = plt.subplots(1, len(panels), figsize=(15.5, 3.6), constrained_layout=True)
    in_plane = [code for index, code in enumerate(codes) if index != axial]
    for axis, (title, mask, color) in zip(axes, panels):
        axis.imshow(flair, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
        if np.any(mask):
            axis.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=1.1)
        if title != "Ground truth" and np.any(plane(target)):
            axis.contour(plane(target).astype(float), levels=[0.5], colors=["#f4d35e"], linewidths=0.7, linestyles="dashed")
        axis.set_title(title, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])
    position_mm = slice_index * float(spacing[axial])
    figure.suptitle(
        f"{case_id}  {region}  constant-{codes[axial]} slice {slice_index} ({position_mm:.1f} mm)  "
        f"in-plane {in_plane[0]} x {in_plane[1]}  array order",
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {"axis": int(axial), "axis_code": codes[axial], "slice_index": slice_index, "slice_mm": position_mm, "in_plane": in_plane}


def evaluate_case(case_id, record, models, gates, cfg, device, channels):
    import shutil

    if shutil.disk_usage(OUTPUT).free < STOP_FREE_BYTES:
        raise BenchmarkStop("internal free space fell below 20 GiB")
    path = OUTPUT / "evaluation" / f"{case_id}.json"
    if path.is_file():
        print(json.dumps({"skipped_existing_evaluation": case_id}), flush=True)
        return json.loads(path.read_text())
    monitor = CaseResourceMonitor(OUTPUT)
    before = resource_snapshot(OUTPUT)
    monitor.start()
    patient = outputs = image = None
    started = time.perf_counter()
    try:
        patient = load_patient(record, cfg)
        if patient["patient_id"] != case_id or patient["label"] is None:
            raise RuntimeError("evaluation patient did not load with its label")
        expert_started = time.perf_counter()
        outputs = predict_experts_compact(patient, models, cfg, device)
        expert_seconds = time.perf_counter() - expert_started
        image = torch.from_numpy(np.ascontiguousarray(patient["image"]))[None]
        if image.shape[1] != 4:
            raise RuntimeError("expected four MRI channels")
        labels, stats = stream_gates(gates, image, outputs, cfg, device, slab_depth=SLAB_DEPTH, halo=HALO)
        for name, error in stats["max_probability_sum_error"].items():
            if error > 1e-3 or stats["max_weight_sum_error"][name] > 1e-3:
                raise RuntimeError(f"{name} fusion is not a normalized distribution")
            if stats["max_unavailable_expert_weight"][name] > 1e-3:
                raise RuntimeError(f"{name} gave weight to an unavailable expert")
        transformer = outputs["transformer"]["probabilities"].argmax(1)[0].to(torch.uint8).cpu().numpy()
        prediction = {
            "transformer": transformer,
            "frozen_beta1": labels["frozen_beta1"].numpy(),
            "pointwise": labels["pointwise"].numpy(),
            "spatial": labels["spatial"].numpy(),
        }
        for name, volume in prediction.items():
            if volume.shape != patient["label"].shape or not np.isin(volume, (0, 1, 2, 3)).all():
                raise RuntimeError(f"{name} prediction is not an aligned four-class label")
        target = np.asarray(patient["label"])
        spacing = tuple(float(value) for value in patient["spacing"])
        metrics = {name: region_scores(volume, target, spacing) for name, volume in prediction.items()}
        stories = {}
        for method, volume in prediction.items():
            stories[method] = {}
            for region, region_labels in REGIONS.items():
                stories[method][region] = component_story(
                    volume, target, region_labels, spacing, metrics[method][region]["hd95_mm"]
                )
        comparisons = {}
        for reference in ("transformer", "frozen_beta1", "pointwise"):
            comparisons[f"spatial_minus_{reference}"] = {}
            for region in REGIONS:
                candidate = {
                    "dice": metrics["spatial"][region]["dice"],
                    "hd95_mm": metrics["spatial"][region]["hd95_mm"],
                    **stories["spatial"][region],
                }
                baseline = {
                    "dice": metrics[reference][region]["dice"],
                    "hd95_mm": metrics[reference][region]["hd95_mm"],
                    **stories[reference][region],
                }
                comparisons[f"spatial_minus_{reference}"][region] = {
                    "dice": candidate["dice"] - baseline["dice"],
                    "hd95_mm": None
                    if candidate["hd95_mm"] is None or baseline["hd95_mm"] is None
                    else candidate["hd95_mm"] - baseline["hd95_mm"],
                    "flags": classify_disagreement(candidate, baseline),
                }
        changes = {}
        for reference in ("transformer", "frozen_beta1", "pointwise"):
            reference_wrong = prediction[reference] != target
            spatial_wrong = prediction["spatial"] != target
            changes[reference] = {
                "spatial_errors": int(spatial_wrong.sum()),
                "reference_errors": int(reference_wrong.sum()),
                "repaired": int(np.count_nonzero(reference_wrong & ~spatial_wrong)),
                "introduced": int(np.count_nonzero(~reference_wrong & spatial_wrong)),
            }
        figure_meta = {}
        for region, region_labels in REGIONS.items():
            figure_meta[region] = save_contour(
                OUTPUT / "figures" / f"{case_id}_{region}.png",
                np.asarray(patient["image"]),
                prediction,
                target,
                patient["affine"],
                spacing,
                case_id,
                region,
                region_labels,
            )
        label_path = OUTPUT / "evaluation" / f"{case_id}_labels.npz"
        temporary = label_path.with_name(label_path.stem + ".partial.npz")
        np.savez_compressed(
            temporary,
            transformer=prediction["transformer"],
            frozen_beta1=prediction["frozen_beta1"],
            pointwise=prediction["pointwise"],
            spatial=prediction["spatial"],
            target=target.astype(np.uint8),
        )
        os.replace(temporary, label_path)
        summary = {
            "patient_id": case_id,
            "subject_id": subject_of(case_id),
            "spacing_mm": list(spacing),
            "seconds": time.perf_counter() - started,
            "expert_seconds": expert_seconds,
            "feature_seconds": stats["feature_seconds"],
            "controller_forward_seconds": stats["forward_seconds"],
            "channels": channels,
            "metrics": metrics,
            "voxel_changes_spatial_versus": changes,
            "components": stories,
            "paired": comparisons,
            "weight_neighbor_abs_difference": stats["mean_neighbor_weight_abs_difference"],
            "fusion_checks": {
                "max_probability_sum_error": stats["max_probability_sum_error"],
                "max_weight_sum_error": stats["max_weight_sum_error"],
                "max_unavailable_expert_weight": stats["max_unavailable_expert_weight"],
            },
            "figures": figure_meta,
            "resources": {**monitor.summary(), "before": before, "after": resource_snapshot(OUTPUT)},
        }
        write_progress(path, as_json(summary))
        print(
            json.dumps(
                {
                    "evaluated": case_id,
                    "mean_dice": {name: value["mean_dice"] for name, value in metrics.items()},
                    "controller_forward_seconds": stats["forward_seconds"],
                }
            ),
            flush=True,
        )
        return summary
    finally:
        del patient, outputs, image
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        monitor.stop()


def aggregate(cases):
    methods = ("transformer", "frozen_beta1", "pointwise", "spatial")
    summary = {"cases": [case["patient_id"] for case in cases], "per_case": {}, "mean_dice": {}, "mean_hd95_mm": {}}
    for method in methods:
        summary["mean_dice"][method] = float(np.mean([case["metrics"][method]["mean_dice"] for case in cases]))
        summary["mean_hd95_mm"][method] = {}
        for region in REGIONS:
            values = [
                case["metrics"][method][region]["hd95_mm"]
                for case in cases
                if case["metrics"][method][region]["hd95_mm"] is not None
            ]
            summary["mean_hd95_mm"][method][region] = float(np.mean(values)) if values else None
    for case in cases:
        summary["per_case"][case["patient_id"]] = {
            method: {
                "mean_dice": case["metrics"][method]["mean_dice"],
                "voxel_errors": case["metrics"][method]["voxel_errors"],
                **{region: case["metrics"][method][region] for region in REGIONS},
            }
            for method in methods
        }
    summary["paired_spatial_minus"] = {}
    for reference in ("pointwise", "frozen_beta1", "transformer"):
        summary["paired_spatial_minus"][reference] = [
            {
                "patient_id": case["patient_id"],
                "mean_dice": case["metrics"]["spatial"]["mean_dice"] - case["metrics"][reference]["mean_dice"],
                "regions": case["paired"][f"spatial_minus_{reference}"],
                "voxels": case["voxel_changes_spatial_versus"][reference],
            }
            for case in cases
        ]
    return summary


def save_loss_figure(curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    for name, losses in curves.items():
        axis.plot(range(1, len(losses) + 1), losses, label=name, linewidth=1.5)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Training loss")
    axis.set_title("Matched budget, training subjects only")
    axis.legend(frameon=False)
    path = OUTPUT / "figures" / "training_loss.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def time_left(started):
    return SOFT_STOP_SECONDS - (time.perf_counter() - started)


def main():
    import shutil

    stability.MINIMUM_FREE_BYTES = STOP_FREE_BYTES
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "patches").mkdir(exist_ok=True)
    (OUTPUT / "evaluation").mkdir(exist_ok=True)
    resources = require_resources(OUTPUT)
    cfg = load_config(ROOT / "configs" / "brats_controller.yaml")
    channels = controller_width_channels(cfg)
    if channels != 65:
        raise BenchmarkStop(f"expected 65 controller channels, found {channels}")
    if parameter_count(PointwiseMatchedGate(channels, 11, 4)) != 774:
        raise BenchmarkStop("the pointwise control no longer has 774 parameters")
    if parameter_count(SpatialContextGate(channels, 8, 4)) != 788:
        raise BenchmarkStop("the spatial gate no longer has 788 parameters")
    if parameter_count(GatingNetwork(channels, 8, 4)) != 564:
        raise BenchmarkStop("the frozen Beta 1 width no longer has 564 parameters")
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    assert_split_isolation(splits)
    validate_training_audit(cfg)
    _protocol, by_id, identities = require_protocol_and_split(cfg, records, splits)
    expert_hashes, controller_hash = checkpoint_hashes(cfg)
    if expert_hashes | {"controller": controller_hash} != EXPECTED_CHECKPOINTS:
        raise BenchmarkStop("frozen checkpoint hashes do not match the Beta 1 manifest")
    stage2_hash = sha256_file(STAGE2_GATE) if STAGE2_GATE.is_file() else None
    run_record = {
        "status": "running",
        "seed": SEED,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip": GRAD_CLIP,
        "optimizer": "Adam",
        "loss": "voxel NLL plus one minus the mean soft Dice of classes 1, 2, and 3",
        "selection": "final epoch; evaluation subjects are not used for selection",
        "spatial_regularizer": None,
        "interior": INTERIOR,
        "halo": HALO,
        "patches_per_subject": PATCHES_PER_SUBJECT,
        "subject_00056_scans_split_the_subject_budget": True,
        "tumor_patch_fraction": TUMOR_PATCHES / PATCHES_PER_SUBJECT,
        "train_cases": list(TRAIN_CASES),
        "evaluation_cases": list(EVAL_CASES),
        "subjects": identities,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "manifest_hash": EXPECTED_MANIFEST_HASH,
        "checkpoint_sha256": EXPECTED_CHECKPOINTS,
        "stage2_gate_sha256_before": stage2_hash,
        "resources_before": resources,
        "storage_plan": "real halo windows only; no full-volume 65-channel feature file",
    }
    existing = OUTPUT / "predeclared_run.json"
    if existing.is_file():
        previous = json.loads(existing.read_text())
        comparable = {key: previous.get(key) for key in run_record if key not in {"status", "resources_before", "stage2_gate_sha256_before"}}
        current = {key: run_record[key] for key in comparable}
        if comparable != current:
            raise BenchmarkStop("a previous run record used a different predeclared protocol")
    else:
        write_progress(existing, run_record)
    device = device_for(cfg["device"])
    if device.type != "mps":
        raise BenchmarkStop("expert inference requires MPS")

    def stop_handler(_signum, _frame):
        raise BenchmarkStop("safety or wall-clock limit")

    signal.signal(signal.SIGALRM, stop_handler)
    signal.signal(signal.SIGUSR1, stop_handler)
    signal.alarm(WALL_SECONDS)
    started = time.perf_counter()
    status, error = "running", None
    extraction, curves, evaluations = [], {}, []
    try:
        models, loaded_experts = load_experts(cfg, splits)
        controller, loaded_controller = load_controller(cfg, splits, loaded_experts)
        if loaded_experts != expert_hashes or loaded_controller != controller_hash:
            raise BenchmarkStop("checkpoint hashes changed while loading")
        for case_id in TRAIN_CASES:
            if time_left(started) < 0:
                raise BenchmarkStop("stopping before the next training case so the report can be saved")
            extraction.append(extract_case(case_id, by_id[case_id], models, cfg, device, channels))
        del models, controller
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        stacked, pools = load_patches(channels)
        _subjects, schedule = training_schedule(pools)
        write_progress(
            OUTPUT / "training_schedule.json",
            {
                "epochs": EPOCHS,
                "steps_per_epoch": len(schedule[0]),
                "patches_per_subject": {subject: len(indices) for subject, indices in pools.items()},
                "batching": "one window from each subject at every step",
            },
        )
        for kind in ("pointwise", "spatial"):
            checkpoint = OUTPUT / f"{kind}.pt"
            if checkpoint.is_file():
                _model, curves[kind] = load_trained(kind, channels)
                print(json.dumps({"skipped_existing_training": kind, "final_loss": curves[kind][-1]}), flush=True)
            else:
                _model, curves[kind] = train_one(kind, stacked, schedule, channels)
            del _model
        save_loss_figure(curves)
        write_progress(OUTPUT / "training_loss.json", curves)
        del stacked
        gc.collect()
        models, loaded_experts = load_experts(cfg, splits)
        controller, loaded_controller = load_controller(cfg, splits, loaded_experts)
        if loaded_experts != expert_hashes or loaded_controller != controller_hash:
            raise BenchmarkStop("checkpoint hashes changed before evaluation")
        pointwise, _loss = load_trained("pointwise", channels)
        spatial, _loss = load_trained("spatial", channels)
        gates = {"frozen_beta1": controller, "pointwise": pointwise, "spatial": spatial}
        for case_id in EVAL_CASES:
            if time_left(started) < 0:
                raise BenchmarkStop("stopping before the next evaluation case so completed cases stay saved")
            evaluations.append(evaluate_case(case_id, by_id[case_id], models, gates, cfg, device, channels))
        status = "complete" if len(evaluations) == len(EVAL_CASES) and len(extraction) == len(TRAIN_CASES) else "partial"
    except Exception as exc:
        status = "stopped" if isinstance(exc, BenchmarkStop) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, BenchmarkStop):
            raise
    finally:
        signal.alarm(0)
        ordered = []
        for case_id in EVAL_CASES:
            path = OUTPUT / "evaluation" / f"{case_id}.json"
            if path.is_file():
                ordered.append(json.loads(path.read_text()))
        saved_extraction = []
        for case_id in TRAIN_CASES:
            path = OUTPUT / "patches" / f"{case_id}.json"
            if path.is_file():
                saved_extraction.append(json.loads(path.read_text()))
        if saved_extraction:
            extraction = saved_extraction
        loss_path = OUTPUT / "training_loss.json"
        if loss_path.is_file():
            curves = json.loads(loss_path.read_text())
        after_hashes = None
        try:
            after_experts, after_controller = checkpoint_hashes(cfg)
            after_hashes = after_experts | {"controller": after_controller}
        except Exception as exc:
            after_hashes = {"error": f"{type(exc).__name__}: {exc}"}
        write_progress(
            OUTPUT / "report.json",
            as_json(
                {
                    "format": "beta2_spatial_controller_pilot_v1",
                    "status": status,
                    "error": error,
                    "seconds": time.perf_counter() - started,
                    "train_cases": list(TRAIN_CASES),
                    "evaluation_cases": list(EVAL_CASES),
                    "extraction": extraction,
                    "training_loss": curves,
                    "evaluation": ordered,
                    "aggregate": aggregate(ordered) if ordered else {},
                    "checkpoints_unchanged": after_hashes == EXPECTED_CHECKPOINTS,
                    "stage2_gate_unchanged": sha256_file(STAGE2_GATE) == stage2_hash if stage2_hash else None,
                    "protocol_unchanged": sha256_file(PROTOCOL_PATH) == EXPECTED_PROTOCOL_SHA256,
                    "scope": "Eight previously observed development subjects, including both scans of BraTS-GLI-00056, and four predeclared unused fit subjects. Exploratory. Not an external test.",
                }
            ),
        )
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
