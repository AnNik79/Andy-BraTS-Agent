"""Matched pointwise controllers that differ only by the geometric loss.

Model A uses the original segmentation loss. Model B adds the validated
distance term. Expert predictions are the frozen patches from the spatial
pilot. Full-volume distance maps are cropped onto those windows.

Evaluation reuses the CPU one-expert path and is refused while free disk
is below 20 GiB.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import beta2_spatial_pilot as pilot
from beta2_adaptive_pilot import load_patient
from beta2_stability_benchmark import BenchmarkStop, memory_free_percent, swap_used_mib
from brats_debate.controller.spatial_gate import PointwiseMatchedGate, parameter_count
from brats_debate.controller.spatial_stream import interior_prediction
from brats_debate.training.geometric_loss import (
    DISTANCE_CAP_MM,
    EPSILON,
    geometric_loss,
    reference_distances,
)
from brats_debate.training.train_expert import segmentation_loss

OUTPUT = ROOT / "outputs" / "beta2" / "geometric_loss_20261103"
PATCHES = ROOT / "outputs" / "beta2" / "spatial_controller_pilot" / "lowmem_20261025" / "patches"
EVAL_CASES = (
    "BraTS-GLI-01065-000",
    "BraTS-GLI-01071-000",
    "BraTS-GLI-01077-000",
    "BraTS-GLI-01081-000",
)
LAMBDA_FRACTION = 0.25
STOP_FREE_BYTES = 20 * 1024**3
WALL_SECONDS = 8 * 60 * 60


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(pilot.as_json(payload), indent=2))
    os.replace(temporary, path)


def disk_free() -> int:
    return shutil.disk_usage(ROOT).free


def verify_subjects(cfg, records, splits):
    manifest = json.loads((ROOT / "configs" / "beta2" / "subject_partitions.json").read_text())
    fit = set(manifest["partitions"]["fit"])
    train_subjects = {pilot.subject_of(case_id) for case_id in pilot.TRAIN_CASES}
    for case_id in EVAL_CASES:
        subject = pilot.subject_of(case_id)
        if case_id not in fit:
            raise BenchmarkStop(f"{case_id} is not in the fit partition")
        if case_id in set(splits["test"]):
            raise BenchmarkStop(f"{case_id} is a historical test case")
        if subject in train_subjects:
            raise BenchmarkStop(f"{case_id} belongs to a training subject")
        siblings = [item for item in fit if pilot.subject_of(item) == subject]
        if any(item in pilot.TRAIN_CASES for item in siblings):
            raise BenchmarkStop(f"another scan of {subject} is in training")
    wanted = list(pilot.TRAIN_CASES) + list(EVAL_CASES)
    from brats_debate.training.train_controller import records_for

    selected = records_for(records, wanted, splits["test"])
    return {record["patient_id"]: record for record in selected}


def build_geometry(case_id: str, record, cfg) -> None:
    destination = OUTPUT / "geometry" / f"{case_id}.npz"
    if destination.is_file():
        return
    patient = load_patient(record, cfg)
    label = np.asarray(patient["label"])
    spacing = tuple(float(value) for value in patient["spacing"])
    stored = np.load(PATCHES / f"{case_id}.npz")
    origins = stored["origins"]
    target = stored["target"]
    distances = []
    for _name, labels in (("WT", (1, 2, 3)), ("TC", (1, 3)), ("ET", (3,))):
        distances.append(reference_distances(np.isin(label, labels), spacing, DISTANCE_CAP_MM))
    cropped_set = []
    cropped_boundary = []
    for index, origin in enumerate(origins):
        z, y, x = (int(value) for value in origin)
        sl = (slice(z, z + pilot.INTERIOR), slice(y, y + pilot.INTERIOR), slice(x, x + pilot.INTERIOR))
        if not np.array_equal(target[index], label[sl]):
            raise RuntimeError(f"{case_id} patch {index} is not aligned with the full-volume label")
        cropped_set.append(np.stack([item[0][sl] for item in distances]))
        cropped_boundary.append(np.stack([item[1][sl] for item in distances]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".partial.npz")
    np.savez_compressed(
        temporary,
        distance_to_set=np.stack(cropped_set).astype(np.float32),
        distance_to_boundary=np.stack(cropped_boundary).astype(np.float32),
        spacing_mm=np.asarray(spacing, dtype=np.float32),
    )
    os.replace(temporary, destination)
    stored.close()
    print(json.dumps({"geometry": case_id, "windows": int(len(origins)), "spacing_mm": list(spacing)}), flush=True)


def load_training():
    store = {"features": [], "probabilities": [], "availability": [], "target": [], "distance_to_set": [], "distance_to_boundary": []}
    pools = {pilot.subject_of(case_id): [] for case_id in pilot.TRAIN_CASES}
    cursor = 0
    for case_id in pilot.TRAIN_CASES:
        patches = np.load(PATCHES / f"{case_id}.npz")
        geometry = np.load(OUTPUT / "geometry" / f"{case_id}.npz")
        count = int(patches["features"].shape[0])
        if geometry["distance_to_set"].shape[0] != count:
            raise RuntimeError(f"{case_id} geometry does not match its expert patches")
        store["features"].append(patches["features"])
        store["probabilities"].append(patches["probabilities"])
        store["availability"].append(patches["availability"])
        store["target"].append(patches["target"])
        store["distance_to_set"].append(geometry["distance_to_set"])
        store["distance_to_boundary"].append(geometry["distance_to_boundary"])
        pools[pilot.subject_of(case_id)].extend(range(cursor, cursor + count))
        cursor += count
        patches.close()
        geometry.close()
    stacked = {key: np.concatenate(value, axis=0) for key, value in store.items()}
    return stacked, pools


def batch_tensors(stacked, batch):
    return {
        "features": torch.from_numpy(stacked["features"][batch]),
        "probabilities": torch.from_numpy(stacked["probabilities"][batch]),
        "availability": torch.from_numpy(stacked["availability"][batch]),
        "target": torch.from_numpy(stacked["target"][batch].astype(np.int64)),
        "distance_to_set": torch.from_numpy(stacked["distance_to_set"][batch]),
        "distance_to_boundary": torch.from_numpy(stacked["distance_to_boundary"][batch]),
    }


def prediction_of(model, tensors):
    return interior_prediction(
        model, tensors["features"], tensors["probabilities"], tensors["availability"], pilot.HALO
    )["probabilities"]


def loss_parts(prediction, tensors, geometric: bool, lam: float):
    baseline = torch.stack(
        [
            segmentation_loss(prediction[index : index + 1], tensors["target"][index : index + 1])
            for index in range(prediction.shape[0])
        ]
    ).mean()
    distance = prediction.new_zeros(())
    if geometric:
        distance = geometric_loss(
            prediction, tensors["target"], tensors["distance_to_set"], tensors["distance_to_boundary"]
        )
    return baseline, distance, baseline + lam * distance


def measure_lambda(model, stacked, schedule) -> dict:
    baseline_values = []
    distance_values = []
    model.eval()
    with torch.no_grad():
        for batch in schedule[0]:
            tensors = batch_tensors(stacked, batch)
            prediction = prediction_of(model, tensors)
            baseline, distance, _total = loss_parts(prediction, tensors, True, 0.0)
            baseline_values.append(float(baseline))
            distance_values.append(float(distance))
    baseline_mean = float(np.mean(baseline_values))
    distance_mean = float(np.mean(distance_values))
    if not np.isfinite(baseline_mean) or not np.isfinite(distance_mean) or distance_mean <= 1e-8:
        raise BenchmarkStop("cannot set lambda from a non-finite or zero geometric loss")
    lam = LAMBDA_FRACTION * baseline_mean / distance_mean
    return {
        "lambda": lam,
        "lambda_fraction_of_baseline": LAMBDA_FRACTION,
        "initialization_baseline_loss": baseline_mean,
        "initialization_geometric_loss": distance_mean,
        "distance_cap_mm": DISTANCE_CAP_MM,
        "epsilon": EPSILON,
        "selection": "On the first-epoch training batches, at the shared initialization, lambda = 0.25 * mean baseline loss / mean geometric loss. Evaluation subjects are not used.",
    }


def train_model(name, geometric: bool, lam: float, init_state, stacked, schedule) -> dict:
    destination = OUTPUT / f"{name}.pt"
    history_path = OUTPUT / f"training_{name}.json"
    if destination.is_file() and history_path.is_file():
        return json.loads(history_path.read_text())
    torch.manual_seed(pilot.SEED)
    model = PointwiseMatchedGate(65, width=11, experts=4)
    model.load_state_dict(init_state)
    if parameter_count(model) != 774:
        raise BenchmarkStop("the pointwise controller is no longer 774 parameters")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=pilot.LEARNING_RATE, weight_decay=pilot.WEIGHT_DECAY)
    history = []
    started = time.perf_counter()
    peak = 0
    for epoch, steps in enumerate(schedule, start=1):
        totals, baselines, distances, grad_baseline, grad_distance = [], [], [], [], []
        for batch in steps:
            tensors = batch_tensors(stacked, batch)
            prediction = prediction_of(model, tensors)
            baseline, distance, total = loss_parts(prediction, tensors, geometric, lam)
            if not torch.isfinite(total):
                raise BenchmarkStop(f"{name} produced a non-finite loss")
            optimizer.zero_grad(set_to_none=True)
            baseline.backward(retain_graph=True)
            baseline_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            optimizer.zero_grad(set_to_none=True)
            if geometric:
                (lam * distance).backward(retain_graph=True)
                distance_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
            else:
                distance_norm = prediction.new_zeros(())
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), pilot.GRAD_CLIP)
            optimizer.step()
            totals.append(float(total.detach()))
            baselines.append(float(baseline.detach()))
            distances.append(float(distance.detach()))
            grad_baseline.append(float(baseline_norm))
            grad_distance.append(float(distance_norm))
            peak = max(peak, int(__import__("psutil").Process().memory_info().rss))
        row = {
            "epoch": epoch,
            "total": float(np.mean(totals)),
            "baseline": float(np.mean(baselines)),
            "geometric": float(np.mean(distances)),
            "grad_norm_baseline": float(np.mean(grad_baseline)),
            "grad_norm_geometric": float(np.mean(grad_distance)),
        }
        history.append(row)
        print(json.dumps({"train": name, **row}), flush=True)
    model.eval()
    temporary = destination.with_suffix(".partial.pt")
    torch.save(
        {
            "format": "beta2_geometric_loss_gate_v1",
            "name": name,
            "geometric": geometric,
            "lambda": lam if geometric else 0.0,
            "distance_cap_mm": DISTANCE_CAP_MM,
            "state_dict": model.state_dict(),
            "epochs": pilot.EPOCHS,
            "seed": pilot.SEED,
            "parameter_count": parameter_count(model),
        },
        temporary,
    )
    os.replace(temporary, destination)
    payload = {
        "name": name,
        "geometric": geometric,
        "lambda": lam if geometric else 0.0,
        "seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak,
        "history": history,
    }
    write_json(history_path, payload)
    return payload


def initial_state():
    path = OUTPUT / "init.pt"
    if path.is_file():
        return torch.load(path, map_location="cpu", weights_only=True)
    torch.manual_seed(pilot.SEED)
    model = PointwiseMatchedGate(65, width=11, experts=4)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    torch.save(state, path)
    return state


def load_named(path: Path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = PointwiseMatchedGate(65, width=11, experts=4)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def save_geometry_figure(path, image, prediction, target, affine, spacing, case_id, region, labels):
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
        ("Frozen Beta 1", plane(prediction["frozen_beta1"]), "#f4845f"),
        ("Pointwise baseline", plane(prediction["baseline"]), "#f8f9fa"),
        ("Geometric loss", plane(prediction["geometric"]), "#7bdff2"),
    )
    figure, axes = plt.subplots(1, len(panels), figsize=(12.8, 3.6), constrained_layout=True)
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
    figure.suptitle(
        f"{case_id}  {region}  constant-{codes[axial]} slice {slice_index} "
        f"({slice_index * float(spacing[axial]):.1f} mm)  in-plane {in_plane[0]} x {in_plane[1]}",
        fontsize=10,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return {"axis": int(axial), "axis_code": codes[axial], "slice_index": slice_index}


def evaluate_saved_case(case_id: str, record, cfg, splits, expert_hashes) -> None:
    from brats_debate.config import SEGMENTATION_EXPERTS
    from brats_debate.controller.lowmem_store import stream_stored_gates
    from brats_debate.inference.pipeline import load_controller

    path = OUTPUT / "evaluation" / f"{case_id}.json"
    if path.is_file():
        return
    patient = load_patient(record, cfg)
    image = torch.from_numpy(np.ascontiguousarray(patient["image"]))[None]
    target = np.asarray(patient["label"])
    controller, _hash = load_controller(cfg, splits, {name: expert_hashes[name] for name in SEGMENTATION_EXPERTS})
    baseline = load_named(OUTPUT / "baseline.pt")
    geometric = load_named(OUTPUT / "geometric.pt")
    started = time.perf_counter()
    labels, stats = stream_stored_gates(
        {"frozen_beta1": controller, "baseline": baseline, "geometric": geometric},
        image,
        OUTPUT / "cases" / case_id,
        cfg,
        slab_depth=pilot.SLAB_DEPTH,
        halo=1,
    )
    transformer = np.load(OUTPUT / "cases" / case_id / "transformer" / "segmentation.npy")
    if transformer.ndim == 4:
        transformer = transformer[0]
    prediction = {
        "transformer": transformer.astype(np.uint8),
        "frozen_beta1": labels["frozen_beta1"].numpy(),
        "baseline": labels["baseline"].numpy(),
        "geometric": labels["geometric"].numpy(),
    }
    spacing = tuple(float(value) for value in patient["spacing"])
    metrics = {name: pilot.region_scores(volume, target, spacing) for name, volume in prediction.items()}
    stories = {
        method: {
            region: pilot.component_story(volume, target, region_labels, spacing, metrics[method][region]["hd95_mm"])
            for region, region_labels in pilot.REGIONS.items()
        }
        for method, volume in prediction.items()
    }
    summary = {
        "patient_id": case_id,
        "gate_seconds": time.perf_counter() - started,
        "controller_forward_seconds": stats["forward_seconds"],
        "metrics": metrics,
        "components": stories,
        "fusion_checks": {
            "max_probability_sum_error": stats["max_probability_sum_error"],
            "max_weight_sum_error": stats["max_weight_sum_error"],
            "max_unavailable_expert_weight": stats["max_unavailable_expert_weight"],
        },
    }
    summary["paired_geometric_minus"] = {}
    for reference_name in ("baseline", "frozen_beta1", "transformer"):
        summary["paired_geometric_minus"][reference_name] = {
            "mean_dice": metrics["geometric"]["mean_dice"] - metrics[reference_name]["mean_dice"],
            "regions": {
                region: {
                    "dice": metrics["geometric"][region]["dice"] - metrics[reference_name][region]["dice"],
                    "hd95_mm": None
                    if metrics["geometric"][region]["hd95_mm"] is None or metrics[reference_name][region]["hd95_mm"] is None
                    else metrics["geometric"][region]["hd95_mm"] - metrics[reference_name][region]["hd95_mm"],
                    "false_positive_islands": stories["geometric"][region]["false_positive_islands"],
                    "false_positive_island_voxels": stories["geometric"][region]["false_positive_island_voxels"],
                    "reference_false_positive_islands": stories[reference_name][region]["false_positive_islands"],
                    "missed_gt_components": stories["geometric"][region]["missed_gt_components"],
                }
                for region in pilot.REGIONS
            },
        }
    summary["figures"] = {
        region: save_geometry_figure(
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
        for region, region_labels in pilot.REGIONS.items()
    }
    write_json(path, summary)
    print(json.dumps({"evaluated": case_id, "mean_dice": {name: value["mean_dice"] for name, value in metrics.items()}}), flush=True)


def completed_training():
    """Return the saved protocol when both controllers are already trained."""

    protocol_path = OUTPUT / "protocol.json"
    required = (
        OUTPUT / "baseline.pt",
        OUTPUT / "geometric.pt",
        OUTPUT / "training_baseline.json",
        OUTPUT / "training_geometric.json",
        protocol_path,
    )
    if not all(path.is_file() for path in required):
        return None
    protocol = json.loads(protocol_path.read_text())
    recorded = float(protocol["lambda"])
    if abs(recorded - 0.0005969823474760704) > 1e-12:
        raise BenchmarkStop(f"recorded lambda is {recorded}, not 0.000596982")
    if list(protocol["evaluation_cases"]) != list(EVAL_CASES):
        raise BenchmarkStop("the saved evaluation subjects differ from the predeclared list")
    if int(protocol["epochs"]) != pilot.EPOCHS or int(protocol["seed"]) != pilot.SEED:
        raise BenchmarkStop("the saved protocol does not match the 40-epoch training budget")
    for name, geometric in (("baseline", False), ("geometric", True)):
        saved = torch.load(OUTPUT / f"{name}.pt", map_location="cpu", weights_only=False)
        history = json.loads((OUTPUT / f"training_{name}.json").read_text())
        if saved.get("format") != "beta2_geometric_loss_gate_v1":
            raise BenchmarkStop(f"{name}.pt is not a geometric-loss checkpoint")
        if bool(saved.get("geometric")) != geometric:
            raise BenchmarkStop(f"{name}.pt does not match its training role")
        if int(saved.get("epochs", 0)) != pilot.EPOCHS or int(saved.get("seed", -1)) != pilot.SEED:
            raise BenchmarkStop(f"{name}.pt was not trained for the full recorded budget")
        if geometric and abs(float(saved.get("lambda", -1)) - recorded) > 1e-12:
            raise BenchmarkStop("geometric.pt lambda does not match the protocol")
        if len(history.get("history", [])) != pilot.EPOCHS:
            raise BenchmarkStop(f"{name} training history is not 40 epochs")
        model = PointwiseMatchedGate(65, width=11, experts=4)
        model.load_state_dict(saved["state_dict"])
        if parameter_count(model) != 774:
            raise BenchmarkStop(f"{name}.pt does not have 774 parameters")
        del model
    return protocol


def monitor_subprocess(args: list[str]) -> None:
    """Run one worker in a separate process and apply the low-memory stop rules."""

    import beta2_spatial_lowmem as low

    low.assert_child_can_start()
    start_swap = swap_used_mib() or 0
    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *args], cwd=ROOT)
    low_pressure = 0
    try:
        while process.poll() is None:
            time.sleep(2)
            try:
                rss = int(psutil.Process(process.pid).memory_info().rss)
            except psutil.NoSuchProcess:
                break
            swap = swap_used_mib() or 0
            pressure = memory_free_percent()
            growth = swap - start_swap
            if rss > low.MAX_RSS_BYTES:
                process.terminate()
                raise BenchmarkStop(f"worker RSS reached {rss / 1024**3:.2f} GiB")
            if growth > 1500 or (growth > low.MAX_SWAP_GROWTH_MIB and pressure is not None and pressure < 25):
                process.terminate()
                raise BenchmarkStop(f"swap grew by {growth:.0f} MiB during {args} while memory free was {pressure}%")
            if pressure is not None and pressure <= low.CRITICAL_FREE_PERCENT:
                low_pressure += 1
            else:
                low_pressure = 0
            if low_pressure >= 2:
                process.terminate()
                raise BenchmarkStop(f"memory free percentage stayed at {pressure}%")
            if disk_free() < STOP_FREE_BYTES:
                process.terminate()
                raise BenchmarkStop("internal free space fell below 20 GiB")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
    code = process.wait()
    print(json.dumps({"worker": args, "exit": code, "output": str(OUTPUT)}), flush=True)
    if code != 0:
        raise BenchmarkStop(f"worker {args} failed with exit {code}")


def run_expert_worker(case_id: str, expert_name: str) -> None:
    import gc

    import beta2_spatial_lowmem as low
    from brats_debate.controller.lowmem_store import expert_complete, proposal_from_experts, save_expert_compact
    from brats_debate.inference.patches import sliding_predict
    from brats_debate.inference.pipeline import load_experts
    from brats_debate.uncertainty.extraction import _compact_expert_output
    from beta2_stability_benchmark import checkpoint_hashes

    low.OUTPUT = OUTPUT
    destination = low.expert_dir(case_id, expert_name)
    if "geometric_loss_" not in str(destination):
        raise BenchmarkStop(f"refusing to write expert output outside the geometric experiment: {destination}")
    print(json.dumps({"expert_output": str(destination)}), flush=True)
    low.assert_child_can_start()
    if expert_complete(destination):
        print(json.dumps({"skipped_expert": case_id, "expert": expert_name}), flush=True)
        return
    if destination.exists():
        shutil.rmtree(destination)
    cfg, records, splits = evaluation_context()
    by_id = verify_subjects(cfg, records, splits)
    expert_hashes, controller_hash = checkpoint_hashes(cfg)
    if expert_hashes | {"controller": controller_hash} != pilot.EXPECTED_CHECKPOINTS:
        raise BenchmarkStop("frozen checkpoint hashes do not match")
    if case_id not in by_id:
        raise BenchmarkStop(f"{case_id} is not one of the predeclared evaluation scans")
    patient = load_patient(by_id[case_id], cfg)
    if patient["patient_id"] != case_id:
        raise RuntimeError("loaded case does not match the requested id")
    image = torch.from_numpy(np.ascontiguousarray(patient["image"]))[None]
    proposal = None
    if expert_name == "highres":
        proposal = proposal_from_experts(low.case_dir(case_id), threshold=float(cfg["inference"]["uncertainty_threshold"]))
    models, _loaded = load_experts(cfg, splits, names=(expert_name,))
    model = models[expert_name]
    started = time.perf_counter()
    print(json.dumps({"stage": "forward_begin", "rss_mib": psutil.Process().memory_info().rss / 1024**2}), flush=True)
    rich = sliding_predict(
        model,
        image,
        cfg["highres_patch_size"] if expert_name == "highres" else cfg["crop_size"],
        torch.device("cpu"),
        cfg["inference"]["overlap"],
        cfg["inference"]["mc_samples"],
        proposal,
        cfg["inference"]["highres_max_patches"] if expert_name == "highres" else None,
    )
    compact = _compact_expert_output(expert_name, rich)
    save_expert_compact(destination, compact)
    done = json.loads((destination / "done.json").read_text())
    done.update(
        {
            "patient_id": case_id,
            "expert": expert_name,
            "device": "cpu",
            "seconds": time.perf_counter() - started,
            "spatial_shape": list(image.shape[-3:]),
        }
    )
    (destination / "done.json").write_text(json.dumps(done))
    print(json.dumps({"expert_done": case_id, "expert": expert_name, "seconds": done["seconds"]}), flush=True)
    del rich, compact, model, image, patient
    gc.collect()


def run_score_worker(case_id: str) -> None:
    from beta2_stability_benchmark import checkpoint_hashes

    cfg, records, splits = evaluation_context()
    by_id = verify_subjects(cfg, records, splits)
    expert_hashes, controller_hash = checkpoint_hashes(cfg)
    if expert_hashes | {"controller": controller_hash} != pilot.EXPECTED_CHECKPOINTS:
        raise BenchmarkStop("frozen checkpoint hashes do not match")
    evaluate_saved_case(case_id, by_id[case_id], cfg, splits, expert_hashes)


def evaluation_context():
    from brats_debate.config import load_config
    from brats_debate.data.brats_dataset import discover_patients, make_splits, validate_training_audit
    from brats_debate.training.train_controller import assert_split_isolation

    cfg = load_config(ROOT / "configs" / "brats_controller.yaml")
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    assert_split_isolation(splits)
    validate_training_audit(cfg)
    return cfg, records, splits


def run_evaluation() -> None:
    import beta2_spatial_lowmem as low

    if disk_free() < 25 * 1024**3:
        raise BenchmarkStop(
            f"internal free space is {disk_free() / 1024**3:.2f} GiB. Evaluation needs about 2 GiB of temporary expert shards and is refused below 25 GiB so free space stays above 20 GiB."
        )
    low.OUTPUT = OUTPUT
    for case_id in EVAL_CASES:
        if (OUTPUT / "evaluation" / f"{case_id}.json").is_file():
            print(json.dumps({"skipped_evaluation": case_id}), flush=True)
            continue
        for name in ("cnn", "transformer", "boundary", "highres"):
            if (OUTPUT / "cases" / case_id / name / "done.json").is_file():
                continue
            monitor_subprocess(["expert", case_id, name])
        monitor_subprocess(["score", case_id])
        low.remove_experts(case_id)
        write_json(OUTPUT / "progress.json", progress_payload("running", None, 0))
    cases = []
    for case_id in EVAL_CASES:
        path = OUTPUT / "evaluation" / f"{case_id}.json"
        if path.is_file():
            cases.append(json.loads(path.read_text()))
    if len(cases) == len(EVAL_CASES):
        methods = ("transformer", "frozen_beta1", "baseline", "geometric")
        aggregate = {"mean_dice": {}, "mean_hd95_mm": {}}
        for method in methods:
            aggregate["mean_dice"][method] = float(np.mean([case["metrics"][method]["mean_dice"] for case in cases]))
            aggregate["mean_hd95_mm"][method] = {
                region: float(np.mean([
                    case["metrics"][method][region]["hd95_mm"]
                    for case in cases
                    if case["metrics"][method][region]["hd95_mm"] is not None
                ]))
                for region in pilot.REGIONS
            }
        write_json(OUTPUT / "report.json", {"status": "complete", "cases": cases, "aggregate": aggregate})

def progress_payload(status: str, error, seconds: float) -> dict:
    return {
        "status": status,
        "error": error,
        "seconds": seconds,
        "geometry_complete": [
            case_id for case_id in pilot.TRAIN_CASES if (OUTPUT / "geometry" / f"{case_id}.npz").is_file()
        ],
        "baseline_checkpoint": (OUTPUT / "baseline.pt").is_file(),
        "geometric_checkpoint": (OUTPUT / "geometric.pt").is_file(),
        "evaluation_complete": [
            case_id for case_id in EVAL_CASES if (OUTPUT / "evaluation" / f"{case_id}.json").is_file()
        ],
        "disk_free_bytes": disk_free(),
        "swap_used_mib": swap_used_mib(),
        "memory_free_percent": memory_free_percent(),
        "resume_command": "caffeinate -dimsu /opt/anaconda3/bin/python scripts/beta2_geometric_loss.py",
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status, error = "running", None
    try:
        from beta2_stability_benchmark import checkpoint_hashes
        import beta2_spatial_lowmem as low

        saved = completed_training()
        if saved is None:
            from brats_debate.config import load_config
            from brats_debate.data.brats_dataset import discover_patients, make_splits, validate_training_audit
            from brats_debate.training.train_controller import assert_split_isolation

            cfg = load_config(ROOT / "configs" / "brats_controller.yaml")
            records = discover_patients(cfg)
            splits = make_splits(records, cfg)
            assert_split_isolation(splits)
            validate_training_audit(cfg)
            by_id = verify_subjects(cfg, records, splits)
            expert_hashes, controller_hash = checkpoint_hashes(cfg)
            if expert_hashes | {"controller": controller_hash} != pilot.EXPECTED_CHECKPOINTS:
                raise BenchmarkStop("frozen checkpoint hashes do not match")
            if memory_free_percent() is not None and memory_free_percent() < 15:
                raise BenchmarkStop("system memory free is below 15 percent")
            for case_id in pilot.TRAIN_CASES:
                build_geometry(case_id, by_id[case_id], cfg)
            stacked, pools = load_training()
            _subjects, schedule = pilot.training_schedule(pools)
            state = initial_state()
            probe = PointwiseMatchedGate(65, width=11, experts=4)
            probe.load_state_dict(state)
            measured = measure_lambda(probe, stacked, schedule)
            protocol = {
                "train_cases": list(pilot.TRAIN_CASES),
                "evaluation_cases": list(EVAL_CASES),
                "seed": pilot.SEED,
                "epochs": pilot.EPOCHS,
                "optimizer": "Adam",
                "learning_rate": pilot.LEARNING_RATE,
                "weight_decay": pilot.WEIGHT_DECAY,
                "grad_clip": pilot.GRAD_CLIP,
                "architecture": "PointwiseMatchedGate width 11, 774 parameters",
                "baseline_loss": "voxel NLL plus one minus the mean soft Dice of classes 1, 2, and 3",
                "geometric_loss": "mean of WT, TC, and ET distance-to-set and interior-to-surface terms",
                "ground_truth_used_for_inference": False,
                "checkpoint_sha256": expert_hashes | {"controller": controller_hash},
                **measured,
            }
            write_json(OUTPUT / "protocol.json", protocol)
            lam = float(protocol["lambda"])
            del probe
            train_model("baseline", False, lam, state, stacked, schedule)
            train_model("geometric", True, lam, state, stacked, schedule)
            del stacked
        else:
            print(json.dumps({"skipped_training": True, "lambda": saved["lambda"], "output": str(OUTPUT)}), flush=True)
        low.assert_child_can_start()
        run_evaluation()
        status = "complete"
    except Exception as exc:
        status = "stopped" if isinstance(exc, BenchmarkStop) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"status": status, "error": error}), flush=True)
        if not isinstance(exc, BenchmarkStop):
            raise
    finally:
        write_json(
            OUTPUT / "progress.json",
            {
                "status": status,
                "error": error,
                "seconds": time.perf_counter() - started,
                "geometry_complete": [
                    case_id for case_id in pilot.TRAIN_CASES if (OUTPUT / "geometry" / f"{case_id}.npz").is_file()
                ],
                "baseline_checkpoint": (OUTPUT / "baseline.pt").is_file(),
                "geometric_checkpoint": (OUTPUT / "geometric.pt").is_file(),
                "evaluation_complete": [
                    case_id for case_id in EVAL_CASES if (OUTPUT / "evaluation" / f"{case_id}.json").is_file()
                ],
                "disk_free_bytes": disk_free(),
                "swap_used_mib": swap_used_mib(),
                "memory_free_percent": memory_free_percent(),
                "resume_command": "caffeinate -dimsu /opt/anaconda3/bin/python scripts/beta2_geometric_loss.py",
            },
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "expert":
        try:
            run_expert_worker(sys.argv[2], sys.argv[3])
        except BenchmarkStop as exc:
            print(json.dumps({"status": "stopped", "error": str(exc)}), flush=True)
            raise SystemExit(3) from exc
    elif len(sys.argv) > 1 and sys.argv[1] == "score":
        try:
            run_score_worker(sys.argv[2])
        except BenchmarkStop as exc:
            print(json.dumps({"status": "stopped", "error": str(exc)}), flush=True)
            raise SystemExit(3) from exc
    else:
        main()
