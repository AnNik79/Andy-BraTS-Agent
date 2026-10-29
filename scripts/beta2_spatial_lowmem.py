"""Low-memory spatial pilot.

The original ``beta2_spatial_pilot.py`` and its swap gate stay as they are.
This runner executes the same subjects, losses, and epoch budget one expert
at a time on CPU. Patch forwards on this Mac used about 1.4 GiB of MPS
driver memory and were no faster than CPU, while CPU and MPS probabilities
on a 64-cubed patch differed by less than 1e-6.

Full expert volumes are depth-sharded, then deleted after the patches or
the evaluation labels have been saved. Workers are subprocesses so the
memory is returned when each expert exits.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import psutil

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))

import beta2_spatial_pilot as pilot
from beta2_stability_benchmark import BenchmarkStop, memory_free_percent, swap_used_mib

OUTPUT = ROOT / "outputs" / "beta2" / "spatial_controller_pilot" / "lowmem_20261025"
REFERENCE = ROOT / "outputs" / "beta2" / "oracle_gap_20261019" / "cases" / "BraTS-GLI-00006-000.npz"
FEASIBILITY_CASE = "BraTS-GLI-00006-000"
SCRIPT = Path(__file__).resolve()
STOP_FREE_BYTES = 20 * 1024**3
START_FREE_BYTES = 25 * 1024**3
MIN_AVAILABLE_BYTES = int(3.5 * 1024**3)
MAX_RSS_BYTES = int(4.0 * 1024**3)
MAX_SWAP_GROWTH_MIB = 400
MIN_FREE_PERCENT = 15
CRITICAL_FREE_PERCENT = 10
EXPERT_PROB_ATOL = 1e-4
CONTROLLER_PROB_ATOL = 1e-3
WALL_SECONDS = 6 * 60 * 60
CHILD = None


def case_dir(case_id: str) -> Path:
    return OUTPUT / "cases" / case_id


def expert_dir(case_id: str, name: str) -> Path:
    return case_dir(case_id) / name


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(pilot.as_json(payload), indent=2))
    os.replace(temporary, path)


def disk_free() -> int:
    return shutil.disk_usage(ROOT).free


def resource_snapshot() -> dict:
    memory = psutil.virtual_memory()
    return {
        "available_bytes": int(memory.available),
        "memory_free_percent": memory_free_percent(),
        "swap_used_mib": swap_used_mib(),
        "disk_free_bytes": disk_free(),
        "rss_bytes": int(psutil.Process().memory_info().rss),
    }


def assert_child_can_start() -> None:
    snapshot = resource_snapshot()
    if snapshot["disk_free_bytes"] < START_FREE_BYTES:
        raise BenchmarkStop(f"disk free is {snapshot['disk_free_bytes'] / 1024**3:.2f} GiB")
    if snapshot["available_bytes"] < MIN_AVAILABLE_BYTES:
        raise BenchmarkStop(f"available RAM is {snapshot['available_bytes'] / 1024**3:.2f} GiB, below the 2 GiB reserve")
    if snapshot["memory_free_percent"] is not None and snapshot["memory_free_percent"] < MIN_FREE_PERCENT:
        raise BenchmarkStop(f"system memory free is {snapshot['memory_free_percent']}%")


def run_worker(args: list[str]) -> None:
    global CHILD
    assert_child_can_start()
    start_swap = swap_used_mib() or 0
    CHILD = subprocess.Popen([sys.executable, str(SCRIPT), *args], cwd=ROOT)
    low_pressure = 0
    peak_rss = 0
    try:
        while CHILD.poll() is None:
            time.sleep(2)
            try:
                rss = int(psutil.Process(CHILD.pid).memory_info().rss)
            except psutil.NoSuchProcess:
                break
            peak_rss = max(peak_rss, rss)
            swap = swap_used_mib() or 0
            pressure = memory_free_percent()
            swap_growth = swap - start_swap
            if rss > MAX_RSS_BYTES:
                CHILD.terminate()
                raise BenchmarkStop(f"worker RSS reached {rss / 1024**3:.2f} GiB")
            # macOS can write inactive Chrome pages to swap while this worker
            # stays near 1 GiB and system memory remains about half free.
            # Stop for swap only when that write coincides with low free memory,
            # or when one worker grows swap by more than 1.5 GiB.
            if swap_growth > 1500 or (swap_growth > MAX_SWAP_GROWTH_MIB and pressure is not None and pressure < 25):
                CHILD.terminate()
                raise BenchmarkStop(
                    f"swap grew by {swap_growth:.0f} MiB during {args} while memory free was {pressure}%"
                )
            if pressure is not None and pressure <= CRITICAL_FREE_PERCENT:
                low_pressure += 1
            else:
                low_pressure = 0
            if low_pressure >= 2:
                CHILD.terminate()
                raise BenchmarkStop(f"memory free percentage stayed at {pressure}%")
            if disk_free() < STOP_FREE_BYTES:
                CHILD.terminate()
                raise BenchmarkStop("internal free space fell below 20 GiB")
    finally:
        if CHILD is not None and CHILD.poll() is None:
            CHILD.terminate()
            try:
                CHILD.wait(timeout=20)
            except subprocess.TimeoutExpired:
                CHILD.kill()
                CHILD.wait(timeout=10)
    code = CHILD.returncode
    print(json.dumps({"worker": args, "exit": code, "peak_rss_mib": peak_rss / 1024**2}), flush=True)
    if code == 2:
        raise BenchmarkStop("the low-memory outputs do not match the saved reference")
    if code == 3:
        raise BenchmarkStop("a worker stopped because its own memory reserve was gone")
    if code != 0:
        raise BenchmarkStop(f"worker {args} failed with exit {code}")


def time_left(started: float) -> float:
    return WALL_SECONDS - (time.perf_counter() - started)


def remove_experts(case_id: str) -> None:
    for name in ("cnn", "transformer", "boundary", "highres"):
        directory = expert_dir(case_id, name)
        if directory.exists():
            shutil.rmtree(directory)


def manifest_from_disk() -> dict:
    experts = {}
    patches = []
    for case_id in pilot.TRAIN_CASES + pilot.EVAL_CASES:
        done = [name for name in ("cnn", "transformer", "boundary", "highres") if (expert_dir(case_id, name) / "done.json").is_file()]
        if done:
            experts[case_id] = done
        if (OUTPUT / "patches" / f"{case_id}.npz").is_file():
            patches.append(case_id)
    return {
        "experts_complete": experts,
        "patches_complete": patches,
        "pointwise_checkpoint": (OUTPUT / "pointwise.pt").is_file(),
        "spatial_checkpoint": (OUTPUT / "spatial.pt").is_file(),
        "evaluation_complete": [
            case_id for case_id in pilot.EVAL_CASES if (OUTPUT / "evaluation" / f"{case_id}.json").is_file()
        ],
        "feasibility_written": (OUTPUT / "feasibility_00006.json").is_file(),
    }


def prepare_context():
    from brats_debate.config import load_config
    from brats_debate.data.brats_dataset import discover_patients, make_splits, validate_training_audit
    from brats_debate.training.train_controller import assert_split_isolation
    from beta2_stability_benchmark import checkpoint_hashes

    cfg = load_config(ROOT / "configs" / "brats_controller.yaml")
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    assert_split_isolation(splits)
    validate_training_audit(cfg)
    _protocol, by_id, identities = pilot.require_protocol_and_split(cfg, records, splits)
    expert_hashes, controller_hash = checkpoint_hashes(cfg)
    if expert_hashes | {"controller": controller_hash} != pilot.EXPECTED_CHECKPOINTS:
        raise BenchmarkStop("frozen checkpoint hashes do not match the manifest")
    return cfg, splits, by_id, identities, expert_hashes | {"controller": controller_hash}


def worker_expert(case_id: str, expert_name: str) -> None:
    import gc

    import numpy as np
    import torch

    from brats_debate.controller.lowmem_store import expert_complete, proposal_from_experts, save_expert_compact
    from brats_debate.inference.patches import sliding_predict
    from brats_debate.inference.pipeline import load_experts
    from brats_debate.uncertainty.extraction import _compact_expert_output
    from beta2_adaptive_pilot import load_patient

    try:
        assert_child_can_start()
        destination = expert_dir(case_id, expert_name)
        if expert_complete(destination):
            print(json.dumps({"skipped_expert": case_id, "expert": expert_name}), flush=True)
            return
        if destination.exists():
            shutil.rmtree(destination)
        cfg, splits, by_id, _identities, _hashes = prepare_context()
        print(json.dumps({"stage": "context", "rss_mib": psutil.Process().memory_info().rss / 1024**2}), flush=True)
        patient = load_patient(by_id[case_id], cfg)
        print(json.dumps({"stage": "patient", "rss_mib": psutil.Process().memory_info().rss / 1024**2}), flush=True)
        if patient["patient_id"] != case_id:
            raise RuntimeError("loaded case does not match the requested id")
        image = torch.from_numpy(np.ascontiguousarray(patient["image"]))[None]
        proposal = None
        if expert_name == "highres":
            proposal = proposal_from_experts(case_dir(case_id), threshold=float(cfg["inference"]["uncertainty_threshold"]))
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
    except BenchmarkStop as exc:
        print(f"RESOURCE {exc}", flush=True)
        raise SystemExit(3) from exc


def worker_patches(case_id: str) -> None:
    import numpy as np
    import torch

    from brats_debate.controller.lowmem_store import crop_loaded, load_expert_depth
    from brats_debate.controller.spatial_stream import build_gate_inputs
    from brats_debate.config import SEGMENTATION_EXPERTS
    from beta2_adaptive_pilot import load_patient

    assert_child_can_start()
    destination = OUTPUT / "patches" / f"{case_id}.npz"
    sidecar = OUTPUT / "patches" / f"{case_id}.json"
    if destination.is_file() and sidecar.is_file():
        print(json.dumps({"skipped_patches": case_id}), flush=True)
        return
    cfg, _splits, by_id, _identities, _hashes = prepare_context()
    patient = load_patient(by_id[case_id], cfg)
    label = np.asarray(patient["label"])
    image = torch.from_numpy(np.ascontiguousarray(patient["image"]))[None]
    quota = pilot.patches_for_case(case_id)
    tumor_count = round(quota * pilot.TUMOR_PATCHES / pilot.PATCHES_PER_SUBJECT)
    other_count = quota - tumor_count
    rng = np.random.default_rng(np.random.SeedSequence([pilot.SEED, pilot.TRAIN_CASES.index(case_id)]))
    tumor_origins, other_origins = pilot.sample_origins(label, label.shape, rng, tumor_count, other_count)
    if len(tumor_origins) < max(4, tumor_count // 2):
        raise BenchmarkStop(f"{case_id} produced only {len(tumor_origins)} tumor windows")
    features, probabilities, availability, targets, origins = [], [], [], [], []
    for origin in tumor_origins + other_origins:
        z0, y0, x0 = (origin[axis] - pilot.HALO for axis in range(3))
        z1, y1, x1 = (origin[axis] + pilot.INTERIOR + pilot.HALO for axis in range(3))
        outputs = {
            name: crop_loaded(load_expert_depth(expert_dir(case_id, name), z0, z1), y0, y1, x0, x1)
            for name in SEGMENTATION_EXPERTS
        }
        window_features, window_probabilities, window_availability = build_gate_inputs(
            image[..., z0:z1, y0:y1, x0:x1], outputs
        )
        if window_features.shape[1] != 65:
            raise RuntimeError("controller feature width is not 65")
        features.append(window_features[0].contiguous().numpy())
        probabilities.append(window_probabilities[0].contiguous().numpy())
        availability.append(window_availability[0].contiguous().numpy())
        targets.append(
            np.ascontiguousarray(
                label[
                    origin[0] : origin[0] + pilot.INTERIOR,
                    origin[1] : origin[1] + pilot.INTERIOR,
                    origin[2] : origin[2] + pilot.INTERIOR,
                ],
                dtype=np.int16,
            )
        )
        origins.append(origin)
    payload = {
        "features": np.stack(features).astype(np.float32),
        "probabilities": np.stack(probabilities).astype(np.float32),
        "availability": np.stack(availability).astype(np.float32),
        "target": np.stack(targets),
        "origins": np.asarray(origins, dtype=np.int16),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".partial.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, destination)
    write_json(
        sidecar,
        {
            "patient_id": case_id,
            "subject_id": pilot.subject_of(case_id),
            "channels": 65,
            "interior": pilot.INTERIOR,
            "halo": pilot.HALO,
            "tumor_windows": len(tumor_origins),
            "other_windows": len(other_origins),
            "windows": len(origins),
            "ground_truth_used_for_inference": False,
            "ground_truth_used_for_patch_location": True,
            "ground_truth_used_as_feature": False,
            "highres_full_volume_used": False,
            "execution": "cpu_one_expert_subprocess",
        },
    )
    print(json.dumps({"patches_done": case_id, "windows": len(origins)}), flush=True)


def worker_reference(case_id: str) -> None:
    import numpy as np
    import torch

    from brats_debate.config import SEGMENTATION_EXPERTS
    from brats_debate.controller.gating_network import GatingNetwork
    from brats_debate.controller.lowmem_store import gather_probabilities, stream_stored_gates
    from brats_debate.inference.pipeline import load_controller
    from beta2_adaptive_pilot import load_patient

    if case_id != FEASIBILITY_CASE:
        raise RuntimeError("the saved reference comparison is only defined for BraTS-GLI-00006-000")
    cfg, splits, by_id, _identities, expert_hashes = prepare_context()
    reference = np.load(REFERENCE)
    flat = reference["flat_index"]
    saved = reference["expert_probabilities"]
    report = {"patient_id": case_id, "experts": {}, "tolerance": {"expert_probability_abs": EXPERT_PROB_ATOL}}
    failed = False
    for index, name in enumerate(SEGMENTATION_EXPERTS):
        got = gather_probabilities(expert_dir(case_id, name), flat)
        difference = np.abs(got - saved[:, index])
        agreement = float((got.argmax(1) == saved[:, index].argmax(1)).mean())
        item = {"max_abs": float(difference.max()), "mean_abs": float(difference.mean()), "argmax_agreement": agreement}
        report["experts"][name] = item
        if item["max_abs"] > EXPERT_PROB_ATOL or agreement < 0.999:
            failed = True
    patient = load_patient(by_id[case_id], cfg)
    image = torch.from_numpy(np.ascontiguousarray(patient["image"]))[None]
    controller, _hash = load_controller(cfg, splits, {name: expert_hashes[name] for name in SEGMENTATION_EXPERTS})
    _labels, stats = stream_stored_gates(
        {"frozen_beta1": controller},
        image,
        case_dir(case_id),
        cfg,
        slab_depth=pilot.SLAB_DEPTH,
        halo=1,
        capture_flat=flat,
    )
    fused = stats["captured_probabilities"]["frozen_beta1"]
    reference_controller = reference["controller_probabilities"]
    difference = np.abs(fused - reference_controller)
    agreement = float((fused.argmax(1) == reference_controller.argmax(1)).mean())
    report["frozen_controller"] = {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "argmax_agreement": agreement,
    }
    if report["frozen_controller"]["max_abs"] > CONTROLLER_PROB_ATOL or agreement < 0.99:
        failed = True
    report["passed"] = not failed
    write_json(OUTPUT / "feasibility_00006.json", report)
    print(json.dumps({"feasibility": report["passed"], "controller_max_abs": report["frozen_controller"]["max_abs"]}), flush=True)
    if failed:
        raise SystemExit(2)


def worker_train() -> None:
    pilot.OUTPUT = OUTPUT
    cfg_channels = 65
    stacked, pools = pilot.load_patches(cfg_channels)
    _subjects, schedule = pilot.training_schedule(pools)
    curves = {}
    for kind in ("pointwise", "spatial"):
        checkpoint = OUTPUT / f"{kind}.pt"
        if checkpoint.is_file():
            _model, curves[kind] = pilot.load_trained(kind, cfg_channels)
        else:
            _model, curves[kind] = pilot.train_one(kind, stacked, schedule, cfg_channels)
        del _model
    pilot.save_loss_figure(curves)
    write_json(OUTPUT / "training_loss.json", curves)
    print(json.dumps({"training_done": {name: values[-1] for name, values in curves.items()}}), flush=True)


def worker_evaluate(case_id: str) -> None:
    import numpy as np
    import torch

    from brats_debate.config import SEGMENTATION_EXPERTS
    from brats_debate.controller.lowmem_store import stream_stored_gates
    from brats_debate.inference.pipeline import load_controller
    from beta2_adaptive_pilot import load_patient

    pilot.OUTPUT = OUTPUT
    path = OUTPUT / "evaluation" / f"{case_id}.json"
    if path.is_file():
        print(json.dumps({"skipped_evaluation": case_id}), flush=True)
        return
    assert_child_can_start()
    cfg, splits, by_id, _identities, expert_hashes = prepare_context()
    patient = load_patient(by_id[case_id], cfg)
    image = torch.from_numpy(np.ascontiguousarray(patient["image"]))[None]
    target = np.asarray(patient["label"])
    controller, _hash = load_controller(cfg, splits, {name: expert_hashes[name] for name in SEGMENTATION_EXPERTS})
    pointwise, _loss = pilot.load_trained("pointwise", 65)
    spatial, _loss = pilot.load_trained("spatial", 65)
    started = time.perf_counter()
    labels, stats = stream_stored_gates(
        {"frozen_beta1": controller, "pointwise": pointwise, "spatial": spatial},
        image,
        case_dir(case_id),
        cfg,
        slab_depth=pilot.SLAB_DEPTH,
        halo=1,
    )
    for name, error in stats["max_probability_sum_error"].items():
        if error > 1e-3 or stats["max_weight_sum_error"][name] > 1e-3:
            raise RuntimeError(f"{name} fusion is not normalized")
        if stats["max_unavailable_expert_weight"][name] > 1e-3:
            raise RuntimeError(f"{name} weighted an unavailable expert")
    transformer = np.load(expert_dir(case_id, "transformer") / "segmentation.npy")
    if transformer.ndim == 4:
        transformer = transformer[0]
    prediction = {
        "transformer": transformer.astype(np.uint8),
        "frozen_beta1": labels["frozen_beta1"].numpy(),
        "pointwise": labels["pointwise"].numpy(),
        "spatial": labels["spatial"].numpy(),
    }
    spacing = tuple(float(value) for value in patient["spacing"])
    metrics = {name: pilot.region_scores(volume, target, spacing) for name, volume in prediction.items()}
    stories = {}
    for method, volume in prediction.items():
        stories[method] = {
            region: pilot.component_story(volume, target, region_labels, spacing, metrics[method][region]["hd95_mm"])
            for region, region_labels in pilot.REGIONS.items()
        }
    comparisons = {}
    for reference_name in ("transformer", "frozen_beta1", "pointwise"):
        comparisons[f"spatial_minus_{reference_name}"] = {}
        for region in pilot.REGIONS:
            candidate = {"dice": metrics["spatial"][region]["dice"], "hd95_mm": metrics["spatial"][region]["hd95_mm"], **stories["spatial"][region]}
            baseline = {
                "dice": metrics[reference_name][region]["dice"],
                "hd95_mm": metrics[reference_name][region]["hd95_mm"],
                **stories[reference_name][region],
            }
            hd_delta = None
            if candidate["hd95_mm"] is not None and baseline["hd95_mm"] is not None:
                hd_delta = candidate["hd95_mm"] - baseline["hd95_mm"]
            comparisons[f"spatial_minus_{reference_name}"][region] = {
                "dice": candidate["dice"] - baseline["dice"],
                "hd95_mm": hd_delta,
                "flags": pilot.classify_disagreement(candidate, baseline),
            }
    changes = {}
    for reference_name in ("transformer", "frozen_beta1", "pointwise"):
        reference_wrong = prediction[reference_name] != target
        spatial_wrong = prediction["spatial"] != target
        changes[reference_name] = {
            "spatial_errors": int(spatial_wrong.sum()),
            "reference_errors": int(reference_wrong.sum()),
            "repaired": int(np.count_nonzero(reference_wrong & ~spatial_wrong)),
            "introduced": int(np.count_nonzero(~reference_wrong & spatial_wrong)),
        }
    figure_meta = {}
    for region, region_labels in pilot.REGIONS.items():
        figure_meta[region] = pilot.save_contour(
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
    label_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = label_path.with_name(label_path.stem + ".partial.npz")
    np.savez_compressed(temporary, **{name: volume.astype(np.uint8) for name, volume in prediction.items()}, target=target.astype(np.uint8))
    os.replace(temporary, label_path)
    summary = {
        "patient_id": case_id,
        "subject_id": pilot.subject_of(case_id),
        "spacing_mm": list(spacing),
        "gate_seconds": time.perf_counter() - started,
        "feature_seconds": stats["feature_seconds"],
        "controller_forward_seconds": stats["forward_seconds"],
        "execution": "cpu_sharded_experts",
        "metrics": metrics,
        "voxel_changes_spatial_versus": changes,
        "components": stories,
        "paired": comparisons,
        "fusion_checks": {
            "max_probability_sum_error": stats["max_probability_sum_error"],
            "max_weight_sum_error": stats["max_weight_sum_error"],
            "max_unavailable_expert_weight": stats["max_unavailable_expert_weight"],
        },
        "figures": figure_meta,
        "resources": resource_snapshot(),
    }
    write_json(path, summary)
    print(json.dumps({"evaluated": case_id, "mean_dice": {name: value["mean_dice"] for name, value in metrics.items()}}), flush=True)


def ensure_experts(case_id: str, started: float) -> None:
    from brats_debate.controller.lowmem_store import expert_complete

    for name in ("cnn", "transformer", "boundary", "highres"):
        if expert_complete(expert_dir(case_id, name)):
            continue
        if time_left(started) < 20 * 60:
            raise BenchmarkStop("fewer than 20 minutes remain before the next expert")
        run_worker(["expert", case_id, name])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert_child_can_start()
    identities = None
    hashes = None
    started = time.perf_counter()
    status, error = "running", None

    def stop_handler(_signum, _frame):
        if CHILD is not None and CHILD.poll() is None:
            CHILD.terminate()
        raise BenchmarkStop("six-hour wall clock")

    signal.signal(signal.SIGALRM, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    signal.alarm(WALL_SECONDS)
    try:
        _cfg, _splits, _by_id, identities, hashes = prepare_context()
        write_json(
            OUTPUT / "predeclared_run.json",
            {
                "execution": "cpu_one_expert_subprocess",
                "why_not_mps": "A warmed 64-cubed patch reserved about 1.4 GiB of MPS driver memory and was not faster than CPU. CPU and MPS probabilities differed by less than 1e-6.",
                "original_swap_gate": "unchanged in scripts/beta2_spatial_pilot.py; this mode does not use that gate because it does not launch the MPS workload",
                "rss_budget_gib": MAX_RSS_BYTES / 1024**3,
                "minimum_available_gib": MIN_AVAILABLE_BYTES / 1024**3,
                "train_cases": list(pilot.TRAIN_CASES),
                "evaluation_cases": list(pilot.EVAL_CASES),
                "epochs": pilot.EPOCHS,
                "seed": pilot.SEED,
                "spatial_regularizer": None,
                "protocol_sha256": pilot.EXPECTED_PROTOCOL_SHA256,
                "checkpoint_sha256": hashes,
                "subjects": identities,
            },
        )
        for case_id in pilot.TRAIN_CASES:
            if (OUTPUT / "patches" / f"{case_id}.npz").is_file():
                print(json.dumps({"skipped_training_case": case_id}), flush=True)
                continue
            ensure_experts(case_id, started)
            if case_id == FEASIBILITY_CASE and not (OUTPUT / "feasibility_00006.json").is_file():
                run_worker(["reference", case_id])
                feasibility = json.loads((OUTPUT / "feasibility_00006.json").read_text())
                if not feasibility.get("passed"):
                    raise BenchmarkStop("feasibility reference check failed")
            run_worker(["patches", case_id])
            remove_experts(case_id)
            write_json(OUTPUT / "manifest.json", manifest_from_disk())
        if not ((OUTPUT / "pointwise.pt").is_file() and (OUTPUT / "spatial.pt").is_file()):
            run_worker(["train"])
        for case_id in pilot.EVAL_CASES:
            if (OUTPUT / "evaluation" / f"{case_id}.json").is_file():
                continue
            ensure_experts(case_id, started)
            run_worker(["evaluate", case_id])
            remove_experts(case_id)
            write_json(OUTPUT / "manifest.json", manifest_from_disk())
        status = "complete"
    except Exception as exc:
        status = "stopped" if isinstance(exc, BenchmarkStop) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"status": status, "error": error}), flush=True)
        if not isinstance(exc, BenchmarkStop):
            raise
    finally:
        signal.alarm(0)
        evaluations = []
        for case_id in pilot.EVAL_CASES:
            path = OUTPUT / "evaluation" / f"{case_id}.json"
            if path.is_file():
                evaluations.append(json.loads(path.read_text()))
        loss_path = OUTPUT / "training_loss.json"
        write_json(
            OUTPUT / "report.json",
            {
                "format": "beta2_spatial_lowmem_v1",
                "status": status,
                "error": error,
                "seconds": time.perf_counter() - started,
                "manifest": manifest_from_disk(),
                "training_loss": json.loads(loss_path.read_text()) if loss_path.is_file() else {},
                "evaluation": evaluations,
                "aggregate": pilot.aggregate(evaluations) if evaluations else {},
                "feasibility": json.loads((OUTPUT / "feasibility_00006.json").read_text())
                if (OUTPUT / "feasibility_00006.json").is_file()
                else None,
                "resume_command": "caffeinate -dimsu /opt/anaconda3/bin/python scripts/beta2_spatial_lowmem.py",
                "scope": "Same predeclared subjects as the spatial protocol. CPU execution. Exploratory fit cohort, not an external test.",
            },
        )
    if status == "failed":
        raise SystemExit(1)


def worker_main() -> None:
    command = sys.argv[1]
    try:
        if command == "expert":
            worker_expert(sys.argv[2], sys.argv[3])
        elif command == "patches":
            worker_patches(sys.argv[2])
        elif command == "reference":
            worker_reference(sys.argv[2])
        elif command == "train":
            worker_train()
        elif command == "evaluate":
            worker_evaluate(sys.argv[2])
        else:
            raise SystemExit(f"unknown worker {command}")
    except BenchmarkStop as exc:
        print(f"RESOURCE {exc}", flush=True)
        raise SystemExit(3) from exc
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        worker_main()
    else:
        main()
