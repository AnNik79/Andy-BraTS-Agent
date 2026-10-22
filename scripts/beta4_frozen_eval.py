"""Frozen Stage 2 versus Beta 1 on 20 unused fit subjects.

No extra HighRes windows. The Stage 2 checkpoint is only read.
"""

import gc
import json
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import beta2_stability_benchmark as stability
from beta2_adaptive_pilot import load_patient, metric_block
from beta2_stability_benchmark import (
    BenchmarkStop,
    CaseResourceMonitor,
    checkpoint_hashes,
    resource_snapshot,
    write_progress,
)
from beta2_stage3_experiment import GATE_PATH, PATCH, REGIONS, SPLIT_PATH, adaptive_label, file_sha256
from brats_debate.config import device_for, load_config
from brats_debate.controller.adaptive_gate import ResidualAdaptiveGate
from brats_debate.data.brats_dataset import discover_patients, make_splits, subject_id, validate_training_audit
from brats_debate.inference.pipeline import load_controller, load_experts
from brats_debate.training.train_controller import assert_split_isolation, records_for
from brats_debate.uncertainty.adaptive_pilot import score_methods, voxel_change_counts
from brats_debate.uncertainty.extraction import predict_experts_compact, stream_controller_and_uncertainty
from brats_debate.uncertainty.stage2_gate import REVISED_PROTOCOL
from freeze_beta2_partitions import validate_frozen_manifest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "beta2" / "stage4_frozen_eval_20261020"
OBSERVED = (
    "BraTS-GLI-00414-000",
    "BraTS-GLI-00451-000",
    "BraTS-GLI-00493-000",
    "BraTS-GLI-00500-000",
    "BraTS-GLI-00501-000",
    "BraTS-GLI-00514-000",
    "BraTS-GLI-00519-000",
    "BraTS-GLI-00523-000",
    "BraTS-GLI-00529-000",
    "BraTS-GLI-00552-000",
)
CASES = (
    "BraTS-GLI-00556-000",
    "BraTS-GLI-00558-000",
    "BraTS-GLI-00559-000",
    "BraTS-GLI-00568-000",
    "BraTS-GLI-00577-000",
    "BraTS-GLI-00579-000",
    "BraTS-GLI-00584-000",
    "BraTS-GLI-00591-000",
    "BraTS-GLI-00649-000",
    "BraTS-GLI-00675-000",
    "BraTS-GLI-00704-000",
    "BraTS-GLI-00709-000",
    "BraTS-GLI-00730-000",
    "BraTS-GLI-00744-000",
    "BraTS-GLI-00747-000",
    "BraTS-GLI-00791-000",
    "BraTS-GLI-00804-000",
    "BraTS-GLI-00834-000",
    "BraTS-GLI-01002-000",
    "BraTS-GLI-01013-000",
)
STOP_FREE_BYTES = 20 * 1024**3
NEXT_GROUP_FREE_BYTES = 23 * 1024**3


def run_case(case_id, record, models, controller, gate, cfg, device):
    if shutil.disk_usage(OUTPUT).free < STOP_FREE_BYTES:
        raise BenchmarkStop("internal free space fell below 20 GiB")
    monitor = CaseResourceMonitor(OUTPUT)
    before = resource_snapshot(OUTPUT)
    monitor.start()
    started = time.perf_counter()
    patient = outputs = None
    try:
        patient = load_patient(record, cfg)
        if patient["patient_id"] != case_id:
            raise RuntimeError("loaded patient does not match the predeclared case")
        image = torch.from_numpy(patient["image"])[None]
        expert_started = time.perf_counter()
        outputs = predict_experts_compact(patient, models, cfg, device)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        final, uncertainty, weights = stream_controller_and_uncertainty(
            controller, image, outputs, cfg, device, slab_depth=8, return_weights=True
        )
        pipeline_seconds = time.perf_counter() - expert_started
        gate_started = time.perf_counter()
        stage2_label, detail = adaptive_label(gate, image, outputs, weights, uncertainty)
        gate_seconds = time.perf_counter() - gate_started
        target = np.asarray(patient["label"])
        frozen_label = final.argmax(1)[0].to(torch.uint8).cpu().numpy()
        predictions = {
            "cnn": detail["cnn"].numpy(),
            "majority": detail["majority"].numpy(),
            "frozen_beta1": frozen_label,
            "stage2_gate": stage2_label,
        }
        metrics = score_methods(predictions, target, patient["spacing"], REGIONS)
        entropy = uncertainty["predictive_entropy"][0, 0].numpy()
        disagreement = detail["expert_disagreement"].numpy()
        changes = voxel_change_counts(frozen_label, stage2_label, target, entropy, disagreement)
        summary = {
            "patient_id": case_id,
            "subject_id": subject_id(record, cfg),
            "spacing_mm": [float(value) for value in patient["spacing"]],
            "seconds": time.perf_counter() - started,
            "pipeline_seconds": pipeline_seconds,
            "adaptive_gate_seconds": gate_seconds,
            "highres_setting": "proposal_limited_no_extra_windows",
            "highres_patch": list(PATCH),
            "metrics": {name: metric_block(value) for name, value in metrics.items()},
            "voxel_changes_versus_beta1": changes,
            "resources": {**monitor.summary(), "before": before, "after": resource_snapshot(OUTPUT)},
        }
    finally:
        del patient, outputs
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        monitor.stop()
    return summary


def bootstrap_mean_difference(differences, draws=10000, seed=20261020):
    values = np.asarray(differences, dtype=np.float64)
    if values.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(1)
    return {
        "subjects": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "improved": int((values > 0).sum()),
        "worsened": int((values < 0).sum()),
        "unchanged": int((values == 0).sum()),
        "bootstrap_draws": draws,
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def aggregate(cases):
    methods = ("cnn", "majority", "frozen_beta1", "stage2_gate")
    regions = ("WT", "TC", "ET")
    summary = {"mean_dice": {}, "mean_region_dice": {}, "mean_hd95_mm": {}}
    for method in methods:
        dices = [case["metrics"][method]["mean_dice"] for case in cases]
        summary["mean_dice"][method] = float(np.mean(dices))
        summary["mean_region_dice"][method] = {
            region: float(np.mean([case["metrics"][method][region]["dice"] for case in cases]))
            for region in regions
        }
        summary["mean_hd95_mm"][method] = {
            region: float(np.mean([case["metrics"][method][region]["hd95_mm"] for case in cases if case["metrics"][method][region]["hd95_mm"] is not None]))
            for region in regions
        }
    paired = [case["metrics"]["stage2_gate"]["mean_dice"] - case["metrics"]["frozen_beta1"]["mean_dice"] for case in cases]
    versus_vote = [case["metrics"]["stage2_gate"]["mean_dice"] - case["metrics"]["majority"]["mean_dice"] for case in cases]
    summary["stage2_minus_beta1_dice"] = bootstrap_mean_difference(paired)
    summary["stage2_minus_majority_dice"] = bootstrap_mean_difference(versus_vote)
    summary["per_subject_stage2_minus_beta1"] = [
        {"patient_id": case["patient_id"], "difference": paired[index]} for index, case in enumerate(cases)
    ]
    changes = [case["voxel_changes_versus_beta1"]["all"] for case in cases]
    summary["voxel_errors"] = {
        "beta1": int(sum(item["frozen_errors"] for item in changes)),
        "stage2": int(sum(item["adaptive_errors"] for item in changes)),
        "repaired": int(sum(item["corrected"] for item in changes)),
        "introduced": int(sum(item["introduced"] for item in changes)),
    }
    return summary


def main():
    if shutil.disk_usage(ROOT).free < 30 * 1024**3:
        raise BenchmarkStop("internal free space is below 30 GiB")
    stability.MINIMUM_FREE_BYTES = STOP_FREE_BYTES
    cfg = load_config(ROOT / "configs" / "brats_controller.yaml")
    stage_b = yaml.safe_load((ROOT / "configs" / "beta2" / "foundation.yaml").read_text())["stage_b"]
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    assert_split_isolation(splits)
    validate_training_audit(cfg)
    manifest = json.loads((ROOT / "configs" / "beta2" / "subject_partitions.json").read_text())
    validate_frozen_manifest(manifest, cfg=cfg, records=records, splits=splits, stage_b_cfg=stage_b)
    stage2_split = json.loads(SPLIT_PATH.read_text())
    used = set(stage2_split["added_training_cases"]) | set(stage2_split["evaluation_cases"]) | set(OBSERVED)
    for subject in stage2_split["development_rich_subjects_assigned_to_training"]:
        used.update(case for case in manifest["partitions"]["fit"] if case.startswith(subject + "-"))
    if set(CASES) & used or set(CASES) & set(splits["test"]) or len(CASES) != len(set(CASES)):
        raise RuntimeError("the Stage 4 list overlaps prior subjects or the historical test split")
    gate_hash = file_sha256(GATE_PATH)
    saved = torch.load(GATE_PATH, map_location="cpu", weights_only=False)
    if saved["protocol"] != REVISED_PROTOCOL:
        raise RuntimeError("the Stage 2 checkpoint protocol does not match the frozen record")
    OUTPUT.mkdir(exist_ok=True)
    write_progress(OUTPUT / "protocol.json", {
        "cases": list(CASES),
        "rule": "first fit-manifest scan of the next 20 subjects after every subject used in Stage 2 or Stage 3",
        "one_scan_per_subject": True,
        "gate_sha256": gate_hash,
        "gate_path": str(GATE_PATH),
        "stage2_protocol": REVISED_PROTOCOL,
        "highres_extra_windows": 0,
        "manifest_hash": manifest["manifest_hash"],
    })
    gate = ResidualAdaptiveGate(saved["in_features"], saved["protocol"]["hidden"], 4)
    gate.load_state_dict(saved["state_dict"])
    gate.eval()
    for parameter in gate.parameters():
        parameter.requires_grad_(False)
    device = device_for(cfg["device"])
    if device.type != "mps":
        raise RuntimeError("Stage 4 requires MPS")
    expert_hashes, controller_hash = checkpoint_hashes(cfg)
    selected = records_for(records, CASES, splits["test"])
    models, loaded_experts = load_experts(cfg, splits)
    controller, loaded_controller = load_controller(cfg, splits, loaded_experts)
    if loaded_experts != expert_hashes or loaded_controller != controller_hash:
        raise RuntimeError("checkpoint hashes changed while loading")

    def stop_handler(_signum, _frame):
        raise BenchmarkStop("Stage 4 safety or time limit triggered")

    signal.signal(signal.SIGALRM, stop_handler)
    signal.signal(signal.SIGUSR1, stop_handler)
    signal.alarm(6 * 60 * 60)
    completed = []
    status, error = "running", None
    started = time.perf_counter()
    drops = []
    try:
        pending_index = 0
        for case_id, record in zip(CASES, selected):
            path = OUTPUT / f"{case_id}.json"
            if path.is_file():
                completed.append(json.loads(path.read_text()))
                continue
            if pending_index and pending_index % 5 == 0:
                free = shutil.disk_usage(OUTPUT).free
                projected = free - max(drops) if drops else free
                if free < NEXT_GROUP_FREE_BYTES or projected < STOP_FREE_BYTES:
                    status = "stopped"
                    error = "insufficient disk headroom for the next group of five"
                    break
            summary = run_case(case_id, record, models, controller, gate, cfg, device)
            completed.append(summary)
            write_progress(path, summary)
            before_bytes = summary["resources"]["before"]["disk_free_bytes"]
            after_bytes = summary["resources"]["after"]["disk_free_bytes"]
            drops.append(max(0, before_bytes - after_bytes))
            pending_index += 1
            print(json.dumps({
                "completed_case": case_id,
                "index": len(completed),
                "mean_dice": {name: value["mean_dice"] for name, value in summary["metrics"].items()},
            }, indent=2), flush=True)
        else:
            status = "complete"
    except Exception as exc:
        status = "stopped" if isinstance(exc, BenchmarkStop) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, BenchmarkStop):
            raise
    finally:
        signal.alarm(0)
        ordered = []
        by_id = {item["patient_id"]: item for item in completed}
        for case_id in CASES:
            if case_id in by_id:
                ordered.append(by_id[case_id])
        write_progress(OUTPUT / "report.json", {
            "format": "beta2_stage4_frozen_eval_v1",
            "status": status,
            "error": error,
            "cases": ordered,
            "aggregate": aggregate(ordered) if ordered else {},
            "seconds": time.perf_counter() - started,
            "gate_unchanged": file_sha256(GATE_PATH) == gate_hash,
            "checkpoints_unchanged": checkpoint_hashes(cfg) == (expert_hashes, controller_hash),
            "scope": "Twenty unused fit subjects. The validation pool was used for Beta 1 checkpoint selection. Not an external test set.",
        })
        del models, controller, gate
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
