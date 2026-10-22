"""Train the revised Stage 2 gate, then score five fresh fit subjects.

The evaluation cases are not read until the checkpoint has been saved.
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
from beta2_adaptive_pilot import (
    REGIONS,
    _expert_rows,
    load_patient,
    metric_block,
    weight_summary,
)
from beta2_stability_benchmark import (
    BenchmarkStop,
    CaseResourceMonitor,
    checkpoint_hashes,
    resource_snapshot,
    write_progress,
)
from brats_debate.config import device_for, load_config, seed_everything
from brats_debate.controller.adaptive_gate import ResidualAdaptiveGate, feature_names
from brats_debate.data.brats_dataset import discover_patients, make_splits, subject_id, validate_training_audit
from brats_debate.inference.pipeline import load_controller, load_experts
from brats_debate.training.train_controller import assert_split_isolation, records_for
from brats_debate.uncertainty.adaptive_pilot import (
    apply_residual_volume,
    score_methods,
    voxel_change_counts,
)
from brats_debate.uncertainty.extraction import (
    load_rich_artifact,
    predict_experts_compact,
    stream_controller_and_uncertainty,
)
from brats_debate.uncertainty.stage2_gate import REVISED_PROTOCOL, train_revised_gate
from freeze_beta2_partitions import validate_frozen_manifest


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "outputs" / "beta2" / "stage2_adaptive_20261020"
OLD_RICH = ROOT / "outputs" / "beta2" / "oracle_gap_20261019" / "cases"
NEW_RICH = STAGE / "rich" / "cases"
STOP_FREE_BYTES = 20 * 1024**3
INFERENCE_SECONDS = 100 * 60


def artifact_path(case_id):
    for directory in (OLD_RICH, NEW_RICH):
        path = directory / f"{case_id}.npz"
        if path.is_file():
            return path
    raise FileNotFoundError(case_id)


def training_rows(split, case_to_subject):
    development_scans = {
        "BraTS-GLI-00006": ["BraTS-GLI-00006-000"],
        "BraTS-GLI-00025": ["BraTS-GLI-00025-000"],
        "BraTS-GLI-00049": ["BraTS-GLI-00049-000"],
        "BraTS-GLI-00056": ["BraTS-GLI-00056-000", "BraTS-GLI-00056-001"],
        "BraTS-GLI-00101": ["BraTS-GLI-00101-000"],
        "BraTS-GLI-00104": ["BraTS-GLI-00104-000"],
        "BraTS-GLI-00112": ["BraTS-GLI-00112-000"],
        "BraTS-GLI-00133": ["BraTS-GLI-00133-000"],
        "BraTS-GLI-00167": ["BraTS-GLI-00167-000"],
    }
    cases = []
    for subject in split["development_rich_subjects_assigned_to_training"]:
        cases.extend((scan, subject) for scan in development_scans[subject])
    for scan in split["added_training_cases"]:
        subject = case_to_subject[scan]
        cases.append((scan, subject))
    forbidden = set(split["evaluation_cases"])
    if any(scan in forbidden for scan, _ in cases):
        raise RuntimeError("an evaluation scan entered training")
    keys = (
        "expert_probabilities",
        "expert_entropy",
        "expert_available",
        "controller_weights",
        "X",
        "target_class",
        "sample_weight",
        "inclusion_probability",
    )
    collected = {key: [] for key in keys}
    subject_index = []
    subjects = list(split["development_rich_subjects_assigned_to_training"])
    subjects.extend(case_to_subject[scan] for scan in split["added_training_cases"])
    for scan, subject in cases:
        artifact = load_rich_artifact(artifact_path(scan))
        metadata = artifact["metadata"]
        if metadata["patient_id"] != scan or metadata["subject_id"] != subject:
            raise RuntimeError(f"{scan} identity does not match the frozen split")
        if metadata["ground_truth_used_for_inference"] or metadata["highres_full_volume_used"]:
            raise RuntimeError(f"{scan} used a forbidden inference input")
        if metadata["class_order"] != [0, 1, 2, 3]:
            raise RuntimeError(f"{scan} class mapping changed")
        arrays = artifact["arrays"]
        inclusion = np.asarray(arrays["inclusion_probability"], dtype=np.float64)
        weight = np.asarray(arrays["sample_weight"], dtype=np.float64)
        if (inclusion <= 0).any() or not np.allclose(weight, 1.0 / inclusion, rtol=0, atol=1e-4):
            raise RuntimeError(f"{scan} sample weight is not the inverse inclusion probability")
        if list(arrays["feature_names"]) != list(feature_names()[-8:]):
            raise RuntimeError(f"{scan} base features changed")
        for key in keys:
            collected[key].append(np.asarray(arrays[key]))
        subject_index.append(np.full(inclusion.shape[0], subjects.index(subject), dtype=np.int64))
    rows = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    rows["feature_names"] = list(feature_names()[-8:])
    return rows, np.concatenate(subject_index), [scan for scan, _ in cases], subjects


def run_case(case_id, record, models, controller, gate, cfg, device):
    if shutil.disk_usage(STAGE).free < STOP_FREE_BYTES:
        raise BenchmarkStop("internal free space fell below 20 GiB")
    monitor = CaseResourceMonitor(STAGE)
    before = resource_snapshot(STAGE)
    monitor.start()
    started = time.perf_counter()
    patient = outputs = final = uncertainty = frozen_weights = adapted = None
    try:
        patient = load_patient(record, cfg)
        if patient["patient_id"] != case_id:
            raise RuntimeError("loaded patient does not match the evaluation case")
        expert_started = time.perf_counter()
        outputs = predict_experts_compact(patient, models, cfg, device)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        image = torch.from_numpy(patient["image"])[None]
        final, uncertainty, frozen_weights = stream_controller_and_uncertainty(
            controller, image, outputs, cfg, device, slab_depth=8, return_weights=True
        )
        frozen_seconds = time.perf_counter() - expert_started
        gate_started = time.perf_counter()
        adapted = apply_residual_volume(
            gate,
            image,
            outputs,
            frozen_weights,
            uncertainty["js_disagreement"],
            uncertainty["predictive_entropy"],
            restrict_to_disagreement=True,
        )
        gate_seconds = time.perf_counter() - gate_started
        if float((adapted["weights"].sum(0) - 1).abs().max()) > 1e-5:
            raise RuntimeError("adaptive weights do not sum to one")
        unavailable = torch.stack(
            [
                (outputs[name]["extra"]["availability"] <= 0).detach().cpu().reshape(-1)
                for name in ("cnn", "transformer", "boundary", "highres")
            ]
        )
        leaked = adapted["weights"].reshape(4, -1)[unavailable]
        if leaked.numel() and float(leaked.abs().max()) != 0:
            raise RuntimeError("an unavailable expert received adaptive weight")
        target = np.asarray(patient["label"])
        spacing = tuple(float(value) for value in patient["spacing"])
        frozen_label = final.argmax(1)[0].to(torch.uint8).cpu().numpy()
        adaptive_label = adapted["probabilities"].argmax(0).to(torch.uint8).numpy()
        agreement = ~adapted["expert_disagreement"].numpy()
        if np.any((frozen_label != adaptive_label) & agreement):
            raise RuntimeError("the revised gate changed a voxel where the experts agree")
        predictions = {
            "cnn": adapted["cnn"].numpy(),
            "majority": adapted["majority"].numpy(),
            "frozen_controller": frozen_label,
            "adaptive_controller": adaptive_label,
        }
        metrics = score_methods(predictions, target, patient["spacing"], REGIONS)
        entropy = uncertainty["predictive_entropy"][0, 0].cpu().numpy()
        changes = voxel_change_counts(
            frozen_label, adaptive_label, target, entropy, adapted["expert_disagreement"].numpy()
        )
        hard = _expert_rows(np.stack([
            outputs[name]["probabilities"].argmax(1)[0].cpu().numpy()
            for name in ("cnn", "transformer", "boundary", "highres")
        ]))
        availability_rows = _expert_rows(np.stack([
            (outputs[name]["extra"]["availability"] > 0)[0, 0].cpu().numpy()
            for name in ("cnn", "transformer", "boundary", "highres")
        ]))
        weights = weight_summary(
            _expert_rows(adapted["weights"].numpy()),
            _expert_rows(frozen_weights[0].cpu().numpy()),
            availability_rows,
            entropy,
            adapted["expert_disagreement"].numpy(),
            target,
            hard,
        )
    finally:
        del patient, outputs, final, uncertainty, frozen_weights, adapted
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        monitor.stop()
    total = changes["all"]["voxels"]
    return {
        "patient_id": case_id,
        "spacing_mm": list(spacing),
        "seconds": time.perf_counter() - started,
        "frozen_pipeline_seconds": frozen_seconds,
        "adaptive_gate_seconds": gate_seconds,
        "highres_setting": "proposal_limited_fusion",
        "metrics": {name: metric_block(value) for name, value in metrics.items()},
        "voxel_changes": changes,
        "error_rate": {
            "frozen": changes["all"]["frozen_errors"] / total,
            "adaptive": changes["all"]["adaptive_errors"] / total,
        },
        "weights": weights,
        "resources": {**monitor.summary(), "before": before, "after": resource_snapshot(STAGE)},
    }


def main():
    if shutil.disk_usage(ROOT).free < 30 * 1024**3:
        raise BenchmarkStop("internal free space is below 30 GiB before revised-gate training")
    stability.MINIMUM_FREE_BYTES = STOP_FREE_BYTES
    split = json.loads((STAGE / "split.json").read_text())
    cfg = load_config(ROOT / "configs" / "brats_controller.yaml")
    stage_b = yaml.safe_load((ROOT / "configs" / "beta2" / "foundation.yaml").read_text())["stage_b"]
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    assert_split_isolation(splits)
    validate_training_audit(cfg)
    manifest = json.loads((ROOT / "configs" / "beta2" / "subject_partitions.json").read_text())
    validate_frozen_manifest(manifest, cfg=cfg, records=records, splits=splits, stage_b_cfg=stage_b)
    if manifest["manifest_hash"] != split["manifest_hash"]:
        raise RuntimeError("the partition manifest changed after the split was frozen")
    case_to_subject = {record["patient_id"]: subject_id(record, cfg) for record in records}
    if set(split["evaluation_cases"]) & set(splits["test"]):
        raise RuntimeError("a fresh evaluation case is in the historical test split")
    rows, subject_index, train_cases, subjects = training_rows(split, case_to_subject)
    if len(set(subjects)) != 20:
        raise RuntimeError("training does not contain 20 distinct subjects")
    seed_everything(cfg["seed"])
    started = time.perf_counter()
    gate, training = train_revised_gate(rows, subject_index)
    training["seconds"] = time.perf_counter() - started
    training["train_cases"] = train_cases
    training["train_subjects"] = subjects
    training["evaluation_cases_not_used"] = list(split["evaluation_cases"])
    torch.save(
        {"state_dict": gate.state_dict(), "protocol": REVISED_PROTOCOL, "in_features": len(feature_names())},
        STAGE / "revised_gate.pt",
    )
    write_progress(STAGE / "revised_training.json", training)
    print(json.dumps({
        "training_seconds": training["seconds"],
        "frozen_loss": training["frozen_loss"],
        "final_loss": training["final_loss"],
        "agreement_change": training["max_abs_weight_change_on_agreement"],
        "disagreement_change": training["mean_abs_weight_change_on_disagreement"],
    }, indent=2), flush=True)
    del rows
    gc.collect()
    if shutil.disk_usage(ROOT).free < 30 * 1024**3:
        raise BenchmarkStop("internal free space fell below 30 GiB before fresh evaluation")
    device = device_for(cfg["device"])
    if device.type != "mps":
        raise RuntimeError("fresh evaluation requires MPS")
    saved = torch.load(STAGE / "revised_gate.pt", map_location="cpu", weights_only=False)
    if saved["protocol"] != REVISED_PROTOCOL:
        raise RuntimeError("the saved protocol does not match the frozen revised protocol")
    gate = ResidualAdaptiveGate(saved["in_features"], REVISED_PROTOCOL["hidden"], 4)
    gate.load_state_dict(saved["state_dict"])
    gate.eval()
    expert_hashes, controller_hash = checkpoint_hashes(cfg)
    selected = records_for(records, split["evaluation_cases"], splits["test"])
    models, loaded_experts = load_experts(cfg, splits)
    controller, loaded_controller = load_controller(cfg, splits, loaded_experts)
    if loaded_experts != expert_hashes or loaded_controller != controller_hash:
        raise RuntimeError("checkpoint hashes changed while loading")

    def stop_handler(_signum, _frame):
        raise BenchmarkStop("revised evaluation safety or time limit triggered")

    signal.signal(signal.SIGALRM, stop_handler)
    signal.signal(signal.SIGUSR1, stop_handler)
    signal.alarm(INFERENCE_SECONDS)
    completed = []
    status, error = "running", None
    inference_started = time.perf_counter()
    try:
        for case_id, record in zip(split["evaluation_cases"], selected):
            (STAGE / "evaluation").mkdir(exist_ok=True)
            summary = run_case(case_id, record, models, controller, gate, cfg, device)
            summary["subject_id"] = case_to_subject[case_id]
            completed.append(summary)
            write_progress(STAGE / "evaluation" / f"{case_id}.json", summary)
            print(json.dumps({
                "completed_case": case_id,
                "mean_dice": {name: value["mean_dice"] for name, value in summary["metrics"].items()},
            }, indent=2), flush=True)
        status = "complete"
    except Exception as exc:
        status = "stopped" if isinstance(exc, BenchmarkStop) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, BenchmarkStop):
            raise
    finally:
        signal.alarm(0)
        write_progress(STAGE / "revised_evaluation_report.json", {
            "format": "beta2_stage2_revised_evaluation_v1",
            "status": status,
            "error": error,
            "evaluation_cases": list(split["evaluation_cases"]),
            "training_subjects": subjects,
            "cases": completed,
            "inference_seconds": time.perf_counter() - inference_started,
            "checkpoint_hashes_unchanged": checkpoint_hashes(cfg) == (expert_hashes, controller_hash),
            "scope": "Five fresh fit subjects. Not a test-set result and not a significance claim.",
        })
        del controller, models, gate
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    if status != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
