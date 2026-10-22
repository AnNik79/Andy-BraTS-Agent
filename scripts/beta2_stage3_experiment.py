"""Five-case Stage 3 comparison. The Stage 2 gate file is only read."""

import gc
import hashlib
import json
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import beta2_stability_benchmark as stability
from beta2_adaptive_pilot import _expert_rows, load_patient, metric_block, weight_summary
from beta2_stability_benchmark import (
    BenchmarkStop,
    CaseResourceMonitor,
    checkpoint_hashes,
    resource_snapshot,
    write_progress,
)
from brats_debate.config import SEGMENTATION_EXPERTS, device_for, load_config
from brats_debate.controller.adaptive_gate import ResidualAdaptiveGate, feature_names
from brats_debate.data.brats_dataset import discover_patients, make_splits, subject_id, validate_training_audit
from brats_debate.inference.pipeline import load_controller, load_experts
from brats_debate.perception.active import (
    ACTIVE_PROTOCOL,
    executed_highres_starts,
    matched_control_starts,
    merge_highres_compact,
    plan_additional_highres,
)
from brats_debate.training.train_controller import assert_split_isolation, records_for
from brats_debate.uncertainty.adaptive_pilot import (
    apply_residual_volume,
    score_methods,
    voxel_change_counts,
)
from brats_debate.uncertainty.extraction import predict_experts_compact, stream_controller_and_uncertainty
from brats_debate.uncertainty.stage2_gate import REVISED_PROTOCOL
from freeze_beta2_partitions import validate_frozen_manifest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "beta2" / "stage3_mri_20261020"
GATE_PATH = ROOT / "outputs" / "beta2" / "stage2_adaptive_20261020" / "revised_gate.pt"
SPLIT_PATH = ROOT / "outputs" / "beta2" / "stage2_adaptive_20261020" / "split.json"
CASES = (
    "BraTS-GLI-00414-000",
    "BraTS-GLI-00451-000",
    "BraTS-GLI-00493-000",
    "BraTS-GLI-00500-000",
    "BraTS-GLI-00501-000",
)
REGIONS = {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3]}
PATCH = (32, 32, 32)
STOP_FREE_BYTES = 18 * 1024**3


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clone_output(output):
    cloned = {}
    for key, value in output.items():
        if key == "extra":
            cloned[key] = {name: item.clone() if torch.is_tensor(item) else item for name, item in value.items()}
        elif torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def proposal_mask(outputs, threshold):
    names = ("cnn", "transformer", "boundary")
    votes = torch.stack([(outputs[name]["probabilities"].argmax(1) > 0) for name in names]).any(0)[:, None]
    uncertain = torch.stack([outputs[name]["uncertainty"] for name in names]).mean(0) > threshold
    return (votes | uncertain).float()


def predict_windows(model, image, starts, device):
    windows = []
    for start in starts:
        slices = tuple(slice(origin, origin + length) for origin, length in zip(start, PATCH))
        prediction = model.predict(image[(..., *slices)].contiguous().to(device), 1)
        windows.append({
            "probabilities": prediction["probabilities"].detach().cpu(),
            "features": prediction["features"].detach().cpu(),
        })
        del prediction
    return windows


def adaptive_label(gate, image, outputs, weights, uncertainty):
    adapted = apply_residual_volume(
        gate,
        image,
        outputs,
        weights,
        uncertainty["js_disagreement"],
        uncertainty["predictive_entropy"],
        restrict_to_disagreement=True,
    )
    if float((adapted["weights"].sum(0) - 1).abs().max()) > 1e-4:
        raise RuntimeError("adaptive weights do not sum to one")
    return adapted["probabilities"].argmax(0).to(torch.uint8).numpy(), adapted


def run_case(case_id, record, models, controller, gate, cfg, device):
    if shutil.disk_usage(OUTPUT).free < STOP_FREE_BYTES:
        raise BenchmarkStop("internal free space fell below 18 GiB")
    monitor = CaseResourceMonitor(OUTPUT)
    before = resource_snapshot(OUTPUT)
    monitor.start()
    started = time.perf_counter()
    patient = outputs = None
    try:
        patient = load_patient(record, cfg)
        if patient["patient_id"] != case_id:
            raise RuntimeError("loaded patient does not match the authorized case")
        image = torch.from_numpy(patient["image"])[None]
        outputs = predict_experts_compact(patient, models, cfg, device)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        final, uncertainty, weights = stream_controller_and_uncertainty(
            controller, image, outputs, cfg, device, slab_depth=8, return_weights=True
        )
        frozen_label = final.argmax(1)[0].to(torch.uint8).cpu().numpy()
        stage2_label, stage2_detail = adaptive_label(gate, image, outputs, weights, uncertainty)
        proposal = proposal_mask(outputs, cfg["inference"]["uncertainty_threshold"])
        shape = tuple(image.shape[2:])
        executed = executed_highres_starts(
            proposal, shape, patch=PATCH, overlap=0.5, max_patches=cfg["inference"]["highres_max_patches"]
        )
        expert_probabilities = torch.stack([outputs[name]["probabilities"][0] for name in SEGMENTATION_EXPERTS], 0)
        expert_availability = torch.stack(
            [outputs[name]["extra"]["availability"][0, 0] for name in SEGMENTATION_EXPERTS], 0
        )
        plan = plan_additional_highres(
            outputs["highres"]["extra"]["availability"],
            expert_probabilities,
            expert_availability,
            uncertainty["predictive_entropy"],
            patch=PATCH,
            overlap=0.5,
            entropy_threshold=ACTIVE_PROTOCOL["entropy_threshold"],
            min_eligible_fraction=ACTIVE_PROTOCOL["min_eligible_fraction"],
            max_actions=ACTIVE_PROTOCOL["max_actions"],
            excluded_starts=executed,
        )
        targeted = list(plan["starts"])
        control = matched_control_starts(
            proposal,
            shape,
            patch=PATCH,
            overlap=0.5,
            already_run=cfg["inference"]["highres_max_patches"],
            count=len(targeted),
        )
        if set(targeted) & set(executed) or set(control) & set(executed):
            raise RuntimeError("an already computed HighRes window was selected again")

        def branch(starts):
            if not starts:
                return stage2_label, 0.0, 0, None
            model = models["highres"]
            model.to(device)
            highres_started = time.perf_counter()
            try:
                windows = predict_windows(model, image, starts, device)
            finally:
                model.cpu()
            highres_seconds = time.perf_counter() - highres_started
            original = clone_output(outputs["highres"])
            before_probabilities = original["probabilities"]
            before_availability = original["extra"]["availability"]
            branch_outputs = dict(outputs)
            branch_outputs["highres"] = merge_highres_compact(original, starts, windows)
            preserved = before_availability > 0
            if not torch.allclose(
                branch_outputs["highres"]["probabilities"][preserved.expand_as(before_probabilities)],
                before_probabilities[preserved.expand_as(before_probabilities)],
                atol=1e-5,
            ):
                raise RuntimeError("an existing HighRes prediction changed")
            refreshed, refreshed_uncertainty, refreshed_weights = stream_controller_and_uncertainty(
                controller, image, branch_outputs, cfg, device, slab_depth=8, return_weights=True
            )
            label, detail = adaptive_label(gate, image, branch_outputs, refreshed_weights, refreshed_uncertainty)
            new_voxels = (before_availability <= 0) & (branch_outputs["highres"]["extra"]["availability"] > 0)
            anchor = refreshed_weights[0, 3][new_voxels[0, 0]]
            adapted_weight = detail["weights"][3][new_voxels[0, 0]]
            del refreshed, branch_outputs, windows
            return label, highres_seconds, int(new_voxels.sum()), {
                "mean_recomputed_frozen_highres_weight": float(anchor.mean()) if anchor.numel() else None,
                "mean_adaptive_highres_weight": float(adapted_weight.mean()) if adapted_weight.numel() else None,
            }

        targeted_label, targeted_seconds, targeted_voxels, targeted_weights = branch(targeted)
        control_label, control_seconds, control_voxels, control_weights = branch(control)
        target = np.asarray(patient["label"])
        predictions = {
            "frozen_beta1": frozen_label,
            "stage2_gate": stage2_label,
            "targeted_highres": targeted_label,
            "proposal_control": control_label,
            "cnn": stage2_detail["cnn"].numpy(),
            "majority": stage2_detail["majority"].numpy(),
        }
        metrics = score_methods(predictions, target, patient["spacing"], REGIONS)
        entropy = uncertainty["predictive_entropy"][0, 0].numpy()
        disagreement = stage2_detail["expert_disagreement"].numpy()
        changes = {
            "targeted_versus_stage2": voxel_change_counts(stage2_label, targeted_label, target, entropy, disagreement),
            "control_versus_stage2": voxel_change_counts(stage2_label, control_label, target, entropy, disagreement),
        }
        hard = _expert_rows(np.stack([
            outputs[name]["probabilities"].argmax(1)[0].cpu().numpy() for name in SEGMENTATION_EXPERTS
        ]))
        availability_rows = _expert_rows(np.stack([
            (outputs[name]["extra"]["availability"] > 0)[0, 0].cpu().numpy() for name in SEGMENTATION_EXPERTS
        ]))
        weights_report = weight_summary(
            _expert_rows(stage2_detail["weights"].numpy()),
            _expert_rows(weights[0].cpu().numpy()),
            availability_rows,
            entropy,
            disagreement,
            target,
            hard,
        )
        summary = {
            "patient_id": case_id,
            "spacing_mm": [float(value) for value in patient["spacing"]],
            "seconds": time.perf_counter() - started,
            "actions": {
                "targeted_windows": [list(start) for start in targeted],
                "control_windows": [list(start) for start in control],
                "shared_windows": [list(start) for start in set(targeted) & set(control)],
                "targeted_new_voxels": targeted_voxels,
                "control_new_voxels": control_voxels,
                "targeted_highres_seconds": targeted_seconds,
                "control_highres_seconds": control_seconds,
                "no_action": len(targeted) == 0,
                "targeted_highres_weight": targeted_weights,
                "control_highres_weight": control_weights,
            },
            "metrics": {name: metric_block(value) for name, value in metrics.items()},
            "voxel_changes": changes,
            "stage2_weights_on_original_evidence": weights_report,
            "resources": {**monitor.summary(), "before": before, "after": resource_snapshot(OUTPUT)},
        }
    finally:
        del patient, outputs
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        monitor.stop()
    return summary


def main():
    if any(OUTPUT.glob("BraTS-*.json")):
        raise RuntimeError("Stage 3 MRI case results already exist")
    if shutil.disk_usage(ROOT).free < 25 * 1024**3:
        raise BenchmarkStop("internal free space is below 25 GiB")
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
    used = set(stage2_split["added_training_cases"]) | set(stage2_split["evaluation_cases"])
    for subject in stage2_split["development_rich_subjects_assigned_to_training"]:
        used.update(case for case in manifest["partitions"]["fit"] if case.startswith(subject + "-"))
    if set(CASES) & used or set(CASES) & set(splits["test"]):
        raise RuntimeError("an authorized case was already used or is in the historical test split")
    gate_hash = file_sha256(GATE_PATH)
    saved = torch.load(GATE_PATH, map_location="cpu", weights_only=False)
    if saved["protocol"] != REVISED_PROTOCOL:
        raise RuntimeError("the Stage 2 checkpoint protocol does not match the frozen record")
    gate = ResidualAdaptiveGate(saved["in_features"], saved["protocol"]["hidden"], 4)
    gate.load_state_dict(saved["state_dict"])
    gate.eval()
    for parameter in gate.parameters():
        parameter.requires_grad_(False)
    device = device_for(cfg["device"])
    if device.type != "mps":
        raise RuntimeError("the authorized experiment requires MPS")
    expert_hashes, controller_hash = checkpoint_hashes(cfg)
    selected = records_for(records, CASES, splits["test"])
    models, loaded_experts = load_experts(cfg, splits)
    controller, loaded_controller = load_controller(cfg, splits, loaded_experts)
    if loaded_experts != expert_hashes or loaded_controller != controller_hash:
        raise RuntimeError("checkpoint hashes changed while loading")
    OUTPUT.mkdir(exist_ok=True)
    write_progress(OUTPUT / "protocol.json", {
        "cases": list(CASES),
        "gate_sha256": gate_hash,
        "gate_path": str(GATE_PATH),
        "active_protocol": ACTIVE_PROTOCOL,
        "stage2_protocol": REVISED_PROTOCOL,
        "expert_hashes": expert_hashes,
        "controller_hash": controller_hash,
    })

    def stop_handler(_signum, _frame):
        raise BenchmarkStop("Stage 3 safety or time limit triggered")

    signal.signal(signal.SIGALRM, stop_handler)
    signal.signal(signal.SIGUSR1, stop_handler)
    signal.alarm(120 * 60)
    completed = []
    status, error = "running", None
    started = time.perf_counter()
    try:
        for case_id, record in zip(CASES, selected):
            summary = run_case(case_id, record, models, controller, gate, cfg, device)
            summary["subject_id"] = subject_id(record, cfg)
            completed.append(summary)
            write_progress(OUTPUT / f"{case_id}.json", summary)
            print(json.dumps({
                "completed_case": case_id,
                "actions": len(summary["actions"]["targeted_windows"]),
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
        write_progress(OUTPUT / "report.json", {
            "format": "beta2_stage3_mri_v1",
            "status": status,
            "error": error,
            "cases": completed,
            "seconds": time.perf_counter() - started,
            "gate_unchanged": file_sha256(GATE_PATH) == gate_hash,
            "checkpoints_unchanged": checkpoint_hashes(cfg) == (expert_hashes, controller_hash),
            "scope": "Five fresh fit subjects. Not a test-set result.",
        })
        del models, controller, gate
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    if status != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
