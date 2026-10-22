"""Stage 3 proposal-extension comparison. The Stage 2 gate is only read."""

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
from beta2_adaptive_pilot import load_patient
from beta2_stage3_experiment import (
    GATE_PATH,
    PATCH,
    REGIONS,
    SPLIT_PATH,
    adaptive_label,
    clone_output,
    file_sha256,
    proposal_mask,
    predict_windows,
)
from beta2_stability_benchmark import (
    BenchmarkStop,
    CaseResourceMonitor,
    checkpoint_hashes,
    resource_snapshot,
    write_progress,
)
from brats_debate.config import device_for, load_config
from brats_debate.controller.adaptive_gate import ResidualAdaptiveGate
from brats_debate.data.brats_dataset import discover_patients, make_splits, subject_id, validate_training_audit
from brats_debate.inference.pipeline import load_controller, load_experts
from brats_debate.perception.active import far_proposal_starts, merge_highres_compact, plan_proposal_extension
from brats_debate.training.train_controller import assert_split_isolation, records_for
from brats_debate.uncertainty.adaptive_pilot import score_methods, voxel_change_counts
from brats_debate.uncertainty.extraction import predict_experts_compact, stream_controller_and_uncertainty
from brats_debate.uncertainty.stage2_gate import REVISED_PROTOCOL
from freeze_beta2_partitions import validate_frozen_manifest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "beta2" / "stage3_proposal_extension_20261020"
OBSERVED = (
    "BraTS-GLI-00414-000",
    "BraTS-GLI-00451-000",
    "BraTS-GLI-00493-000",
    "BraTS-GLI-00500-000",
    "BraTS-GLI-00501-000",
)
CASES = (
    "BraTS-GLI-00514-000",
    "BraTS-GLI-00519-000",
    "BraTS-GLI-00523-000",
    "BraTS-GLI-00529-000",
    "BraTS-GLI-00552-000",
)
STOP_FREE_BYTES = 18 * 1024**3


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
        extension = plan_proposal_extension(
            proposal,
            shape,
            patch=PATCH,
            overlap=0.5,
            already_run=cfg["inference"]["highres_max_patches"],
            max_actions=16,
        )
        proposal_starts = list(extension["starts"])
        weak_starts = far_proposal_starts(
            proposal,
            shape,
            patch=PATCH,
            overlap=0.5,
            already_run=cfg["inference"]["highres_max_patches"],
            count=len(proposal_starts),
        )
        if len(weak_starts) != len(proposal_starts):
            raise RuntimeError("the weak-ranked control does not match the proposal-extension budget")
        if set(proposal_starts) & set(weak_starts):
            raise RuntimeError("proposal-ranked and weak-ranked windows overlap")

        def branch(starts):
            if not starts:
                return stage2_label, 0.0, 0, 0, None, 0
            model = models["highres"]
            model.to(device)
            highres_started = time.perf_counter()
            try:
                windows = predict_windows(model, image, starts, device)
            finally:
                model.cpu()
            highres_seconds = time.perf_counter() - highres_started
            original = clone_output(outputs["highres"])
            before_availability = original["extra"]["availability"]
            before_probabilities = original["probabilities"]
            branch_outputs = dict(outputs)
            branch_outputs["highres"] = merge_highres_compact(original, starts, windows)
            preserved = before_availability > 0
            if not torch.allclose(
                branch_outputs["highres"]["probabilities"][preserved.expand_as(before_probabilities)],
                before_probabilities[preserved.expand_as(before_probabilities)],
                atol=1e-5,
            ):
                raise RuntimeError("an existing HighRes prediction changed")
            _refreshed, refreshed_uncertainty, refreshed_weights = stream_controller_and_uncertainty(
                controller, image, branch_outputs, cfg, device, slab_depth=8, return_weights=True
            )
            label, detail = adaptive_label(gate, image, branch_outputs, refreshed_weights, refreshed_uncertainty)
            new_voxels = (before_availability <= 0) & (branch_outputs["highres"]["extra"]["availability"] > 0)
            anchor = refreshed_weights[0, 3][new_voxels[0, 0]]
            adapted_weight = detail["weights"][3][new_voxels[0, 0]]
            changed = int((label != stage2_label).sum())
            del branch_outputs, windows
            return label, highres_seconds, int(new_voxels.sum()), int(preserved.sum()), {
                "mean_recomputed_frozen_highres_weight": float(anchor.mean()) if anchor.numel() else None,
                "mean_adaptive_highres_weight": float(adapted_weight.mean()) if adapted_weight.numel() else None,
            }, changed

        proposal_label, proposal_seconds, proposal_voxels, baseline_covered, proposal_weights, proposal_changed = branch(proposal_starts)
        weak_label, weak_seconds, weak_voxels, _baseline_again, weak_weights, weak_changed = branch(weak_starts)
        target = np.asarray(patient["label"])
        predictions = {
            "frozen_beta1": frozen_label,
            "stage2_gate": stage2_label,
            "proposal_extension": proposal_label,
            "weak_ranked_control": weak_label,
        }
        metrics = score_methods(predictions, target, patient["spacing"], REGIONS)
        entropy = uncertainty["predictive_entropy"][0, 0].numpy()
        disagreement = stage2_detail["expert_disagreement"].numpy()
        summary = {
            "patient_id": case_id,
            "spacing_mm": [float(value) for value in patient["spacing"]],
            "seconds": time.perf_counter() - started,
            "actions": {
                "proposal_windows": len(proposal_starts),
                "weak_windows": len(weak_starts),
                "shared_windows": len(set(proposal_starts) & set(weak_starts)),
                "baseline_highres_voxels": baseline_covered,
                "proposal_new_voxels": proposal_voxels,
                "weak_new_voxels": weak_voxels,
                "proposal_coverage_increase_percent": (
                    100.0 * proposal_voxels / baseline_covered if baseline_covered else None
                ),
                "weak_coverage_increase_percent": (
                    100.0 * weak_voxels / baseline_covered if baseline_covered else None
                ),
                "proposal_changed_voxels": proposal_changed,
                "weak_changed_voxels": weak_changed,
                "proposal_highres_seconds": proposal_seconds,
                "weak_highres_seconds": weak_seconds,
                "proposal_highres_weight": proposal_weights,
                "weak_highres_weight": weak_weights,
            },
            "metrics": {name: metric_block(value) for name, value in metrics.items()},
            "voxel_changes": {
                "proposal_versus_stage2": voxel_change_counts(stage2_label, proposal_label, target, entropy, disagreement),
                "weak_versus_stage2": voxel_change_counts(stage2_label, weak_label, target, entropy, disagreement),
            },
            "resources": {**monitor.summary(), "before": before, "after": resource_snapshot(OUTPUT)},
        }
    finally:
        del patient, outputs
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        monitor.stop()
    return summary


def metric_block(metrics):
    from beta2_adaptive_pilot import metric_block as block

    return block(metrics)


def main():
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
    used = set(stage2_split["added_training_cases"]) | set(stage2_split["evaluation_cases"]) | set(OBSERVED)
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

    def stop_handler(_signum, _frame):
        raise BenchmarkStop("proposal-extension safety or time limit triggered")

    signal.signal(signal.SIGALRM, stop_handler)
    signal.signal(signal.SIGUSR1, stop_handler)
    signal.alarm(120 * 60)
    completed = []
    for case_id in CASES:
        path = OUTPUT / f"{case_id}.json"
        if path.is_file():
            completed.append(json.loads(path.read_text()))
    status, error = "running", None
    started = time.perf_counter()
    try:
        for case_id, record in zip(CASES, selected):
            if any(item["patient_id"] == case_id for item in completed):
                continue
            summary = run_case(case_id, record, models, controller, gate, cfg, device)
            summary["subject_id"] = subject_id(record, cfg)
            completed.append(summary)
            write_progress(OUTPUT / f"{case_id}.json", summary)
            print(json.dumps({
                "completed_case": case_id,
                "proposal_new_voxels": summary["actions"]["proposal_new_voxels"],
                "weak_new_voxels": summary["actions"]["weak_new_voxels"],
                "mean_dice": {name: value["mean_dice"] for name, value in summary["metrics"].items()},
            }, indent=2), flush=True)
        if len(completed) == len(CASES):
            status = "complete"
        else:
            status = "incomplete"
    except Exception as exc:
        status = "stopped" if isinstance(exc, BenchmarkStop) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, BenchmarkStop):
            raise
    finally:
        signal.alarm(0)
        write_progress(OUTPUT / "report.json", {
            "format": "beta2_stage3_proposal_extension_v1",
            "status": status,
            "error": error,
            "cases": completed,
            "seconds": time.perf_counter() - started,
            "gate_unchanged": file_sha256(GATE_PATH) == gate_hash,
            "checkpoints_unchanged": checkpoint_hashes(cfg) == (expert_hashes, controller_hash),
            "scope": "Proposal-extension feasibility on fit subjects. Not a test-set result and not a learned action policy.",
        })
        del models, controller, gate
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    if status == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
