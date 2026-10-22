"""Train one residual adaptive gate and score three held-out fit subjects.

The historical test split is not read. Frozen expert and controller
checkpoints are not modified.
"""

import argparse
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
from beta2_stability_benchmark import (
    BenchmarkStop,
    CaseResourceMonitor,
    checkpoint_hashes,
    resource_snapshot,
    write_progress,
)
from brats_debate.config import device_for, load_config, seed_everything
from brats_debate.data.brats_dataset import (
    discover_patients,
    make_splits,
    subject_id,
    validate_training_audit,
)
from brats_debate.evaluation.metrics import BRATS_REGIONS
from brats_debate.inference.pipeline import load_controller, load_experts
from brats_debate.training.train_controller import assert_split_isolation, records_for
from brats_debate.uncertainty.adaptive_pilot import (
    PILOT_EVAL_SUBJECTS,
    PILOT_TRAIN_SUBJECTS,
    TRAINING_PROTOCOL,
    apply_residual_volume,
    pilot_subjects,
    score_methods,
    train_residual_gate,
    voxel_change_counts,
)
from brats_debate.uncertainty.extraction import (
    load_rich_artifact,
    predict_experts_compact,
    stream_controller_and_uncertainty,
)
from freeze_beta2_partitions import validate_frozen_manifest


ROOT = Path(__file__).resolve().parents[1]
RICH_CASES = ROOT / "outputs" / "beta2" / "oracle_gap_20261019" / "cases"
OUTPUT = ROOT / "outputs" / "beta2" / "adaptive_pilot_20261019"
STOP_FREE_BYTES = 18 * 1024**3
START_FREE_BYTES = 25 * 1024**3
INFERENCE_SECONDS = 90 * 60
REGIONS = {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3]}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "brats_controller.yaml")
    parser.add_argument("--stage-b-config", type=Path, default=ROOT / "configs" / "beta2" / "foundation.yaml")
    parser.add_argument("--partition-manifest", type=Path, default=ROOT / "configs" / "beta2" / "subject_partitions.json")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--train-only", action="store_true")
    return parser.parse_args()


def load_patient(record, cfg):
    from brats_debate.data.brats_dataset import load_patient as load

    return load(record, cfg)


def concatenate_training_rows(cases, case_to_subject):
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
    feature_names = None
    for case_id, subject in cases:
        artifact = load_rich_artifact(RICH_CASES / f"{case_id}.npz")
        metadata = artifact["metadata"]
        if metadata["subject_id"] != subject or metadata["patient_id"] != case_id:
            raise RuntimeError(f"{case_id} identity does not match its rich artifact")
        if metadata["ground_truth_used_for_inference"] or metadata["highres_full_volume_used"]:
            raise RuntimeError(f"{case_id} used ground truth or full-volume HighRes at inference")
        if metadata["class_order"] != [0, 1, 2, 3] or metadata["source_split"] != "validation":
            raise RuntimeError(f"{case_id} class mapping or source split is unexpected")
        if subject not in PILOT_TRAIN_SUBJECTS:
            raise RuntimeError("training rows include a held-out subject")
        arrays = artifact["arrays"]
        inclusion = np.asarray(arrays["inclusion_probability"], dtype=np.float64)
        weight = np.asarray(arrays["sample_weight"], dtype=np.float64)
        if (inclusion <= 0).any() or not np.allclose(weight, 1.0 / inclusion, rtol=0, atol=1e-4):
            raise RuntimeError(f"{case_id} sample weight is not the inverse inclusion probability")
        if feature_names is None:
            feature_names = list(arrays["feature_names"])
        elif list(arrays["feature_names"]) != feature_names:
            raise RuntimeError("rich feature order differs between training cases")
        for key in keys:
            collected[key].append(np.asarray(arrays[key]))
        subject_index.append(np.full(inclusion.shape[0], PILOT_TRAIN_SUBJECTS.index(subject), dtype=np.int64))
    rows = {key: np.concatenate(value, axis=0) for key, value in collected.items()}
    rows["feature_names"] = feature_names
    return rows, np.concatenate(subject_index)


def metric_block(metrics):
    return {
        region: {
            "dice": metrics[region]["dice"],
            "hd95_mm": metrics[region]["hd95_mm"],
        }
        for region in BRATS_REGIONS
    } | {"mean_dice": metrics["brats_mean_dice"]}


def _expert_rows(volume):
    array = np.asarray(volume)
    return np.moveaxis(array, 0, -1).reshape(-1, array.shape[0])


def weight_summary(adaptive, frozen, availability, entropy, disagreement, target, expert_hard):
    adaptive = np.asarray(adaptive, dtype=np.float64)
    frozen = np.asarray(frozen, dtype=np.float64)
    availability = np.asarray(availability) > 0
    entropy = np.asarray(entropy).reshape(-1)
    disagreement = np.asarray(disagreement, dtype=bool).reshape(-1)
    target = np.asarray(target).reshape(-1)
    expert_hard = np.asarray(expert_hard)
    correct = availability & (expert_hard == target[:, None])
    incorrect = availability & ~correct
    uncertain = (entropy >= TRAINING_PROTOCOL["entropy_threshold"]) & disagreement

    def means(mask):
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if not mask.any():
            return None
        return {
            "voxels": int(mask.sum()),
            "mean_abs_weight_change": float(np.abs(adaptive[mask] - frozen[mask]).mean()),
            "mean_adaptive_weight": adaptive[mask].mean(0).tolist(),
            "mean_frozen_weight": frozen[mask].mean(0).tolist(),
            "mean_adaptive_weight_on_correct_experts": float((adaptive[mask] * correct[mask]).sum() / mask.sum()),
            "mean_adaptive_weight_on_incorrect_experts": float((adaptive[mask] * incorrect[mask]).sum() / mask.sum()),
            "mean_frozen_weight_on_correct_experts": float((frozen[mask] * correct[mask]).sum() / mask.sum()),
            "mean_frozen_weight_on_incorrect_experts": float((frozen[mask] * incorrect[mask]).sum() / mask.sum()),
        }

    return {"all": means(np.ones(entropy.shape[0], dtype=bool)), "high_entropy_disagreement": means(uncertain)}


def run_case(case_id, record, models, controller, gate, cfg, device, output):
    if shutil.disk_usage(output).free < STOP_FREE_BYTES:
        raise BenchmarkStop("internal free space fell below 18 GiB")
    monitor = CaseResourceMonitor(output)
    before = resource_snapshot(output)
    monitor.start()
    started = time.perf_counter()
    patient = outputs = final = uncertainty = frozen_weights = adapted = None
    try:
        patient = load_patient(record, cfg)
        if patient["patient_id"] != case_id:
            raise RuntimeError("loaded patient does not match the held-out case")
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
            gate, image, outputs, frozen_weights, uncertainty["js_disagreement"], uncertainty["predictive_entropy"]
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
        predictions = {
            "cnn": adapted["cnn"].numpy(),
            "majority": adapted["majority"].numpy(),
            "frozen_controller": frozen_label,
            "adaptive_controller": adaptive_label,
        }
        metrics = score_methods(predictions, target, patient["spacing"], REGIONS)
        entropy = uncertainty["predictive_entropy"][0, 0].cpu().numpy()
        changes = voxel_change_counts(
            frozen_label,
            adaptive_label,
            target,
            entropy,
            adapted["expert_disagreement"].numpy(),
        )
        hard = _expert_rows(np.stack(
            [outputs[name]["probabilities"].argmax(1)[0].cpu().numpy() for name in ("cnn", "transformer", "boundary", "highres")]
        ))
        availability_rows = _expert_rows(np.stack(
            [(outputs[name]["extra"]["availability"] > 0)[0, 0].cpu().numpy() for name in ("cnn", "transformer", "boundary", "highres")]
        ))
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
    summary = {
        "patient_id": case_id,
        "spacing_mm": list(spacing),
        "seconds": time.perf_counter() - started,
        "frozen_pipeline_seconds": frozen_seconds,
        "adaptive_gate_seconds": gate_seconds,
        "highres_setting": "proposal_limited_fusion",
        "metrics": {name: metric_block(value) for name, value in metrics.items()},
        "voxel_changes": changes,
        "weights": weights,
        "resources": {**monitor.summary(), "before": before, "after": resource_snapshot(output)},
    }
    return summary


def main():
    args = parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT.resolve()):
        raise ValueError("pilot output must remain on internal storage")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("pilot output directory is not empty")
    if shutil.disk_usage(ROOT).free < START_FREE_BYTES:
        raise BenchmarkStop("internal free space is below the 25 GiB start threshold")
    stability.MINIMUM_FREE_BYTES = STOP_FREE_BYTES
    cfg = load_config(args.config)
    stage_b = yaml.safe_load(args.stage_b_config.read_text())["stage_b"]
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    assert_split_isolation(splits)
    validate_training_audit(cfg)
    manifest = json.loads(args.partition_manifest.read_text())
    validate_frozen_manifest(manifest, cfg=cfg, records=records, splits=splits, stage_b_cfg=stage_b)
    case_to_subject = {record["patient_id"]: subject_id(record, cfg) for record in records}
    rich_cases = tuple(sorted(path.stem for path in RICH_CASES.glob("*.npz")))
    split = pilot_subjects(manifest["partitions"]["fit"], rich_cases, case_to_subject)
    train_cases = tuple(
        (case_id, case_to_subject[case_id])
        for case_id in manifest["partitions"]["fit"]
        if case_id in rich_cases and case_to_subject[case_id] in PILOT_TRAIN_SUBJECTS
    )
    eval_cases = tuple(
        case_id
        for case_id in manifest["partitions"]["fit"]
        if case_to_subject[case_id] in PILOT_EVAL_SUBJECTS
    )
    if set(eval_cases) & set(splits["test"]) or set(case_id for case_id, _ in train_cases) & set(splits["test"]):
        raise RuntimeError("historical test identities entered the pilot")
    for role in ("calibration", "evaluation"):
        overlap = set(eval_cases) & set(manifest["partitions"][role])
        if overlap:
            raise RuntimeError(f"held-out pilot cases overlap {role}")
    expert_hashes, controller_hash = checkpoint_hashes(cfg)
    args.output.mkdir()
    write_progress(
        args.output / "split.json",
        {
            "rule": "first six distinct rich fit subjects in immutable manifest order; last three held out",
            "train_subjects": list(split["train_subjects"]),
            "eval_subjects": list(split["eval_subjects"]),
            "train_cases": [case_id for case_id, _ in train_cases],
            "eval_cases": list(eval_cases),
            "manifest_hash": manifest["manifest_hash"],
            "not_selected_by_dice_error_size_or_headroom": True,
        },
    )
    rows, subject_index = concatenate_training_rows(train_cases, case_to_subject)
    seed_everything(cfg["seed"])
    started = time.perf_counter()
    gate, training = train_residual_gate(rows, subject_index)
    training["seconds"] = time.perf_counter() - started
    training["train_cases"] = [case_id for case_id, _ in train_cases]
    training["eval_cases_not_used"] = list(eval_cases)
    torch.save({"state_dict": gate.state_dict(), "protocol": TRAINING_PROTOCOL}, args.output / "gate.pt")
    write_progress(args.output / "training.json", training)
    print(json.dumps({"training_seconds": training["seconds"], "frozen_loss": training["frozen_loss"], "final_loss": training["final_loss"]}, indent=2), flush=True)
    if args.train_only:
        return
    if shutil.disk_usage(ROOT).free < START_FREE_BYTES:
        raise BenchmarkStop("internal free space fell below 25 GiB before held-out inference")
    device = device_for(cfg["device"])
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("held-out inference requires MPS")
    selected_records = records_for(records, eval_cases, splits["test"])
    models, loaded_expert_hashes = load_experts(cfg, splits)
    controller, loaded_controller_hash = load_controller(cfg, splits, loaded_expert_hashes)
    if loaded_expert_hashes != expert_hashes or loaded_controller_hash != controller_hash:
        raise RuntimeError("checkpoint hashes changed while loading")

    def stop_handler(_signum, _frame):
        raise BenchmarkStop("pilot inference safety or time limit triggered")

    signal.signal(signal.SIGALRM, stop_handler)
    signal.signal(signal.SIGUSR1, stop_handler)
    signal.alarm(INFERENCE_SECONDS)
    completed = []
    status, error = "running", None
    inference_started = time.perf_counter()
    try:
        for case_id, record in zip(eval_cases, selected_records):
            (args.output / "cases").mkdir(exist_ok=True)
            summary = run_case(case_id, record, models, controller, gate, cfg, device, args.output)
            summary["subject_id"] = case_to_subject[case_id]
            completed.append(summary)
            write_progress(args.output / "cases" / f"{case_id}.json", summary)
            print(json.dumps({"completed_case": case_id, "mean_dice": {name: value["mean_dice"] for name, value in summary["metrics"].items()}}, indent=2), flush=True)
        status = "complete"
    except Exception as exc:
        status = "stopped" if isinstance(exc, BenchmarkStop) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, BenchmarkStop):
            raise
    finally:
        signal.alarm(0)
        after_hashes = checkpoint_hashes(cfg)
        report = {
            "format": "beta2_adaptive_pilot_v1",
            "status": status,
            "error": error,
            "split": split,
            "eval_cases": list(eval_cases),
            "training": {key: value for key, value in training.items() if key != "loss_history"},
            "loss_history": training["loss_history"],
            "cases": completed,
            "inference_seconds": time.perf_counter() - inference_started,
            "checkpoint_hashes_unchanged": after_hashes == (expert_hashes, controller_hash),
            "expert_hashes": expert_hashes,
            "controller_hash": controller_hash,
            "final_resources": resource_snapshot(args.output),
            "scope": "Three held-out fit subjects. Not a test-set result and not a significance claim.",
        }
        write_progress(args.output / "report.json", report)
        del controller, models, gate
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    if status != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
