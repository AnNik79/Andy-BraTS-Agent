"""Rerun ten existing fit cases and measure frozen-controller oracle headroom.

The historical test split is not read. No controller is trained.
"""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import signal
import time

import torch
import yaml

from beta2_fit_collection import discover_collected_cases
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
from brats_debate.inference.pipeline import load_controller, load_experts
from brats_debate.training.train_controller import assert_split_isolation, records_for
from brats_debate.uncertainty.extraction import (
    FEATURE_EQUIVALENCE_ATOL,
    case_provenance,
    compare_reused_sample,
    extract_reused_sample,
    load_case_artifact,
    predict_experts_compact,
    save_rich_artifact,
    stream_controller_and_uncertainty,
)
from brats_debate.uncertainty.oracle_gap import case_oracle_summary, pooled_case_mean
from freeze_beta2_partitions import validate_frozen_manifest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = "oracle_gap_20261019"
GROUP_SIZE = 5
START_FREE_BYTES = 23 * 1024**3
NEXT_GROUP_FREE_BYTES = 22 * 1024**3
STOP_FREE_BYTES = 18 * 1024**3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "brats_controller.yaml")
    parser.add_argument(
        "--stage-b-config", type=Path, default=ROOT / "configs" / "beta2" / "foundation.yaml"
    )
    parser.add_argument(
        "--partition-manifest",
        type=Path,
        default=ROOT / "configs" / "beta2" / "subject_partitions.json",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "beta2" / EXPERIMENT)
    parser.add_argument("--max-seconds", type=int, default=10800)
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--case-id", action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--min-start-gib", type=float, default=23)
    parser.add_argument("--stop-free-gib", type=float, default=18)
    return parser.parse_args()


def select_existing_fit_cases(manifest, collected, count=10):
    selected = [
        case_id
        for case_id in manifest["partitions"]["fit"]
        if case_id in collected
    ][:count]
    if len(selected) != count:
        raise RuntimeError("not enough existing fit artifacts")
    return tuple(selected)


def original_artifacts(case_id):
    matches = sorted(
        path
        for path in (ROOT / "outputs" / "beta2").rglob(f"{case_id}.npz")
        if EXPERIMENT not in path.parts
    )
    loaded = []
    for path in matches:
        try:
            artifact = load_case_artifact(path)
        except (OSError, ValueError, KeyError):
            continue
        if artifact["metadata"].get("patient_id") == case_id:
            loaded.append((path, artifact))
    if not loaded:
        raise RuntimeError(f"no valid Stage B artifact for {case_id}")
    reference = loaded[0][1]["arrays"]["flat_index"]
    for path, artifact in loaded[1:]:
        if not np_equal(artifact["arrays"]["flat_index"], reference):
            raise RuntimeError(f"duplicate artifacts for {case_id} sample different voxels")
    return loaded[0]


def np_equal(left, right):
    import numpy as np

    return np.array_equal(np.asarray(left), np.asarray(right))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_case(case_id, record, case_subject, models, controller, cfg, device, provenance, output, original_path, original):
    import numpy as np

    artifact_path = output / "cases" / f"{case_id}.npz"
    summary_path = output / "cases" / f"{case_id}.summary.json"
    if artifact_path.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite oracle output for {case_id}")
    if shutil.disk_usage(output).free < STOP_FREE_BYTES:
        raise BenchmarkStop("internal free space fell below 18 GiB")
    monitor = CaseResourceMonitor(output)
    before = resource_snapshot(output)
    monitor.start()
    started = time.perf_counter()
    patient = outputs = final = uncertainty_maps = weights = None
    try:
        patient = load_patient(record, cfg)
        outputs = predict_experts_compact(patient, models, cfg, device)
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        image = torch.from_numpy(patient["image"])[None]
        final, uncertainty_maps, weights = stream_controller_and_uncertainty(
            controller, image, outputs, cfg, device, slab_depth=8, return_weights=True
        )
        rich = extract_reused_sample(
            patient,
            outputs,
            final,
            weights,
            uncertainty_maps,
            cfg,
            subject_id=case_subject,
            original=original["arrays"],
        )
        equivalence = compare_reused_sample(
            original["arrays"],
            rich["arrays"],
            original["metadata"]["feature_names"],
            feature_atol=FEATURE_EQUIVALENCE_ATOL,
        )
        oracle = case_oracle_summary(rich["arrays"], rich["metadata"]["feature_names"])
        rich["metadata"]["original_artifact"] = str(original_path)
        rich["metadata"]["feature_equivalence"] = equivalence
        save_rich_artifact(artifact_path, rich, provenance=provenance)
    finally:
        del patient, outputs, final, uncertainty_maps, weights
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        monitor.stop()
    after = resource_snapshot(output)
    summary = {
        "format": "beta2_controller_oracle_case_summary_v1",
        "complete": True,
        "patient_id": case_id,
        "subject_id": case_subject,
        "source_split": "validation",
        "partition_role": "fit",
        "artifact": str(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "sample_count": int(np.load(artifact_path)["flat_index"].size),
        "original_artifact": str(original_path),
        "equivalence": equivalence,
        "oracle": oracle,
        "seconds": time.perf_counter() - started,
        "resources": {
            **monitor.summary(),
            "before": before,
            "after": after,
            "swap_growth_mib": (
                None
                if before["swap_used_mib"] is None or after["swap_used_mib"] is None
                else after["swap_used_mib"] - before["swap_used_mib"]
            ),
        },
        "provenance": provenance,
    }
    write_progress(summary_path, summary)
    return summary


def load_patient(record, cfg):
    from brats_debate.data.brats_dataset import load_patient as load

    return load(record, cfg)


def aggregate(cases):
    keys = (
        ("selection_headroom_when_controller_wrong", "sampled_fraction"),
        ("controller_corrects_any_available_expert", "inclusion_weighted_fraction"),
        ("controller_overrides_available_correct_expert", "inclusion_weighted_fraction"),
        ("mean_weight_on_correct_experts", "inclusion_weighted_mean"),
        ("mean_weight_on_incorrect_available_experts", "inclusion_weighted_mean"),
        ("hard_selection_cannot_repair", "sampled_fraction"),
        ("mixture_correct_when_every_hard_expert_wrong", "sampled_fraction"),
    )
    return {
        ".".join(key): pooled_case_mean([case["oracle"] for case in cases], key)
        for key in keys
    }


def main():
    args = parse_args()
    if args.case_id is None and not 1 <= args.cases <= 10:
        raise ValueError("this authorization allows 1 to 10 cases")
    if args.case_id is not None and not 1 <= len(args.case_id) <= 10:
        raise ValueError("this authorization allows 1 to 10 explicit cases")
    if args.resume and not args.case_id:
        raise ValueError("resume requires explicit case ids")
    if args.max_seconds < 1 or args.max_seconds > 10800:
        raise ValueError("--max-seconds must be in [1,10800]")
    if args.stop_free_gib < 15:
        raise ValueError("the stop threshold must stay at least 15 GiB")
    if args.min_start_gib < args.stop_free_gib:
        raise ValueError("the start threshold must stay at or above the stop threshold")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT.resolve()):
        raise ValueError("oracle output must remain on internal storage")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise RuntimeError("oracle output directory is not empty")
    if shutil.disk_usage(ROOT).free < int(args.min_start_gib * 1024**3):
        raise BenchmarkStop(
            f"internal free space is below the {args.min_start_gib:g} GiB start threshold"
        )
    global STOP_FREE_BYTES
    STOP_FREE_BYTES = int(args.stop_free_gib * 1024**3)
    stability.MINIMUM_FREE_BYTES = STOP_FREE_BYTES

    cfg = load_config(args.config)
    stage_b = yaml.safe_load(args.stage_b_config.read_text())["stage_b"]
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    assert_split_isolation(splits)
    validate_training_audit(cfg)
    manifest = json.loads(args.partition_manifest.read_text())
    validate_frozen_manifest(manifest, cfg=cfg, records=records, splits=splits, stage_b_cfg=stage_b)
    collected = discover_collected_cases(ROOT / "outputs" / "beta2")
    if args.case_id:
        selected = tuple(args.case_id)
        if len(selected) != len(set(selected)):
            raise RuntimeError("duplicate case ids")
        missing = [
            case_id
            for case_id in selected
            if case_id not in manifest["partitions"]["fit"] or case_id not in collected
        ]
        if missing:
            raise RuntimeError(f"explicit cases are not existing fit artifacts: {missing}")
    else:
        selected = select_existing_fit_cases(manifest, collected, args.cases)
    if set(selected) & set(splits["test"]):
        raise RuntimeError("historical test identities entered the oracle run")
    if any(case_id not in manifest["partitions"]["fit"] for case_id in selected):
        raise RuntimeError("selected case is outside the fit partition")
    for role in ("calibration", "evaluation"):
        if set(selected) & set(manifest["partitions"][role]):
            raise RuntimeError(f"selected case belongs to {role}")
    case_to_subject = {record["patient_id"]: subject_id(record, cfg) for record in records}
    subjects = tuple(case_to_subject[case_id] for case_id in selected)
    originals = {case_id: original_artifacts(case_id) for case_id in selected}
    selected_records = records_for(records, selected, splits["test"])
    code_commit = __import__("subprocess").run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    expert_hashes, controller_hash = checkpoint_hashes(cfg)
    provenance = {
        **case_provenance(
            cfg,
            splits,
            expert_hashes=expert_hashes,
            controller_hash=controller_hash,
            policy=__import__(
                "brats_debate.uncertainty.extraction", fromlist=["VoxelSamplingPolicy"]
            ).VoxelSamplingPolicy(),
            code_commit=code_commit,
        ),
        "analysis": "frozen_controller_oracle_gap",
        "sample_reuse": "exact_original_flat_index",
        "highres_setting": "proposal_limited_fusion",
        "uncommitted_extraction_sha256": file_sha256(
            ROOT / "src" / "brats_debate" / "uncertainty" / "extraction.py"
        ),
        "partition_manifest_hash": manifest["manifest_hash"],
        "partition_role": "fit",
    }
    device = device_for(cfg["device"])
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("the authorized oracle run requires MPS")
    args.output.mkdir(exist_ok=args.resume)
    (args.output / "cases").mkdir(exist_ok=args.resume)
    report_path = args.output / (
        "continuation_report.json" if args.resume else "oracle_report.json"
    )
    checkpoint_path = args.output / (
        "continuation_checkpoint.json" if args.resume else "checkpoint.json"
    )
    seed_everything(cfg["seed"])
    models, loaded_expert_hashes = load_experts(cfg, splits)
    controller, loaded_controller_hash = load_controller(cfg, splits, loaded_expert_hashes)
    if loaded_expert_hashes != expert_hashes or loaded_controller_hash != controller_hash:
        raise RuntimeError("checkpoint hashes changed while loading")

    def stop_handler(_signum, _frame):
        raise BenchmarkStop("oracle safety or time limit triggered")

    signal.signal(signal.SIGALRM, stop_handler)
    signal.signal(signal.SIGUSR1, stop_handler)
    signal.alarm(args.max_seconds)
    completed = []
    status, error = "running", None
    started = time.perf_counter()
    try:
        for index, (case_id, record, case_subject) in enumerate(
            zip(selected, selected_records, subjects)
        ):
            if index == GROUP_SIZE:
                free = shutil.disk_usage(args.output).free
                observed_drop = max(
                    (
                        case["resources"]["before"]["disk_free_bytes"]
                        - case["resources"]["after"]["disk_free_bytes"]
                    )
                    for case in completed
                )
                if free < NEXT_GROUP_FREE_BYTES or free - max(0, observed_drop) < STOP_FREE_BYTES:
                    raise BenchmarkStop("insufficient disk headroom for the second group")
            summary = run_case(
                case_id,
                record,
                case_subject,
                models,
                controller,
                cfg,
                device,
                provenance,
                args.output,
                originals[case_id][0],
                originals[case_id][1],
            )
            completed.append(summary)
            write_progress(
                checkpoint_path,
                {
                    "status": "running",
                    "completed_cases": [item["patient_id"] for item in completed],
                    "elapsed_seconds": time.perf_counter() - started,
                    "resources": resource_snapshot(args.output),
                },
            )
            print(json.dumps({"completed_case": case_id, "index": index + 1}, indent=2), flush=True)
        status = "complete"
    except Exception as exc:
        status = "stopped" if isinstance(exc, BenchmarkStop) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, BenchmarkStop):
            raise
    finally:
        signal.alarm(0)
        report = {
            "format": "beta2_controller_oracle_report_v1",
            "status": status,
            "error": error,
            "selected_cases": list(selected),
            "selected_subjects": list(subjects),
            "completed_cases": [item["patient_id"] for item in completed],
            "seconds": time.perf_counter() - started,
            "pooled_case_means": aggregate(completed) if completed else {},
            "cases": completed,
            "provenance": provenance,
            "final_resources": resource_snapshot(args.output),
            "scientific_scope": (
                "Ten fit cases reused from the exploratory validation partition. "
                "Not a Dice or HD95 result and not evidence of generalization."
            ),
        }
        write_progress(report_path, report)
        del controller, models
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    if status != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
