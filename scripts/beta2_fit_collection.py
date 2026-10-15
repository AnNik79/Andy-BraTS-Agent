"""Collect one approved three-case Stage B fit-partition batch."""

import argparse
import gc
import json
from pathlib import Path
import shutil
import signal
import subprocess
import time

import torch
import yaml

from beta2_stability_benchmark import (
    BenchmarkStop,
    MINIMUM_FREE_BYTES,
    checkpoint_hashes,
    resource_snapshot,
    run_case,
    write_progress,
)
from freeze_beta2_partitions import validate_frozen_manifest
from brats_debate.config import device_for, load_config, seed_everything
from brats_debate.data.brats_dataset import (
    discover_patients,
    make_splits,
    subject_id,
    validate_training_audit,
)
from brats_debate.inference.pipeline import load_controller, load_experts
from brats_debate.training.train_controller import (
    assert_split_isolation,
    records_for,
)
from brats_debate.uncertainty.extraction import (
    VoxelSamplingPolicy,
    case_provenance,
    load_case_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_ID = "fit_collection_batch_20261013_01"
BATCH_PROFILES = {
    DEFAULT_BATCH_ID: {
        "cases": (
            "BraTS-GLI-00049-000",
            "BraTS-GLI-00056-000",
            "BraTS-GLI-00101-000",
        ),
        "distinct_subjects": True,
    },
    "fit_collection_batch_20261013_02": {
        "cases": (
            "BraTS-GLI-00056-001",
            "BraTS-GLI-00104-000",
            "BraTS-GLI-00112-000",
            "BraTS-GLI-00133-000",
            "BraTS-GLI-00167-000",
            "BraTS-GLI-00188-000",
            "BraTS-GLI-00222-000",
            "BraTS-GLI-00230-000",
            "BraTS-GLI-00235-000",
            "BraTS-GLI-00240-000",
        ),
        "distinct_subjects": False,
    },
    "fit_collection_batch_20261013_03": {
        "cases": (
            "BraTS-GLI-00247-000",
            "BraTS-GLI-00281-000",
            "BraTS-GLI-00291-000",
            "BraTS-GLI-00293-000",
            "BraTS-GLI-00296-000",
            "BraTS-GLI-00301-000",
            "BraTS-GLI-00304-000",
            "BraTS-GLI-00313-000",
            "BraTS-GLI-00338-000",
            "BraTS-GLI-00373-000",
        ),
        "distinct_subjects": False,
    },
    "fit_collection_batch_20261013_04": {
        "cases": (
            "BraTS-GLI-00382-000",
            "BraTS-GLI-00414-000",
            "BraTS-GLI-00451-000",
            "BraTS-GLI-00493-000",
            "BraTS-GLI-00500-000",
            "BraTS-GLI-00500-001",
            "BraTS-GLI-00501-000",
            "BraTS-GLI-00514-000",
            "BraTS-GLI-00514-001",
            "BraTS-GLI-00519-000",
        ),
        "distinct_subjects": False,
    },
    "fit_collection_batch_20261014_01": {
        "cases": (
            "BraTS-GLI-00523-000",
            "BraTS-GLI-00529-000",
            "BraTS-GLI-00552-000",
            "BraTS-GLI-00552-001",
            "BraTS-GLI-00556-000",
        ),
        "distinct_subjects": False,
    },
    "fit_collection_batch_20261014_02": {
        "cases": (
            "BraTS-GLI-00556-001",
            "BraTS-GLI-00558-000",
            "BraTS-GLI-00558-001",
            "BraTS-GLI-00559-000",
            "BraTS-GLI-00559-001",
        ),
        "distinct_subjects": False,
    },
    "fit_collection_batch_20261014_03": {
        "cases": (
            "BraTS-GLI-00568-000",
            "BraTS-GLI-00577-000",
            "BraTS-GLI-00579-000",
            "BraTS-GLI-00584-000",
            "BraTS-GLI-00591-000",
        ),
        "distinct_subjects": False,
    },
    "fit_collection_batch_20261014_04": {
        "cases": (
            "BraTS-GLI-00649-000",
            "BraTS-GLI-00649-001",
            "BraTS-GLI-00675-000",
            "BraTS-GLI-00675-001",
            "BraTS-GLI-00704-000",
        ),
        "distinct_subjects": False,
    },
    "fit_collection_batch_20261014_05": {
        "cases": (
            "BraTS-GLI-00709-000",
            "BraTS-GLI-00730-000",
            "BraTS-GLI-00730-001",
            "BraTS-GLI-00744-000",
            "BraTS-GLI-00747-000",
        ),
        "distinct_subjects": False,
    },
    "fit_collection_batch_20261014_06": {
        "cases": (
            "BraTS-GLI-00791-000",
            "BraTS-GLI-00804-000",
            "BraTS-GLI-00834-000",
            "BraTS-GLI-01002-000",
            "BraTS-GLI-01013-000",
        ),
        "distinct_subjects": False,
    },
    "fit_collection_batch_20261014_07": {
        "cases": (
            "BraTS-GLI-01016-000",
            "BraTS-GLI-01020-000",
            "BraTS-GLI-01022-000",
            "BraTS-GLI-01064-000",
            "BraTS-GLI-01065-000",
        ),
        "distinct_subjects": False,
    },
    "fit_collection_batch_20261014_08": {
        "cases": (
            "BraTS-GLI-01071-000",
            "BraTS-GLI-01077-000",
            "BraTS-GLI-01081-000",
            "BraTS-GLI-01083-000",
            "BraTS-GLI-01101-000",
        ),
        "distinct_subjects": False,
    },
    "fit_collection_batch_20261014_09": {
        "cases": (
            "BraTS-GLI-01129-000",
            "BraTS-GLI-01166-000",
            "BraTS-GLI-01176-000",
            "BraTS-GLI-01178-000",
            "BraTS-GLI-01185-000",
        ),
        "distinct_subjects": False,
    },
    "fit_collection_batch_20261014_10": {
        "cases": (
            "BraTS-GLI-01204-000",
            "BraTS-GLI-01213-000",
            "BraTS-GLI-01216-000",
            "BraTS-GLI-01232-000",
            "BraTS-GLI-01254-000",
        ),
        "distinct_subjects": False,
    },
    "fit_collection_batch_20261014_11": {
        "cases": (
            "BraTS-GLI-01260-000",
            "BraTS-GLI-01265-000",
            "BraTS-GLI-01268-000",
            "BraTS-GLI-01272-000",
            "BraTS-GLI-01278-000",
        ),
        "distinct_subjects": False,
    },
    "fit_collection_batch_20261014_12": {
        "cases": (
            "BraTS-GLI-01283-000",
            "BraTS-GLI-01288-000",
            "BraTS-GLI-01293-000",
            "BraTS-GLI-01308-000",
            "BraTS-GLI-01310-000",
        ),
        "distinct_subjects": False,
    },
    "fit_collection_batch_20261014_13": {
        "cases": (
            "BraTS-GLI-01321-000",
            "BraTS-GLI-01346-000",
            "BraTS-GLI-01381-000",
            "BraTS-GLI-01412-000",
            "BraTS-GLI-01421-000",
        ),
        "distinct_subjects": False,
    },
}
EXPECTED_CASES = BATCH_PROFILES[DEFAULT_BATCH_ID]["cases"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "brats_controller.yaml"
    )
    parser.add_argument(
        "--stage-b-config",
        type=Path,
        default=ROOT / "configs" / "beta2" / "foundation.yaml",
    )
    parser.add_argument(
        "--partition-manifest",
        type=Path,
        default=ROOT / "configs" / "beta2" / "subject_partitions.json",
    )
    parser.add_argument(
        "--batch-id",
        choices=tuple(BATCH_PROFILES),
        default=DEFAULT_BATCH_ID,
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-seconds", type=int, default=3600)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def verified_code_commit():
    relevant = (
        "src/brats_debate/uncertainty",
        "configs/beta2",
        "scripts/beta2_error_pilot.py",
        "scripts/beta2_stability_benchmark.py",
        "scripts/beta2_fit_collection.py",
        "scripts/freeze_beta2_partitions.py",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *relevant],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("commit Stage B collection code and manifest before extraction")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def discover_collected_cases(output_root):
    collected = set()
    if not output_root.is_dir():
        return collected
    for path in output_root.rglob("*.summary.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        provenance = payload.get("provenance") or {}
        patient_id = payload.get("patient_id")
        if (
            patient_id
            and provenance.get("format") == "beta2_error_prediction_case_v1"
            and provenance.get("source_split") == "validation"
            and payload.get("complete") is True
        ):
            collected.add(str(patient_id))
    return collected


def select_fit_cases(
    manifest, case_to_subject, processed, count=3, *, distinct_subjects=True
):
    selected, selected_subjects = [], set()
    for case_id in manifest["partitions"]["fit"]:
        subject = case_to_subject[case_id]
        if case_id in processed or (
            distinct_subjects and subject in selected_subjects
        ):
            continue
        selected.append(case_id)
        selected_subjects.add(subject)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError("not enough unprocessed fit-partition subjects")
    return tuple(selected)


def load_resume_state(output, selected, candidate_provenance):
    summary_path = output / "collection.summary.json"
    if not summary_path.is_file():
        raise RuntimeError("resume requires an existing collection summary")
    payload = json.loads(summary_path.read_text())
    if payload.get("format") != "beta2_stage_b_fit_collection_v1":
        raise ValueError("resume summary has an unexpected format")
    if tuple(payload.get("selected_cases") or ()) != tuple(selected):
        raise ValueError("resume summary case list differs from the fixed batch")
    if payload.get("partition_role") != "fit":
        raise ValueError("resume summary is not a fit collection")
    completed = list(payload.get("cases") or [])
    completed_ids = [item.get("patient_id") for item in completed]
    if completed_ids != list(selected[: len(completed_ids)]):
        raise ValueError("completed resume cases are not a prefix of the fixed batch")
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("resume summary contains duplicate completed cases")
    original_provenance = payload.get("provenance") or {}
    original_science = {
        key: value
        for key, value in original_provenance.items()
        if key != "code_commit"
    }
    candidate_science = {
        key: value
        for key, value in candidate_provenance.items()
        if key != "code_commit"
    }
    if original_science != candidate_science:
        raise ValueError("resume scientific provenance differs from the original run")
    for case in completed:
        artifact_path = Path(case["artifact"]).resolve()
        if not artifact_path.is_relative_to(output.resolve()):
            raise ValueError("resume artifact is outside the collection directory")
        artifact = load_case_artifact(
            artifact_path,
            expected_provenance=case["provenance"],
        )
        if artifact["metadata"]["patient_id"] != case["patient_id"]:
            raise ValueError("resume artifact patient identity mismatch")
    return {
        "completed": completed,
        "prior_elapsed_seconds": float(payload.get("total_seconds") or 0),
        "prior_model_load_seconds": float(payload.get("model_load_seconds") or 0),
        "provenance": original_provenance,
    }


def main():
    args = parse_args()
    if args.max_seconds < 1 or args.max_seconds > 9900:
        raise ValueError("--max-seconds must be in [1,9900]")
    profile = BATCH_PROFILES[args.batch_id]
    expected_cases = tuple(profile["cases"])
    if args.output is None:
        args.output = ROOT / "outputs" / "beta2" / args.batch_id
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT.resolve()):
        raise ValueError("collection output must remain on internal repository storage")
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise RuntimeError("collection output directory is not empty")
    if args.resume and not args.output.is_dir():
        raise RuntimeError("resume collection directory does not exist")

    cfg = load_config(args.config)
    stage_b_cfg = yaml.safe_load(args.stage_b_config.read_text())["stage_b"]
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    assert_split_isolation(splits)
    validate_training_audit(cfg)
    case_to_subject = {
        record["patient_id"]: subject_id(record, cfg) for record in records
    }
    manifest = json.loads(args.partition_manifest.read_text())
    validate_frozen_manifest(
        manifest,
        cfg=cfg,
        records=records,
        splits=splits,
        stage_b_cfg=stage_b_cfg,
    )
    processed = discover_collected_cases(ROOT / "outputs" / "beta2")
    if args.resume:
        processed -= set(expected_cases)
    selected = select_fit_cases(
        manifest,
        case_to_subject,
        processed,
        count=len(expected_cases),
        distinct_subjects=bool(profile["distinct_subjects"]),
    )
    if selected != expected_cases:
        raise RuntimeError(
            f"deterministic fit selection changed: expected {expected_cases}, got {selected}"
        )
    selected_subjects = tuple(case_to_subject[case_id] for case_id in selected)
    if profile["distinct_subjects"] and len(set(selected_subjects)) != len(
        selected_subjects
    ):
        raise RuntimeError("selected fit cases are not from distinct subjects")
    if set(selected) & set(splits["test"]):
        raise RuntimeError("historical test identities entered fit collection")
    if any(case_id not in manifest["partitions"]["fit"] for case_id in selected):
        raise RuntimeError("selected case is outside the frozen fit partition")
    selected_records = records_for(records, selected, splits["test"])

    sampling_cfg = stage_b_cfg["sampling"]
    policy = VoxelSamplingPolicy(
        seed=int(sampling_cfg["seed"]),
        error=int(sampling_cfg["error"]),
        boundary_correct=int(sampling_cfg["boundary_correct"]),
        foreground_correct=int(sampling_cfg["foreground_correct"]),
        other_correct=int(sampling_cfg["other_correct"]),
    )
    seed_everything(cfg["seed"])
    code_commit = verified_code_commit()
    expected_expert_hashes, expected_controller_hash = checkpoint_hashes(cfg)
    provenance = {
        **case_provenance(
            cfg,
            splits,
            expert_hashes=expected_expert_hashes,
            controller_hash=expected_controller_hash,
            policy=policy,
            code_commit=code_commit,
        ),
        "partition_manifest_hash": manifest["manifest_hash"],
        "partition_role": "fit",
    }
    runner_commit = code_commit
    device = device_for(cfg["device"])
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("the approved collection batch requires MPS")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_BYTES:
        raise BenchmarkStop("internal free space is below the 15 GiB threshold")

    overall_path = args.output / "collection.summary.json"
    prior_elapsed_seconds = 0.0
    prior_model_load_seconds = 0.0
    completed = []
    if args.resume:
        resume = load_resume_state(args.output, selected, provenance)
        completed = resume["completed"]
        prior_elapsed_seconds = resume["prior_elapsed_seconds"]
        prior_model_load_seconds = resume["prior_model_load_seconds"]
        provenance = resume["provenance"]
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "cases").mkdir()
    benchmark_start = time.perf_counter()
    models = controller = None
    model_load_seconds = None
    status, error = "running", None

    def stop_handler(_signum, _frame):
        raise BenchmarkStop("collection safety or time limit triggered")

    remaining_seconds = args.max_seconds - prior_elapsed_seconds
    if remaining_seconds <= 0:
        raise BenchmarkStop("collection has exhausted its cumulative time limit")
    previous_alarm = signal.signal(signal.SIGALRM, stop_handler)
    previous_safety = signal.signal(signal.SIGUSR1, stop_handler)
    signal.alarm(max(1, int(remaining_seconds)))
    try:
        model_start = time.perf_counter()
        models, expert_hashes = load_experts(cfg, splits)
        controller, controller_hash = load_controller(cfg, splits, expert_hashes)
        model_load_seconds = (
            prior_model_load_seconds + time.perf_counter() - model_start
        )
        if (
            expert_hashes != expected_expert_hashes
            or controller_hash != expected_controller_hash
        ):
            raise RuntimeError("checkpoint hashes changed during collection setup")

        for index, (case_id, record, case_subject) in enumerate(
            zip(selected, selected_records, selected_subjects)
        ):
            if case_id in {item["patient_id"] for item in completed}:
                continue
            elapsed = prior_elapsed_seconds + time.perf_counter() - benchmark_start
            if completed:
                conservative_next = (
                    max(item["timings"]["total_seconds"] for item in completed) * 1.25
                    + 120
                )
                if elapsed + conservative_next >= args.max_seconds:
                    raise BenchmarkStop(
                        "insufficient time remains to safely launch the next fit case"
                    )
            summary = run_case(
                case_id=case_id,
                record=record,
                case_subject=case_subject,
                models=models,
                controller=controller,
                cfg=cfg,
                device=device,
                policy=policy,
                provenance=provenance,
                output=args.output,
                experiment_name=args.batch_id,
                case_index=index,
                partition_role="fit",
            )
            completed.append(summary)
            progress = {
                "format": "beta2_stage_b_fit_collection_v1",
                "status": "running",
                "selected_cases": list(selected),
                "selected_subjects": list(selected_subjects),
                "partition_role": "fit",
                "partition_manifest_hash": manifest["manifest_hash"],
                "selection_rationale": (
                    f"First {len(selected)} unprocessed fit-partition cases in "
                    "manifest order; labels and performance were not inspected."
                ),
                "completed_cases": [item["patient_id"] for item in completed],
                "model_load_seconds": model_load_seconds,
                "elapsed_seconds": (
                    prior_elapsed_seconds + time.perf_counter() - benchmark_start
                ),
                "between_case_after_cleanup": resource_snapshot(args.output),
                "cases": completed,
                "provenance": provenance,
                "resume_runner_commit": runner_commit if args.resume else None,
            }
            write_progress(overall_path, progress)
            print(
                json.dumps(
                    {
                        "completed_case": case_id,
                        "case_index": index + 1,
                        "total_cases": len(selected),
                        "elapsed_seconds": progress["elapsed_seconds"],
                        "resources_after_cleanup": progress[
                            "between_case_after_cleanup"
                        ],
                    },
                    indent=2,
                ),
                flush=True,
            )
        status = "complete"
    except Exception as exc:
        status = "stopped" if isinstance(exc, BenchmarkStop) else "failed"
        error = f"{type(exc).__name__}: {exc}"
        if not isinstance(exc, BenchmarkStop):
            raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_alarm)
        signal.signal(signal.SIGUSR1, previous_safety)
        if controller is not None:
            controller.cpu()
        if models is not None:
            for model in models.values():
                model.cpu()
        del controller, models
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        overall = {
            "format": "beta2_stage_b_fit_collection_v1",
            "status": status,
            "selected_cases": list(selected),
            "selected_subjects": list(selected_subjects),
            "partition_role": "fit",
            "partition_manifest_hash": manifest["manifest_hash"],
            "selection_rationale": (
                f"First {len(selected)} unprocessed fit-partition cases in "
                "manifest order; labels and performance were not inspected."
            ),
            "completed_cases": [item["patient_id"] for item in completed],
            "total_seconds": (
                prior_elapsed_seconds + time.perf_counter() - benchmark_start
            ),
            "model_load_seconds": model_load_seconds,
            "error": error,
            "final_resources": resource_snapshot(args.output),
            "cases": completed,
            "provenance": provenance,
            "resume_runner_commit": runner_commit if args.resume else None,
        }
        write_progress(overall_path, overall)
    print(json.dumps(overall, indent=2), flush=True)
    if status != "complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
