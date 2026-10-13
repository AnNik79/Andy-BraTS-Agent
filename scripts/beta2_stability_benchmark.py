"""Run the approved three-case Stage B stability benchmark in one process."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import torch
import yaml

from brats_debate.config import (
    SEGMENTATION_EXPERTS,
    device_for,
    file_hash,
    load_config,
    seed_everything,
)
from brats_debate.data.brats_dataset import (
    discover_patients,
    load_patient,
    make_splits,
    subject_id,
    validate_training_audit,
)
from brats_debate.inference.pipeline import (
    expert_checkpoint_path,
    load_controller,
    load_experts,
)
from brats_debate.training.train_controller import (
    assert_split_isolation,
    atomic_write_text,
    records_for,
)
from brats_debate.uncertainty.extraction import (
    VoxelSamplingPolicy,
    case_provenance,
    extract_sampled_case,
    load_case_artifact,
    predict_experts_compact,
    save_case_artifact,
    stream_controller_and_uncertainty,
)


ROOT = Path(__file__).resolve().parents[1]
APPROVED_CASES = (
    "BraTS-GLI-00006-000",
    "BraTS-GLI-00025-000",
    "BraTS-GLI-00031-000",
)
PREVIOUS_CASES = (
    "BraTS-GLI-00006-000",
    "BraTS-GLI-00019-000",
)
EXPERIMENT_NAME = "stability_benchmark_20261013"
MINIMUM_FREE_BYTES = 15 * 1024**3
MINIMUM_PRESSURE_PERCENT = 5
FEATURE_NAMES = (
    "js_disagreement",
    "predictive_entropy",
    "boundary_probability",
    "boundary_distance",
    "mri_t1n",
    "mri_t1c",
    "mri_t2w",
    "mri_t2f",
)


class BenchmarkStop(RuntimeError):
    pass


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
        "--output",
        type=Path,
        default=ROOT / "outputs" / "beta2" / EXPERIMENT_NAME,
    )
    parser.add_argument("--max-seconds", type=int, default=3600)
    return parser.parse_args()


def verified_code_commit() -> str:
    relevant = (
        "src/brats_debate/uncertainty",
        "scripts/beta2_error_pilot.py",
        "scripts/beta2_stability_benchmark.py",
        "configs/beta2",
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *relevant],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "commit Stage B and benchmark-runner changes before real extraction"
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def swap_used_mib():
    try:
        text = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"used = ([0-9.]+)([KMG])", text)
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2)
    return value * {"K": 1 / 1024, "M": 1, "G": 1024}[unit]


def memory_free_percent():
    try:
        text = subprocess.run(
            ["/usr/bin/memory_pressure"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"System-wide memory free percentage: ([0-9]+)%", text)
    return int(match.group(1)) if match else None


def mps_memory_mib():
    if not torch.backends.mps.is_available():
        return None
    return {
        "current_allocated": torch.mps.current_allocated_memory() / 1024**2,
        "driver_allocated": torch.mps.driver_allocated_memory() / 1024**2,
        "recommended_max": torch.mps.recommended_max_memory() / 1024**2,
    }


def resource_snapshot(output: Path):
    process = psutil.Process()
    return {
        "rss_mib": process.memory_info().rss / 1024**2,
        "mps_mib": mps_memory_mib(),
        "swap_used_mib": swap_used_mib(),
        "memory_free_percent": memory_free_percent(),
        "disk_free_bytes": shutil.disk_usage(output).free,
    }


class CaseResourceMonitor:
    """Sample per-case RSS/MPS and periodic system pressure."""

    def __init__(self, output: Path, interval_seconds=0.5):
        self.output = output
        self.interval_seconds = interval_seconds
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread = None
        self._sample_count = 0
        self._peak_rss = 0
        self._peak_current = 0
        self._peak_driver = 0
        self._min_pressure = None
        self._max_swap = None
        self._min_disk = None
        self._critical_pressure_samples = 0
        self.critical_reason = None

    def _fast_sample(self):
        self._sample_count += 1
        self._peak_rss = max(self._peak_rss, self._process.memory_info().rss)
        self._min_disk = min(
            self._min_disk or shutil.disk_usage(self.output).free,
            shutil.disk_usage(self.output).free,
        )
        if torch.backends.mps.is_available():
            self._peak_current = max(
                self._peak_current, int(torch.mps.current_allocated_memory())
            )
            self._peak_driver = max(
                self._peak_driver, int(torch.mps.driver_allocated_memory())
            )

    def _system_sample(self):
        pressure = memory_free_percent()
        swap = swap_used_mib()
        if pressure is not None:
            self._min_pressure = (
                pressure
                if self._min_pressure is None
                else min(self._min_pressure, pressure)
            )
            self._critical_pressure_samples = (
                self._critical_pressure_samples + 1
                if pressure <= MINIMUM_PRESSURE_PERCENT
                else 0
            )
        if swap is not None:
            self._max_swap = swap if self._max_swap is None else max(self._max_swap, swap)
        if self._min_disk is not None and self._min_disk < MINIMUM_FREE_BYTES:
            self.critical_reason = "internal free space fell below 15 GiB"
        elif self._critical_pressure_samples >= 2:
            self.critical_reason = (
                f"memory free percentage remained <= {MINIMUM_PRESSURE_PERCENT}%"
            )
        if self.critical_reason is not None:
            os.kill(os.getpid(), signal.SIGUSR1)

    def _run(self):
        system_every = max(1, round(5 / self.interval_seconds))
        while not self._stop.wait(self.interval_seconds):
            self._fast_sample()
            if self._sample_count % system_every == 0:
                self._system_sample()

    def start(self):
        self._fast_sample()
        self._system_sample()
        self._thread = threading.Thread(
            target=self._run, name="case-resource-monitor", daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, 4 * self.interval_seconds))
            self._thread = None
        self._fast_sample()
        self._system_sample()

    def summary(self):
        return {
            "sampling_interval_seconds": self.interval_seconds,
            "sample_count": self._sample_count,
            "peak_rss_mib": self._peak_rss / 1024**2,
            "peak_mps_current_allocated_mib": self._peak_current / 1024**2,
            "peak_mps_driver_allocated_mib": self._peak_driver / 1024**2,
            "minimum_memory_free_percent": self._min_pressure,
            "maximum_swap_used_mib": self._max_swap,
            "minimum_disk_free_bytes": self._min_disk,
            "critical_reason": self.critical_reason,
        }


def checkpoint_hashes(cfg):
    expert_hashes = {
        name: file_hash(expert_checkpoint_path(cfg, name))
        for name in SEGMENTATION_EXPERTS
    }
    controller_path = (
        Path(cfg["output_dir"]) / "checkpoints" / "controller" / "best.pt"
    )
    checkpoint = torch.load(controller_path, map_location="cpu", weights_only=True)
    if checkpoint.get("role") != "controller" or checkpoint.get("epoch") != 19:
        raise RuntimeError("the selected controller is not the frozen epoch-19 controller")
    return expert_hashes, file_hash(controller_path)


def validate_case_selection(cases, splits, subjects):
    if tuple(cases) != APPROVED_CASES:
        raise ValueError("the benchmark case list differs from the approved list")
    if any(case not in splits["validation"] for case in cases):
        raise ValueError("every benchmark case must belong to validation")
    if set(cases) & set(splits["test"]):
        raise ValueError("historical test cases are forbidden")
    if len(set(subjects)) != len(cases):
        raise ValueError("benchmark cases must represent different subjects")
    if "BraTS-GLI-00019-000" in cases:
        raise ValueError("previous optimized pilot must not be rerun")


def array_sha256(value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def validate_artifact(artifact):
    arrays, metadata = artifact["arrays"], artifact["metadata"]
    if tuple(metadata["feature_names"]) != FEATURE_NAMES:
        raise ValueError("unexpected Stage B feature schema")
    if arrays["X"].shape != (metadata["sample_count"], len(FEATURE_NAMES)):
        raise ValueError("sample count and feature shape disagree")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("artifact contains non-finite arrays")
    if (
        (arrays["inclusion_probability"] <= 0).any()
        or (arrays["inclusion_probability"] > 1).any()
        or (arrays["sample_weight"] <= 0).any()
    ):
        raise ValueError("sampling probabilities or weights are invalid")
    np.testing.assert_allclose(
        arrays["sample_weight"],
        1 / arrays["inclusion_probability"],
        rtol=1e-6,
    )


def compare_artifacts(original, optimized):
    result = {"arrays": {}, "metadata": {}}
    for key in sorted(original["arrays"]):
        before = original["arrays"][key]
        after = optimized["arrays"][key]
        same_shape = before.shape == after.shape
        exact = same_shape and np.array_equal(before, after)
        comparison = {"same_shape": same_shape, "exact": exact}
        if same_shape and np.issubdtype(before.dtype, np.number):
            difference = np.abs(before.astype(np.float64) - after.astype(np.float64))
            comparison.update(
                maximum_absolute_difference=float(difference.max(initial=0)),
                mean_absolute_difference=float(difference.mean()),
                changed_values=int(np.count_nonzero(difference)),
            )
        result["arrays"][key] = comparison
    for key in (
        "feature_names",
        "spatial_shape",
        "class_order",
        "sample_count",
        "full_domain_voxels",
        "full_error_voxels",
        "controller_prediction_labels",
        "target_labels",
    ):
        result["metadata"][key] = {
            "exact": original["metadata"][key] == optimized["metadata"][key],
            "original": original["metadata"][key],
            "optimized": optimized["metadata"][key],
        }
    result["controller_probability_limitation"] = (
        "Compact artifacts do not archive full controller probability maps. "
        "Equality is directly testable for sampled error targets and uncertainty "
        "features, but not for every full-volume controller probability."
    )
    return result


def write_progress(path, payload):
    atomic_write_text(path, json.dumps(payload, indent=2, allow_nan=False))


def run_case(
    *,
    case_id,
    record,
    case_subject,
    models,
    controller,
    cfg,
    device,
    policy,
    provenance,
    output,
):
    artifact_path = output / "cases" / f"{case_id}.npz"
    summary_path = output / "cases" / f"{case_id}.summary.json"
    if artifact_path.exists() or summary_path.exists():
        raise RuntimeError(f"refusing to overwrite benchmark output for {case_id}")

    monitor = CaseResourceMonitor(output)
    before = resource_snapshot(output)
    monitor.start()
    start = time.perf_counter()
    patient = outputs = final = uncertainty_maps = artifact = None
    sample_count = None
    timings = {}
    try:
        load_start = time.perf_counter()
        patient = load_patient(record, cfg)
        timings["patient_load_seconds"] = time.perf_counter() - load_start

        expert_start = time.perf_counter()
        outputs = predict_experts_compact(patient, models, cfg, device)
        timings["expert_inference_seconds"] = time.perf_counter() - expert_start
        gc.collect()
        torch.mps.empty_cache()

        controller_start = time.perf_counter()
        image = torch.from_numpy(patient["image"])[None]
        final, uncertainty_maps, _ = stream_controller_and_uncertainty(
            controller,
            image,
            outputs,
            cfg,
            device,
            slab_depth=8,
            return_weights=False,
        )
        timings["controller_and_debate_seconds"] = (
            time.perf_counter() - controller_start
        )
        timings.update(
            {
                f"{key}_seconds": value
                for key, value in uncertainty_maps["host_timing_seconds"].items()
            }
        )

        extraction_start = time.perf_counter()
        artifact = extract_sampled_case(
            patient,
            outputs,
            None,
            final,
            cfg,
            subject_id=case_subject,
            policy=policy,
            uncertainty_maps=uncertainty_maps,
        )
        timings["feature_extraction_seconds"] = (
            time.perf_counter() - extraction_start
        )
        validate_artifact(artifact)
        sample_count = artifact["metadata"]["sample_count"]

        prediction = final.argmax(1)[0].numpy()
        numerical = {
            **uncertainty_maps["equivalence"],
            "controller_prediction_sha256": array_sha256(prediction),
            "js_disagreement_sha256": array_sha256(
                uncertainty_maps["js_disagreement"].numpy()
            ),
            "predictive_entropy_sha256": array_sha256(
                uncertainty_maps["predictive_entropy"].numpy()
            ),
            "all_features_finite": bool(np.isfinite(artifact["arrays"]["X"]).all()),
            "sampling_weights_valid": True,
        }
        artifact["metadata"]["benchmark"] = {
            "experiment": EXPERIMENT_NAME,
            "same_process_sequential": True,
            "case_index": APPROVED_CASES.index(case_id),
            "timings": timings,
            "numerical": numerical,
        }
        save_start = time.perf_counter()
        save_case_artifact(artifact_path, artifact, provenance=provenance)
        timings["artifact_save_seconds"] = time.perf_counter() - save_start
    finally:
        del artifact, uncertainty_maps, final, outputs, patient
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        monitor.stop()

    after = resource_snapshot(output)
    timings["total_seconds"] = time.perf_counter() - start
    resources = monitor.summary()
    resources.update(
        before=before,
        after=after,
        swap_growth_mib=(
            after["swap_used_mib"] - before["swap_used_mib"]
            if after["swap_used_mib"] is not None
            and before["swap_used_mib"] is not None
            else None
        ),
        disk_free_change_bytes=after["disk_free_bytes"] - before["disk_free_bytes"],
    )
    summary = {
        "format": "beta2_stage_b_stability_case_v1",
        "complete": True,
        "patient_id": case_id,
        "subject_id": case_subject,
        "source_split": "validation",
        "artifact": str(artifact_path),
        "artifact_bytes": artifact_path.stat().st_size,
        "sample_count": sample_count,
        "timings": timings,
        "resources": resources,
        "numerical": numerical,
        "provenance": provenance,
    }
    write_progress(summary_path, summary)
    return summary


def main():
    args = parse_args()
    if args.max_seconds < 1 or args.max_seconds > 3600:
        raise ValueError("--max-seconds must be in [1,3600]")
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT.resolve()):
        raise ValueError("benchmark output must remain inside the repository")
    if args.output.exists() and any(args.output.iterdir()):
        raise RuntimeError("benchmark output directory is not empty")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cases").mkdir(exist_ok=True)

    cfg = load_config(args.config)
    stage_b_cfg = yaml.safe_load(args.stage_b_config.read_text())["stage_b"]
    sampling_cfg = stage_b_cfg["sampling"]
    policy = VoxelSamplingPolicy(
        seed=int(sampling_cfg["seed"]),
        error=int(sampling_cfg["error"]),
        boundary_correct=int(sampling_cfg["boundary_correct"]),
        foreground_correct=int(sampling_cfg["foreground_correct"]),
        other_correct=int(sampling_cfg["other_correct"]),
    )
    validate_training_audit(cfg)
    seed_everything(cfg["seed"])
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    assert_split_isolation(splits)
    selected_records = records_for(records, APPROVED_CASES, splits["test"])
    subjects = [subject_id(record, cfg) for record in selected_records]
    validate_case_selection(APPROVED_CASES, splits, subjects)

    code_commit = verified_code_commit()
    expected_expert_hashes, expected_controller_hash = checkpoint_hashes(cfg)
    provenance = case_provenance(
        cfg,
        splits,
        expert_hashes=expected_expert_hashes,
        controller_hash=expected_controller_hash,
        policy=policy,
        code_commit=code_commit,
    )
    device = device_for(cfg["device"])
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("the approved stability benchmark requires MPS")
    if shutil.disk_usage(args.output).free < MINIMUM_FREE_BYTES:
        raise BenchmarkStop("internal free space is below the 15 GiB threshold")

    def stop_handler(_signum, _frame):
        raise BenchmarkStop("benchmark safety or time limit triggered")

    previous_alarm = signal.signal(signal.SIGALRM, stop_handler)
    previous_safety = signal.signal(signal.SIGUSR1, stop_handler)
    signal.alarm(args.max_seconds)
    benchmark_start = time.perf_counter()
    models = controller = None
    model_load_seconds = None
    completed = []
    overall_path = args.output / "benchmark.summary.json"
    status = "running"
    error = None
    try:
        model_start = time.perf_counter()
        models, expert_hashes = load_experts(cfg, splits)
        controller, controller_hash = load_controller(cfg, splits, expert_hashes)
        if (
            expert_hashes != expected_expert_hashes
            or controller_hash != expected_controller_hash
        ):
            raise RuntimeError("checkpoint hashes changed during benchmark setup")
        model_load_seconds = time.perf_counter() - model_start

        for index, (case_id, record, case_subject) in enumerate(
            zip(APPROVED_CASES, selected_records, subjects)
        ):
            elapsed = time.perf_counter() - benchmark_start
            if completed:
                conservative_next = (
                    max(item["timings"]["total_seconds"] for item in completed) * 1.25
                    + 120
                )
                if elapsed + conservative_next >= args.max_seconds:
                    raise BenchmarkStop(
                        "insufficient time remains to safely launch the next case"
                    )
            case_summary = run_case(
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
            )
            completed.append(case_summary)
            progress = {
                "format": "beta2_stage_b_stability_benchmark_v1",
                "status": "running",
                "approved_cases": list(APPROVED_CASES),
                "completed_cases": [item["patient_id"] for item in completed],
                "selection_rationale": (
                    "Original pilot plus the first two validation cases in frozen "
                    "order after excluding both previously benchmarked cases; "
                    "selection did not inspect labels or performance."
                ),
                "model_load_seconds": model_load_seconds,
                "elapsed_seconds": time.perf_counter() - benchmark_start,
                "between_case_after_cleanup": resource_snapshot(args.output),
                "cases": completed,
                "provenance": provenance,
            }
            write_progress(overall_path, progress)
            print(
                json.dumps(
                    {
                        "completed_case": case_id,
                        "case_index": index + 1,
                        "total_cases": len(APPROVED_CASES),
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
            "format": "beta2_stage_b_stability_benchmark_v1",
            "status": status,
            "approved_cases": list(APPROVED_CASES),
            "completed_cases": [item["patient_id"] for item in completed],
            "selection_rationale": (
                "Original pilot plus the first two validation cases in frozen "
                "order after excluding both previously benchmarked cases; "
                "selection did not inspect labels or performance."
            ),
            "total_seconds": time.perf_counter() - benchmark_start,
            "model_load_seconds": model_load_seconds,
            "error": error,
            "final_resources": resource_snapshot(args.output),
            "cases": completed,
            "provenance": provenance,
        }
        write_progress(overall_path, overall)

    if status != "complete":
        print(json.dumps(overall, indent=2), flush=True)
        raise SystemExit(2)

    previous_root = (
        ROOT / "outputs" / "beta2" / "error_prediction_pilot" / "cases"
    )
    original_summary = json.loads(
        (previous_root / "BraTS-GLI-00006-000.summary.json").read_text()
    )
    optimized_summary = completed[0]
    original_artifact = load_case_artifact(
        previous_root / "BraTS-GLI-00006-000.npz",
        expected_provenance=original_summary["provenance"],
    )
    optimized_artifact = load_case_artifact(
        Path(optimized_summary["artifact"]),
        expected_provenance=optimized_summary["provenance"],
    )
    comparison = compare_artifacts(original_artifact, optimized_artifact)
    comparison.update(
        original_case="BraTS-GLI-00006-000",
        original_artifact=str(previous_root / "BraTS-GLI-00006-000.npz"),
        optimized_artifact=optimized_summary["artifact"],
        protocol_hash_exact=(
            original_summary["provenance"]["protocol_hash"]
            == optimized_summary["provenance"]["protocol_hash"]
        ),
        split_hash_exact=(
            original_summary["provenance"]["split_hash"]
            == optimized_summary["provenance"]["split_hash"]
        ),
        expert_hashes_exact=(
            original_summary["provenance"]["expert_hashes"]
            == optimized_summary["provenance"]["expert_hashes"]
        ),
        controller_hash_exact=(
            original_summary["provenance"]["controller_hash"]
            == optimized_summary["provenance"]["controller_hash"]
        ),
    )
    write_progress(args.output / "same_case_comparison.json", comparison)
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
