"""Run one bounded validation-case Stage B extraction. Never fits a model."""

import argparse
import gc
import json
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import subprocess
import sys
import threading
import time

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
    case_artifact_is_complete,
    case_provenance,
    extract_sampled_case,
    load_case_artifact,
    predict_experts_compact,
    save_case_artifact,
    stream_controller_and_uncertainty,
)


ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--patient", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-seconds", type=int, default=1800)
    return parser.parse_args()


def verified_code_commit() -> str:
    relevant = (
        "src/brats_debate/uncertainty",
        "scripts/beta2_error_pilot.py",
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
            "commit Stage B source/config changes before real extraction so "
            "resume provenance identifies the executed code"
        )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def peak_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    return value / 1024**2


def swap_used_mib():
    if sys.platform != "darwin":
        return None
    try:
        text = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.search(r"used = ([0-9.]+)([KMG])", text)
    if not match:
        return None
    value, unit = float(match.group(1)), match.group(2)
    return value * {"K": 1 / 1024, "M": 1, "G": 1024}[unit]


def mps_memory_mib():
    if not torch.backends.mps.is_available():
        return None
    return {
        "current_allocated": torch.mps.current_allocated_memory() / 1024**2,
        "driver_allocated": torch.mps.driver_allocated_memory() / 1024**2,
        "recommended_max": torch.mps.recommended_max_memory() / 1024**2,
    }


class MPSMemoryMonitor:
    """Sample allocator and driver memory without forcing synchronization."""

    def __init__(self, interval_seconds=0.5):
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread = None
        self._sample_count = 0
        self._peak_current = 0
        self._peak_driver = 0

    def _sample(self):
        if not torch.backends.mps.is_available():
            return
        self._sample_count += 1
        self._peak_current = max(
            self._peak_current, int(torch.mps.current_allocated_memory())
        )
        self._peak_driver = max(
            self._peak_driver, int(torch.mps.driver_allocated_memory())
        )

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self):
        self._sample()
        if torch.backends.mps.is_available():
            self._thread = threading.Thread(
                target=self._run, name="mps-memory-monitor", daemon=True
            )
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self.interval_seconds))
            self._thread = None
        self._sample()

    def summary(self):
        return {
            "sampling_interval_seconds": self.interval_seconds,
            "sample_count": self._sample_count,
            "peak_current_allocated_mib": self._peak_current / 1024**2,
            "peak_driver_allocated_mib": self._peak_driver / 1024**2,
        }


def checkpoint_hashes(cfg):
    expert_hashes = {
        name: file_hash(expert_checkpoint_path(cfg, name))
        for name in SEGMENTATION_EXPERTS
    }
    controller_path = (
        Path(cfg["output_dir"]) / "checkpoints" / "controller" / "best.pt"
    )
    return expert_hashes, file_hash(controller_path)


class PilotTimeout(TimeoutError):
    pass


def _timeout(_signum, _frame):
    raise PilotTimeout("one-case pilot exceeded its approved wall-clock budget")


def main():
    process_start = time.perf_counter()
    args = parse_args()
    if args.max_seconds < 1 or args.max_seconds > 1800:
        raise ValueError("--max-seconds must be in [1,1800]")
    cfg = load_config(args.config)
    stage_b_cfg = yaml.safe_load(args.stage_b_config.read_text())["stage_b"]
    configured_output = (
        args.stage_b_config.parent / stage_b_cfg["pilot_output_dir"]
    ).resolve()
    args.output = configured_output if args.output is None else args.output.resolve()
    sampling_cfg = dict(stage_b_cfg["sampling"])
    policy = VoxelSamplingPolicy(
        seed=int(sampling_cfg["seed"]),
        error=int(sampling_cfg["error"]),
        boundary_correct=int(sampling_cfg["boundary_correct"]),
        foreground_correct=int(sampling_cfg["foreground_correct"]),
        other_correct=int(sampling_cfg["other_correct"]),
    )
    if policy.maximum_voxels != int(sampling_cfg["maximum_voxels_per_case"]):
        raise ValueError("Stage B sampling caps do not match the configured maximum")
    validate_training_audit(cfg)
    seed_everything(cfg["seed"])
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    assert_split_isolation(splits)
    patient_id = args.patient or splits["validation"][0]
    if patient_id not in splits["validation"] or patient_id in splits["test"]:
        raise RuntimeError("pilot case must belong only to frozen validation")
    record = records_for(records, [patient_id], splits["test"])[0]
    case_subject = subject_id(record, cfg)
    expected_expert_hashes, expected_controller_hash = checkpoint_hashes(cfg)
    provenance = case_provenance(
        cfg,
        splits,
        expert_hashes=expected_expert_hashes,
        controller_hash=expected_controller_hash,
        policy=policy,
        code_commit=verified_code_commit(),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    artifact_path = args.output / "cases" / f"{patient_id}.npz"
    summary_path = args.output / "cases" / f"{patient_id}.summary.json"
    if case_artifact_is_complete(
        artifact_path,
        patient_id=patient_id,
        expected_provenance=provenance,
    ) and summary_path.is_file():
        artifact = load_case_artifact(
            artifact_path, expected_provenance=provenance
        )
        print(
            json.dumps(
                {
                    "resumed": True,
                    "artifact": str(artifact_path),
                    "metadata": artifact["metadata"],
                },
                indent=2,
            ),
            flush=True,
        )
        return
    if artifact_path.exists() or summary_path.exists():
        raise RuntimeError(
            "refusing to replace an existing pilot artifact with different "
            "provenance; select a new permitted validation patient"
        )

    start = time.perf_counter()
    swap_before = swap_used_mib()
    free_before = shutil.disk_usage(args.output.parent).free
    device = device_for(cfg["device"])
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("the approved real-case pilot requires configured MPS")
    models = controller = patient = outputs = final = uncertainty_maps = None
    memory_monitor = MPSMemoryMonitor()
    memory_monitor.start()
    previous_handler = signal.signal(signal.SIGALRM, _timeout)
    signal.alarm(args.max_seconds)
    try:
        load_start = time.perf_counter()
        models, expert_hashes = load_experts(cfg, splits)
        controller, controller_hash = load_controller(cfg, splits, expert_hashes)
        if (
            expert_hashes != expected_expert_hashes
            or controller_hash != expected_controller_hash
        ):
            raise RuntimeError("checkpoint hashes changed during pilot setup")
        patient = load_patient(record, cfg)
        load_seconds = time.perf_counter() - load_start

        expert_start = time.perf_counter()
        outputs = predict_experts_compact(patient, models, cfg, device)
        expert_seconds = time.perf_counter() - expert_start
        mps_before_phase_cleanup = mps_memory_mib()
        gc.collect()
        # Expert models have returned to CPU. Release only their dead cached MPS
        # blocks once, at this phase boundary, rather than clearing per slab.
        torch.mps.empty_cache()
        mps_after_phase_cleanup = mps_memory_mib()

        gate_start = time.perf_counter()
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
        gate_seconds = time.perf_counter() - gate_start
        gate_component_seconds = uncertainty_maps["host_timing_seconds"]
        numerical_equivalence = uncertainty_maps["equivalence"]

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
        extraction_seconds = time.perf_counter() - extraction_start
        runtime_before_save = time.perf_counter() - start
        if runtime_before_save >= args.max_seconds:
            raise PilotTimeout("pilot reached the approved budget before artifact save")
        artifact["metadata"]["runtime"] = {
            "checkpoint_and_patient_load_seconds": load_seconds,
            "expert_inference_seconds": expert_seconds,
            "controller_and_debate_seconds": gate_seconds,
            "controller_host_seconds": gate_component_seconds["controller"],
            "debate_and_uncertainty_host_seconds": gate_component_seconds[
                "debate_and_uncertainty"
            ],
            "feature_extraction_seconds": extraction_seconds,
            "runtime_before_save_seconds": runtime_before_save,
            "peak_rss_mib_before_save": peak_rss_mib(),
            "mps_sampled_peaks": memory_monitor.summary(),
            "mps_before_phase_cleanup_mib": mps_before_phase_cleanup,
            "mps_after_phase_cleanup_mib": mps_after_phase_cleanup,
            "swap_used_mib_before": swap_before,
            "swap_used_mib_before_save": swap_used_mib(),
        }
        save_start = time.perf_counter()
        save_case_artifact(
            artifact_path, artifact, provenance=provenance
        )
        save_seconds = time.perf_counter() - save_start
        total_seconds = time.perf_counter() - start
        free_after = shutil.disk_usage(args.output.parent).free
        summary = {
            "format": "beta2_error_prediction_pilot_summary_v1",
            "complete": True,
            "patient_id": patient_id,
            "subject_id": case_subject,
            "source_split": "validation",
            "exploratory": True,
            "fitted_model": False,
            "artifact": str(artifact_path),
            "artifact_bytes": artifact_path.stat().st_size,
            "runtime_seconds": {
                "checkpoint_and_patient_load": load_seconds,
                "expert_inference": expert_seconds,
                "controller_and_debate": gate_seconds,
                "controller_host": gate_component_seconds["controller"],
                "debate_and_uncertainty_host": gate_component_seconds[
                    "debate_and_uncertainty"
                ],
                "feature_extraction": extraction_seconds,
                "artifact_save": save_seconds,
                "total": total_seconds,
                "command_wall_clock": time.perf_counter() - process_start,
            },
            "resources": {
                "peak_rss_mib": peak_rss_mib(),
                "mps_sampled_peaks": memory_monitor.summary(),
                "mps_after_mib": mps_memory_mib(),
                "swap_used_mib_before": swap_before,
                "swap_used_mib_after": swap_used_mib(),
                "disk_free_bytes_before": free_before,
                "disk_free_bytes_after": free_after,
            },
            "numerical_equivalence": numerical_equivalence,
            "metadata": artifact["metadata"],
            "provenance": provenance,
        }
        atomic_write_text(
            summary_path, json.dumps(summary, indent=2, allow_nan=False)
        )
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        memory_monitor.stop()
        if controller is not None:
            try:
                controller.cpu()
            except Exception:
                pass
        if models is not None:
            for model in models.values():
                try:
                    model.cpu()
                except Exception:
                    pass
        del uncertainty_maps, final, outputs, patient, controller, models
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
