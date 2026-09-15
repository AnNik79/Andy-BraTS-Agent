"""Real-data training-readiness timings. Not training. Does not write checkpoints."""
from collections import defaultdict
from contextlib import contextmanager
import argparse
import gc
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from brats_debate.config import SEGMENTATION_EXPERTS, load_config, save_json
from brats_debate.data.brats_dataset import (
    BraTSPatches, discover_patients, load_patient, make_patch_loader, validate_training_audit,
)
from brats_debate.experts import build_expert
from brats_debate.inference.patches import sliding_probabilities, window_starts
from brats_debate.training.train_expert import expert_loss, segmentation_loss, training_output
from brats_debate.evaluation.metrics import segmentation_metrics


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _memory(device, peaks):
    if device.type != "mps":
        return
    peaks["mps_allocated_bytes"] = max(peaks["mps_allocated_bytes"], torch.mps.current_allocated_memory())
    peaks["mps_driver_bytes"] = max(peaks["mps_driver_bytes"], torch.mps.driver_allocated_memory())


@contextmanager
def timed(timer, name, device, synchronize):
    if synchronize:
        _sync(device)
    start = time.perf_counter()
    yield
    if synchronize:
        _sync(device)
    timer[name] += time.perf_counter() - start


def _unpack(name, batch, device):
    if name == "boundary":
        image, target, boundary, distance = batch
        geometry = (boundary.to(device), distance.to(device))
    else:
        image, target = batch
        geometry = None
    return image.to(device), target.to(device), geometry


def train_probe(name, records, cfg, device, warmup, measure, attribution):
    train = [r for r in records if r["patient_id"] in json_splits(cfg)["expert_train"]]
    dataset = BraTSPatches(train, cfg, specialist=name == "highres", geometry=name == "boundary")
    needed = warmup + measure + attribution
    rng = torch.Generator().manual_seed(cfg["seed"] + SEGMENTATION_EXPERTS.index(name))
    indices = torch.randperm(len(dataset), generator=rng)[:needed].tolist()
    loader = make_patch_loader(dataset, indices, cfg, cfg["seed"] + 17)
    iterator = iter(loader)
    model = build_expert(name, cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    peaks = {"mps_allocated_bytes": 0, "mps_driver_bytes": 0}
    losses = []

    def run_steps(count, synchronize):
        timer = defaultdict(float)
        wall = time.perf_counter()
        for _ in range(count):
            with timed(timer, "data_batch", device, False):
                batch = next(iterator)
            if name == "boundary":
                with timed(timer, "cpu_to_device", device, synchronize):
                    image, target, geometry = _unpack(name, batch, device)
            else:
                with timed(timer, "cpu_to_device", device, synchronize):
                    image, target, geometry = _unpack(name, batch, device)
            with timed(timer, "forward", device, synchronize):
                output = training_output(model, image, name, cfg)
            with timed(timer, "loss", device, synchronize):
                loss = expert_loss(output, target, geometry)
            with timed(timer, "scalar_read", device, synchronize):
                scalar = float(loss.detach().item())
            if not math.isfinite(scalar):
                raise FloatingPointError(f"Nonfinite {name} benchmark loss")
            with timed(timer, "backward", device, synchronize):
                loss.backward()
            with timed(timer, "optimizer", device, synchronize):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            losses.append(scalar)
            _memory(device, peaks)
            del output, loss, image, target, geometry, batch
        _sync(device)
        return time.perf_counter() - wall, dict(timer)

    run_steps(warmup, False)
    throughput_s, throughput_stages = run_steps(measure, False)
    attrib_s, attrib_stages = run_steps(attribution, True)
    del model, optimizer, loader, iterator, dataset
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return {
        "warmup_steps": warmup,
        "throughput_steps": measure,
        "attribution_steps": attribution,
        "losses_finite": all(math.isfinite(v) for v in losses),
        "mean_loss": float(np.mean(losses[warmup:])),
        "throughput_seconds": throughput_s,
        "seconds_per_step": throughput_s / measure,
        "throughput_stage_seconds": throughput_stages,
        "attribution_seconds": attrib_s,
        "attribution_stage_seconds": {k: v / attribution for k, v in attrib_stages.items()},
        "peaks": peaks,
        "note": "Throughput stages are unsynchronized wall time. Attribution stages are MPS-synced.",
    }


def json_splits(cfg):
    return json.loads(Path(cfg["split_file"]).read_text())


def val_probe(name, records, cfg, device, case_ids):
    validation = [r for r in records if r["patient_id"] in case_ids]
    model = build_expert(name, cfg).to(device).eval()
    patch = cfg["highres_patch_size"] if name == "highres" else cfg["crop_size"]
    peaks = {"mps_allocated_bytes": 0, "mps_driver_bytes": 0}
    cases = []
    for record in validation:
        start = time.perf_counter()
        patient = load_patient(record, cfg)
        image = torch.from_numpy(patient["image"])[None]
        n_patches = len(window_starts(tuple(image.shape[2:]), patch, cfg["inference"]["overlap"]))
        infer_start = time.perf_counter()
        probabilities = sliding_probabilities(
            model, image, patch, device, cfg["inference"]["overlap"],
            **cfg.get("validation_performance", {}),
        )
        _sync(device)
        infer_s = time.perf_counter() - infer_start
        target = torch.from_numpy(patient["label"])[None]
        case_loss = float(segmentation_loss(probabilities, target).detach().item())
        prediction = probabilities.detach().numpy()[0].argmax(axis=0)
        metrics = segmentation_metrics(prediction, patient["label"], cfg["regions"], len(cfg["label_mapping"]))
        _memory(device, peaks)
        cases.append({
            "patient_id": record["patient_id"],
            "shape": list(patient["image"].shape[1:]),
            "patches": n_patches,
            "seconds": time.perf_counter() - start,
            "inference_seconds": infer_s,
            "loss": case_loss,
            "brats_mean_dice": float(metrics["brats_mean_dice"]),
            "finite": math.isfinite(case_loss) and math.isfinite(metrics["brats_mean_dice"]),
        })
        del probabilities, target, prediction, metrics, patient, image
    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    seconds = [c["seconds"] for c in cases]
    return {
        "cases": cases,
        "seconds_per_patient": float(np.mean(seconds)),
        "patches_per_patient": cases[0]["patches"] if cases else None,
        "peaks": peaks,
        "projected_191_seconds": float(np.mean(seconds) * 191),
    }


def projections(train_step, steps_per_epoch, val_191, checks):
    train_epoch = train_step * steps_per_epoch
    return {
        "steps_per_epoch": steps_per_epoch,
        "seconds_per_epoch_train": train_epoch,
        "seconds_full_validation": val_191,
        "hours_per_epoch_train": train_epoch / 3600,
        "hours_full_validation": val_191 / 3600,
        **{f"hours_{n}_epochs_plus_val_at_{'_'.join(map(str, checks[:i+1]))}":
           (n * train_epoch + len(checks[:i+1]) * val_191) / 3600
           for n, i in ((5, 1), (10, 2), (20, 4))},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/brats_mps_optimized.yaml")
    parser.add_argument("--output", default="outputs/training_readiness")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--attribution-steps", type=int, default=16)
    parser.add_argument("--val-cases", type=int, default=3)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg.get("torch_threads"):
        torch.set_num_threads(cfg["torch_threads"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    validate_training_audit(cfg)
    records = discover_patients(cfg)
    splits = json_splits(cfg)
    train_n = len([r for r in records if r["patient_id"] in splits["expert_train"]])
    steps_per_epoch = train_n * cfg["patches_per_patient"]
    val_ids = splits["validation"][:args.val_cases]
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "checkpoints").exists():
        raise FileExistsError("Readiness output must not contain checkpoints")
    report = {
        "purpose": "training-readiness benchmark; not training progress",
        "weights_retained": False,
        "device": str(device),
        "validation_every_epochs": cfg.get("validation_every_epochs"),
        "expert_train_cases": train_n,
        "steps_per_epoch": steps_per_epoch,
        "val_case_ids": val_ids,
        "experts": {},
    }
    for name in SEGMENTATION_EXPERTS:
        print(f"=== {name} train probe ===", flush=True)
        train = train_probe(name, records, cfg, device, args.warmup, args.steps, args.attribution_steps)
        print(json.dumps({k: train[k] for k in ("seconds_per_step", "mean_loss", "peaks")}, indent=2), flush=True)
        print(f"=== {name} val probe ===", flush=True)
        val = val_probe(name, records, cfg, device, val_ids)
        print(json.dumps({k: val[k] for k in ("seconds_per_patient", "patches_per_patient", "projected_191_seconds", "peaks")}, indent=2), flush=True)
        checks = [1, 5, 10, 15, 20]
        report["experts"][name] = {
            "train": train,
            "validation": val,
            "projections": projections(train["seconds_per_step"], steps_per_epoch, val["projected_191_seconds"], checks),
        }
        save_json(out / "readiness.json", report)
    sequential = {}
    for label, key in (
        ("hours_5_epochs_plus_val_at_1_5", "hours_5_epochs_plus_val_at_1_5"),
        ("hours_10_epochs_plus_val_at_1_5_10", "hours_10_epochs_plus_val_at_1_5_10"),
        ("hours_20_epochs_plus_val_at_1_5_10_15_20", "hours_20_epochs_plus_val_at_1_5_10_15_20"),
    ):
        sequential[label] = sum(report["experts"][name]["projections"][key] for name in SEGMENTATION_EXPERTS)
    report["sequential_four_experts_hours"] = sequential
    report["weights_retained"] = False
    save_json(out / "readiness.json", report)
    assert not list(out.glob("**/*.pt")), "Benchmark must not retain weights"
    print(json.dumps({"sequential_four_experts_hours": sequential}, indent=2), flush=True)


if __name__ == "__main__":
    main()
