"""Detached, atomic training snapshots without moving the live model off its device."""
from pathlib import Path
import random
import numpy as np
import torch
from ..config import fingerprint, protocol


def cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(cpu_snapshot(v) for v in value)
    return value


def rng_state(device):
    state = np.random.get_state()
    result = {"python": random.getstate(), "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
              "torch": torch.get_rng_state()}
    if device.type == "cuda":
        result["cuda"] = torch.cuda.get_rng_state_all()
    if device.type == "mps":
        result["mps"] = torch.mps.get_rng_state().clone()
    return result


def restore_rng(state, device):
    random.setstate(state["python"])
    algorithm, keys, position, has_gauss, cached = state["numpy"]
    np.random.set_state((algorithm, np.array(keys, dtype=np.uint32), position, has_gauss, cached))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda" and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
    if device.type == "mps" and "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def training_signature(cfg):
    # Epoch target can be extended; changing data flow or debug limits is not an exact resume.
    keys = ("batch_size", "patches_per_patient", "gradient_accumulation", "seed", "learning_rate", "num_workers")
    return fingerprint({**{k: cfg[k] for k in keys}, "protocol": protocol(cfg),
                        "max_train_steps": cfg.get("max_train_steps"),
                        "validation_case_ids": cfg.get("validation_case_ids"),
                        "validation_max_cases": cfg.get("validation_max_cases"),
                        "validation_every_epochs": cfg.get("validation_every_epochs", 1),
                        "validate_first_epoch": cfg.get("validate_first_epoch", True),
                        "early_stopping_patience": cfg.get("early_stopping_patience", 0),
                        "dataset_archive": cfg.get("dataset_archive"), "dataset_root": cfg["dataset_root"]})


def save_training_checkpoint(path, model, optimizer, scaler, cfg, splits, name, seed,
                             cursor, best, device, event, scheduler=None, validation_dice=None):
    path = Path(path)
    event("checkpoint save begin", path=str(path), epoch=cursor["epoch"], global_step=cursor["global_step"])
    event("checkpoint model/optimizer CPU snapshot begin", path=str(path))
    checkpoint = {"format_version": 2, "state_dict": cpu_snapshot(model.state_dict()),
                  "optimizer_state_dict": cpu_snapshot(optimizer.state_dict()),
                  "scheduler_state_dict": cpu_snapshot(scheduler.state_dict()) if scheduler else None,
                  "scaler_state_dict": cpu_snapshot(scaler.state_dict()),
                  "role": "expert", "name": name, "seed": seed,
                  "protocol_hash": fingerprint(protocol(cfg)), "split_hash": fingerprint(splits),
                  "training_signature": training_signature(cfg), "training_patients": splits["expert_train"],
                  "best_validation_metric": float(best), "validation_dice": validation_dice,
                  "architecture": str(model), "device_type": device.type, **cursor}
    event("checkpoint model/optimizer CPU snapshot complete", path=str(path))
    event("checkpoint RNG snapshot begin", path=str(path))
    checkpoint["rng_state"] = rng_state(device)
    event("checkpoint RNG snapshot complete", path=str(path))
    event("checkpoint serialization begin", path=str(path))
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)
    event("checkpoint serialization complete", path=str(path))
    event("checkpoint save complete", path=str(path))


def load_training_checkpoint(path, model, optimizer, scaler, cfg, splits, name, device, scheduler=None):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format_version") != 2 or "optimizer_state_dict" not in data:
        raise ValueError("Legacy weights-only checkpoint cannot resume optimizer/RNG/epoch state; use --init-weights for an explicitly new run")
    if data["name"] != name or data["protocol_hash"] != fingerprint(protocol(cfg)) or data["split_hash"] != fingerprint(splits):
        raise ValueError("Checkpoint expert, protocol, or patient split mismatch")
    if data["training_signature"] != training_signature(cfg):
        raise ValueError("Training data flow/debug limits changed; cannot resume this checkpoint exactly")
    if data["next_batch"] and not data["training_loop_complete"] and cfg["num_workers"] != 0:
        raise ValueError("Exact mid-epoch resume requires num_workers=0; worker augmentation RNG is not captured")
    if (scheduler is None) != (data["scheduler_state_dict"] is None):
        raise ValueError("Scheduler mismatch")
    model.load_state_dict(data["state_dict"])
    optimizer.load_state_dict(data["optimizer_state_dict"])
    scaler.load_state_dict(data["scaler_state_dict"])
    if scheduler:
        scheduler.load_state_dict(data["scheduler_state_dict"])
    # DataLoader creation/iteration uses a separate per-epoch generator, so it cannot alter these RNGs.
    restore_rng(data["rng_state"], device)
    return data
