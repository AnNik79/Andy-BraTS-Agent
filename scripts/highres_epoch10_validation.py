"""Validate HighRes epoch 10 only. Does not train and does not read progress.pt."""
import csv
import hashlib
import math
import os
import shutil
from pathlib import Path
import torch
from brats_debate.config import load_config, device_for, seed_everything, EXPERTS
from brats_debate.data.brats_dataset import discover_patients, make_splits, validate_training_audit
from brats_debate.experts import build_expert
from brats_debate.training.train_expert import RunEvents, validate_expert


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(repo / "configs/brats_highres_curve5.yaml")
    if cfg.get("evaluation", {}).get("hd95"):
        raise ValueError("HD95 must stay off for this validation")
    destination = Path(cfg["output_dir"])
    if not destination.is_absolute():
        destination = (repo / "configs" / destination).resolve()
    checkpoint_dir = destination / "checkpoints" / "highres"
    epoch10 = checkpoint_dir / "epoch10.pt"
    last = checkpoint_dir / "last.pt"
    epoch5 = checkpoint_dir / "epoch5.pt"
    progress = checkpoint_dir / "progress.pt"
    if not epoch10.is_file():
        raise FileNotFoundError(epoch10)
    epoch10_hash = sha256(epoch10)
    epoch5_hash = sha256(epoch5)
    if sha256(last) != epoch10_hash:
        raise RuntimeError("last.pt and epoch10.pt differ; refusing to guess which weights to validate")
    validate_training_audit(cfg)
    if cfg.get("torch_threads"):
        torch.set_num_threads(cfg["torch_threads"])
    seed_everything(cfg["seed"] + EXPERTS.index("highres"))
    device = device_for(cfg["device"])
    if device.type != "mps":
        raise RuntimeError("HighRes validation must run on MPS")
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    validation = [record for record in records if record["patient_id"] in splits["validation"]]
    if len(validation) != 191:
        raise RuntimeError(f"Expected 191 validation cases, found {len(validation)}")
    if any(record["patient_id"] in splits["test"] for record in validation):
        raise RuntimeError("Validation list overlaps test")
    checkpoint = torch.load(epoch10, map_location="cpu", weights_only=True)
    if checkpoint.get("name") != "highres" or int(checkpoint.get("completed_epoch", -1)) != 10:
        raise RuntimeError("epoch10.pt is not the completed HighRes epoch-10 checkpoint")
    model = build_expert("highres", cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    del checkpoint
    event = RunEvents(checkpoint_dir, "highres", device, 10)
    event("epoch10 validation begin", phase="validation", epoch=10, global_step=27280,
          checkpoint=str(epoch10), cases=len(validation))
    val_loss, dice = validate_expert(model, validation, cfg, device, event, 10)
    if not math.isfinite(val_loss) or not math.isfinite(dice):
        raise FloatingPointError("Nonfinite epoch-10 validation result")
    history = checkpoint_dir / "history.csv"
    rows = list(csv.DictReader(history.open()))
    epoch_rows = [row for row in rows if int(row["epoch"]) == 10]
    if len(epoch_rows) != 1:
        raise RuntimeError("history.csv does not contain exactly one epoch-10 row")
    epoch_rows[0]["validation_loss"] = val_loss
    epoch_rows[0]["validation_dice"] = dice
    temporary = history.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(history)
    # First real HighRes validation. epoch5.pt stays untouched and unvalidated.
    best = checkpoint_dir / "best.pt"
    best_tmp = checkpoint_dir / "best.pt.tmp"
    shutil.copyfile(epoch10, best_tmp)
    best_tmp.replace(best)
    if sha256(epoch10) != epoch10_hash or sha256(epoch5) != epoch5_hash or sha256(progress) == epoch10_hash:
        raise RuntimeError("Checkpoint identity changed during validation")
    for path in (history, checkpoint_dir / "validation_epoch_10.json", checkpoint_dir / "events.jsonl", best):
        fd = os.open(path, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)
    event("epoch10 validation complete", phase="completed", state="completed", epoch=10,
          validation_loss=val_loss, validation_dice=dice)
    os._exit(0)


if __name__ == "__main__":
    main()
