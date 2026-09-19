"""Bounded Transformer continuation: epochs 7–10, optional 11–15. Hard stop at 15."""
import json
import os
import shutil
import traceback
from pathlib import Path
import faulthandler
import torch
from brats_debate.config import load_config, save_json
from brats_debate.data.brats_dataset import discover_patients, make_splits, validate_training_audit
from brats_debate.training.train_expert import train_expert

HARD_CAP = 15
STAGE1_EPOCHS = 10
EPOCH6_MEAN = 0.8576247353624641
EPOCH6_REGIONS = {"WT": 0.8886665003596469, "TC": 0.8623116169348014, "ET": 0.8218960887929426}
IMPROVE = 0.005
REGION_COLLAPSE = 0.01


def _set_yaml_epochs(config_path, epochs):
    path = Path(config_path)
    text = path.read_text()
    lines = []
    replaced = False
    for line in text.splitlines(keepends=True):
        if line.startswith("epochs:") and not replaced:
            lines.append(f"epochs: {epochs}\n" if line.endswith("\n") else f"epochs: {epochs}")
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        raise ValueError("Could not update epochs in config")
    path.write_text("".join(lines))


def _load_ckpt(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def pick_resume(destination, target_epochs):
    last = destination / "last.pt"
    progress = destination / "progress.pt"
    if not last.exists():
        if progress.exists():
            return str(progress)
        raise FileNotFoundError("No last.pt or progress.pt to resume")
    last_ck = _load_ckpt(last)
    if last_ck.get("completed_epoch", 0) >= target_epochs and last_ck.get("training_loop_complete"):
        return None
    if progress.exists():
        prog = _load_ckpt(progress)
        if (prog.get("completed_epoch", 0) < prog.get("epoch", 0)
                and prog.get("epoch", 0) <= target_epochs
                and prog.get("completed_epoch", 0) >= last_ck.get("completed_epoch", 0)):
            return str(progress)
    return str(last)


def decide_after_epoch_10(summary):
    mean = float(summary["brats_mean_dice"])
    delta = mean - EPOCH6_MEAN
    regions = {name: float(summary[f"{name}_dice"]) for name in ("WT", "TC", "ET")}
    declined = [name for name, value in regions.items() if value < EPOCH6_REGIONS[name] - REGION_COLLAPSE]
    finite = all(value == value and abs(value) != float("inf") for value in [mean, *regions.values(),
                                                                              float(summary["validation_loss"])])
    continue_run = (
        delta >= IMPROVE
        and mean >= EPOCH6_MEAN
        and len(declined) == 0
        and finite
        and int(summary.get("all_background_cases", 0)) == 0
    )
    return {
        "delta_mean": delta,
        "epoch10_mean": mean,
        "regions": regions,
        "declined_regions": declined,
        "finite": finite,
        "continue_to_15": continue_run,
        "stop_reason": None if continue_run else (
            "nonfinite" if not finite else
            "mean_worse" if mean < EPOCH6_MEAN else
            "near_plateau" if delta < IMPROVE else
            "region_collapse"
        ),
    }


def run_stage(cfg, records, splits, destination, target_epochs, config_path):
    if target_epochs > HARD_CAP:
        raise ValueError("Hard cap is epoch 15")
    _set_yaml_epochs(config_path, target_epochs)
    cfg["epochs"] = target_epochs
    resume = pick_resume(destination, target_epochs)
    if resume is None:
        return destination / "best.pt"
    return train_expert("transformer", records, splits, cfg, resume=resume)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    destination = Path(cfg["output_dir"]) / "checkpoints" / "transformer"
    faulthandler.enable()
    if cfg.get("traceback_interval_seconds"):
        faulthandler.dump_traceback_later(cfg["traceback_interval_seconds"], repeat=True)
    completed = False
    try:
        validate_training_audit(cfg)
        records = discover_patients(cfg)
        splits = make_splits(records, cfg)
        last = _load_ckpt(destination / "last.pt")
        if last.get("completed_epoch", 0) < 6:
            raise ValueError(f"Expected completed_epoch>=6, found {last.get('completed_epoch')}")
        run_stage(cfg, records, splits, destination, STAGE1_EPOCHS, args.config)
        val10 = destination / "validation_epoch_10.json"
        if not val10.exists():
            raise FileNotFoundError("epoch-10 validation artifact missing")
        epoch10 = json.loads(val10.read_text())
        decision = decide_after_epoch_10(epoch10)
        save_json(destination / "BOUNDED_DECISION.json", decision)
        print(json.dumps(decision, allow_nan=False), flush=True)
        if (destination / "best.pt").exists():
            copy = destination / "best_epoch10.pt"
            if not copy.exists():
                shutil.copy2(destination / "best.pt", copy)
        if decision["continue_to_15"]:
            run_stage(cfg, records, splits, destination, HARD_CAP, args.config)
        marker = destination / "RUN_COMPLETE"
        marker.write_text(f"{destination / 'best.pt'}\n")
        completed = True
    except BaseException as exc:
        save_json(destination / "status.json", {"state": "failed", "error": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
        for path in (destination / "history.csv", destination / "events.jsonl", destination / "status.json"):
            if path.exists():
                with path.open("a") as stream:
                    stream.flush()
                    os.fsync(stream.fileno())
    if completed:
        os._exit(0)


if __name__ == "__main__":
    main()
