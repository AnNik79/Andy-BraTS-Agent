"""Supervise a Boundary / Sum Medical training run and record terminal failures."""
import argparse
import os
from pathlib import Path
import traceback
import faulthandler
from brats_debate.config import load_config, save_json
from brats_debate.data.brats_dataset import discover_patients, make_splits, validate_training_audit
from brats_debate.training.train_expert import train_expert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--init-weights")
    args = parser.parse_args()
    cfg = load_config(args.config)
    faulthandler.enable()
    if cfg.get("traceback_interval_seconds"):
        faulthandler.dump_traceback_later(cfg["traceback_interval_seconds"], repeat=True)
    completed = False
    try:
        validate_training_audit(cfg)
        records = discover_patients(cfg)
        splits = make_splits(records, cfg)
        destination = train_expert("boundary", records, splits, cfg,
                                   resume=args.resume, init_weights=args.init_weights)
        marker = Path(destination).parent / "RUN_COMPLETE"
        marker.write_text(f"{destination}\n")
        completed = True
    except BaseException as exc:
        save_json(Path(cfg["output_dir"]) / "checkpoints" / "boundary" / "status.json",
                  {"state": "failed", "error": f"{type(exc).__name__}: {exc}"})
        traceback.print_exc()
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()
    if completed:
        os._exit(0)


if __name__ == "__main__":
    main()
