"""Finish Boundary epochs 11–12, then start a fresh HighRes run. Never overlap MPS training."""
import os
import subprocess
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = ROOT / "outputs/brats2023_gli_boundary_curve2/checkpoints/boundary"
ENV = {**os.environ, "PYTHONPATH": str(ROOT / "src")}


def run(args):
    print("STAGE_BEGIN", args, flush=True)
    completed = subprocess.run([sys.executable, "-u", *args], cwd=ROOT, env=ENV)
    print("STAGE_EXIT", args[0], completed.returncode, flush=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def boundary_ready():
    val = BOUNDARY / "validation_epoch_12.json"
    last = torch.load(BOUNDARY / "last.pt", map_location="cpu", weights_only=True)
    best = torch.load(BOUNDARY / "best.pt", map_location="cpu", weights_only=True)
    if last.get("completed_epoch") != 12 or not val.exists():
        raise SystemExit(f"Boundary epoch-12 validation missing: completed={last.get('completed_epoch')} val={val.exists()}")
    if best.get("completed_epoch") not in (10, 12):
        raise SystemExit(f"Unexpected Boundary best epoch {best.get('completed_epoch')}")
    model_keys = len(best["state_dict"])
    if model_keys < 1:
        raise SystemExit("Boundary best.pt did not load weights")
    for path in (BOUNDARY / "history.csv", BOUNDARY / "last.pt", BOUNDARY / "best.pt", val):
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
    print("BOUNDARY_READY", "completed_epoch", last["completed_epoch"],
          "best_epoch", best["completed_epoch"], "best_dice", best.get("validation_dice"),
          "weights", model_keys, flush=True)


def main():
    run(["scripts/boundary_training_job.py", "--config", "configs/brats_boundary_curve2.yaml",
         "--resume", "outputs/brats2023_gli_boundary_curve2/checkpoints/boundary/last.pt"])
    boundary_ready()
    run(["scripts/highres_training_job.py", "--config", "configs/brats_highres_curve5.yaml"])


if __name__ == "__main__":
    main()
