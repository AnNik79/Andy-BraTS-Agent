"""Launch only the audited CNN run, with a durable log and process ID."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from brats_debate.config import load_config, save_json
from brats_debate.data.brats_dataset import validate_training_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--init-weights")
    args = parser.parse_args()
    config = Path(args.config).resolve()
    cfg = load_config(config)
    validate_training_audit(cfg)
    destination = Path(cfg["output_dir"]) / "checkpoints" / "cnn"
    destination.mkdir(parents=True, exist_ok=True)
    launch = destination / "launch.json"
    if launch.exists():
        old = json.loads(launch.read_text())
        try:
            os.kill(old["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f"Previous CNN process still exists: {old['pid']}")
    repo = Path(__file__).resolve().parents[1]
    command = [sys.executable, "-u", str(repo / "scripts" / "cnn_training_job.py"), "--config", str(config)]
    for option, value in (("--resume", args.resume), ("--init-weights", args.init_weights)):
        if value:
            command.extend([option, str(Path(value).resolve())])
    env = dict(os.environ)
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    log_path = destination / "training.log"
    with open(log_path, "a") as log:
        process = subprocess.Popen(command, cwd=repo, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    save_json(launch, {"pid": process.pid, "command": command, "log": str(log_path),
                       "started_utc": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"pid": process.pid, "log": str(log_path)}, indent=2))


if __name__ == "__main__":
    main()
