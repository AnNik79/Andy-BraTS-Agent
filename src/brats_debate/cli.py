import argparse
from pathlib import Path
import numpy as np
from .config import load_config, save_json, EXPERTS, device_for, seed_everything
from .data.brats_dataset import discover_patients, load_patient, make_splits


def main(command=None):
    parser = argparse.ArgumentParser(description="BraTS independent experts research prototype")
    parser.add_argument("--config", required=True)
    if command == "train_experts":
        parser.add_argument("--expert", choices=["all", *EXPERTS], default="all")
        parser.add_argument("--resume")
        parser.add_argument("--init-weights")
    if command == "evaluate":
        parser.add_argument("--split", choices=["validation", "test"], default="validation")
        parser.add_argument("--initial", action="store_true", help="CNN/transformer comparison before training other experts")
    if command == "run_inference":
        parser.add_argument("--patient", required=True, help="Single patient folder with configured modality suffixes")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    if command == "run_inference":
        import json
        from .inference.pipeline import load_experts, load_controller, run_patient
        split_path = Path(cfg["output_dir"]) / "splits.json"
        splits = json.loads(split_path.read_text())
        local_cfg = {**cfg, "dataset_root": str(Path(args.patient).resolve()), "manifest": None, "dataset_archive": None}
        records = discover_patients(local_cfg, require_seg=False)
        if len(records) != 1:
            raise ValueError("--patient must contain exactly one patient")
        patient = load_patient(records[0], local_cfg)
        models, hashes = load_experts(cfg, splits)
        controller, _ = load_controller(cfg, splits, hashes)
        run_patient(patient, models, controller, cfg, device_for(cfg["device"]))
        print(Path(cfg["output_dir"]) / "predictions" / patient["patient_id"])
        return
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    if command == "audit_dataset":
        rows = []
        for record in records:
            patient = load_patient(record, cfg)
            rows.append({"patient_id": patient["patient_id"], "paths": record["paths"],
                         "shape": list(patient["label"].shape), "spacing": patient["spacing"],
                         "mapped_labels": np.unique(patient["label"]).tolist()})
        save_json(Path(cfg["output_dir"]) / "dataset_audit.json", {"patients": rows, "splits": splits})
        print(f"Audited {len(rows)} aligned patients. Splits: { {k: len(v) for k,v in splits.items()} }")
    elif command == "train_experts":
        from .training.train_expert import train_expert
        names = EXPERTS if args.expert == "all" else [args.expert]
        if args.expert == "all" and (args.resume or args.init_weights):
            raise ValueError("Choose a single --expert when loading training state")
        for name in names:
            train_expert(name, records, splits, cfg, resume=args.resume, init_weights=args.init_weights)
            if args.expert == "all" and name == "transformer":
                from .evaluation.evaluate import evaluate
                evaluate(records, splits, cfg, initial=True)
    elif command == "generate_controller_data":
        from .training.train_controller import generate_controller_data
        generate_controller_data(records, splits, cfg)
    elif command == "train_controller":
        from .training.train_controller import train_controller
        train_controller(records, splits, cfg)
    elif command == "evaluate":
        from .evaluation.evaluate import evaluate
        evaluate(records, splits, cfg, args.split, args.initial)
    else:
        raise ValueError(f"Unknown command: {command}")
