import argparse
from pathlib import Path
import torch
from brats_debate.data.synthetic import create_synthetic
from brats_debate.config import load_config, save_json
from brats_debate.data.brats_dataset import discover_patients, make_splits
from brats_debate.training.train_expert import train_expert
from brats_debate.training.train_controller import generate_controller_data, train_controller
from brats_debate.evaluation.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description="Synthetic-only end-to-end software check")
    parser.add_argument("--output", default="outputs/smoke")
    args = parser.parse_args()
    torch.set_num_threads(2)
    destination = Path(args.output)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Smoke output must be empty; choose a new --output path")
    template = Path(__file__).resolve().parents[1] / "configs" / "brats.yaml"
    config_path = create_synthetic(destination, template)
    cfg = load_config(config_path)
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    for name in ("cnn", "transformer"):
        train_expert(name, records, splits, cfg)
    initial = evaluate(records, splits, cfg, initial=True)
    save_json(destination / "INITIAL_SYNTHETIC_CHECK.json", {"synthetic_only": True, "research_result": False,
                                                           "status": "CNN/transformer comparison executed", "report": initial})
    for name in ("boundary", "highres"):
        train_expert(name, records, splits, cfg)
    generate_controller_data(records, splits, cfg)
    train_controller(records, splits, cfg)
    evaluate(records, splits, cfg, "validation")
    evaluate(records, splits, cfg, "test")
    save_json(destination / "SMOKE_RESULT.json", {"synthetic_only": True, "status": "passed",
                                                 "config": str(config_path), "real_brats_results": None})
    print(f"Synthetic pipeline complete: {destination}. This is not a BraTS research result.")


if __name__ == "__main__":
    main()
