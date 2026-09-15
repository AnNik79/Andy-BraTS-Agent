from pathlib import Path
from itertools import combinations
import json
import numpy as np
import pandas as pd
import torch
from ..config import EXPERTS, device_for, save_json, seed_everything
from ..data.brats_dataset import load_patient
from ..inference.pipeline import load_experts, load_controller, run_patient, predict_experts, save_nifti
from ..inference.patches import sliding_predict
from ..debate.disagreement import analyze_disagreement
from ..experts.base import uncertainty
from ..reasoning.reasoning_engine import as_volume
from ..visualization.visualize import visualize
from .metrics import segmentation_metrics
from .disagreement_analysis import case_error_analysis, cohort_correlation


def evaluate(records, splits, cfg, split="validation", initial=False):
    if split not in ("validation", "test"):
        raise ValueError("Evaluation is restricted to validation/test patients")
    if initial and split != "validation":
        raise ValueError("Initial exploratory comparison uses validation only")
    seed_everything(cfg["seed"])
    device = device_for(cfg["device"])
    models, hashes = load_experts(cfg, splits, EXPERTS[:2] if initial else EXPERTS)
    controller, controller_hash = (None, None) if initial else load_controller(cfg, splits, hashes)
    destination = Path(cfg["output_dir"]) / "evaluation" / ("initial_cnn_transformer" if initial else split)
    destination.mkdir(parents=True, exist_ok=True)
    selected_best = None
    if split == "test":
        validation_file = Path(cfg["output_dir"]) / "evaluation" / "validation" / "report.json"
        if not validation_file.exists():
            raise ValueError("Evaluate validation first to select the best individual expert without test leakage")
        validation = json.loads(validation_file.read_text())
        if validation["expert_hashes"] != hashes or validation["controller_hash"] != controller_hash:
            raise ValueError("Validation report is stale; reevaluate validation with the final checkpoints")
        selected_best = validation["best_individual_expert_selected_on_validation"]
    rows, analyses, diversity_rows = [], [], []
    for record in records:
        if record["patient_id"] not in splits[split]:
            continue
        patient = load_patient(record, cfg)
        if initial:
            outputs = predict_experts(patient, models, cfg, device)
            debate = analyze_disagreement(outputs)
            predictions = {name: as_volume(out["segmentation"]) for name, out in outputs.items()}
            predictions["average"] = as_volume(debate["average_probabilities"].argmax(1))
            predictions["vote"] = as_volume(debate["majority_segmentation"])
            final = debate["average_probabilities"]
            case_dir = destination / patient["patient_id"]
            case_dir.mkdir(exist_ok=True)
            save_nifti(case_dir / "disagreement.nii.gz", as_volume(debate["map"]), patient)
            for name, pred in predictions.items():
                save_nifti(case_dir / f"{name}_seg.nii.gz", pred, patient, True, cfg)
            visualize(patient, predictions, as_volume(debate["map"]), case_dir / "visualization.png",
                      next((cfg["modalities"].index(m) for m in ("flair", "t2f") if m in cfg["modalities"]), 0))
        else:
            predictions, outputs, debate, final, summary = run_patient(patient, models, controller, cfg, device)
            # Fair standalone baseline: full coverage with no other experts proposing regions.
            model = models["highres"].to(device)
            full_highres = sliding_predict(model, torch.from_numpy(patient["image"])[None], cfg["highres_patch_size"],
                                           device, cfg["inference"]["overlap"], cfg["inference"]["mc_samples"])
            model.cpu()
            predictions["highres"] = as_volume(full_highres["segmentation"])
            save_nifti(Path(cfg["output_dir"]) / "predictions" / patient["patient_id"] / "highres_full_seg.nii.gz",
                       predictions["highres"], patient, True, cfg)
        brain_mask = (patient["image"] != 0).any(0) | (patient["label"] > 0)
        for a, b in combinations(models, 2):
            ea = (predictions[a] != patient["label"]) & brain_mask
            eb = (predictions[b] != patient["label"]) & brain_mask
            union = ea | eb
            diversity_rows.append({"patient_id": patient["patient_id"], "expert_a": a, "expert_b": b,
                                   "error_overlap_jaccard": float((ea & eb).sum() / union.sum()) if union.any() else None,
                                   "only_one_expert_wrong_fraction": float((ea ^ eb).sum() / max(1, brain_mask.sum())),
                                   "both_wrong_fraction": float((ea & eb).sum() / max(1, brain_mask.sum()))})
        for method, prediction in predictions.items():
            metrics = segmentation_metrics(prediction, patient["label"], cfg["regions"], len(cfg["label_mapping"]),
                                            patient["spacing"], cfg["evaluation"]["hd95"])
            row = {"patient_id": patient["patient_id"], "method": method,
                   "mean_dice": metrics.pop("mean_dice"), "brats_mean_dice": metrics.pop("brats_mean_dice")}
            for region, values in metrics.items():
                row.update({f"{region}_{key}": value for key, value in values.items()})
            rows.append(row)
            p = outputs[method]["probabilities"] if method in outputs and method != "highres" else final
            if method == "highres":
                p = full_highres["probabilities"]
            if method in ("average", "vote"):
                p = debate["average_probabilities"]
            analysis = case_error_analysis(as_volume(debate["map"]), prediction, patient["label"],
                as_volume(uncertainty(p)["entropy"]), cfg["evaluation"]["high_disagreement_threshold"],
                brain_mask=(patient["image"] != 0).any(0) | (patient["label"] > 0))
            analyses.append({"patient_id": patient["patient_id"], "method": method, **analysis})
        print(f"Evaluated {patient['patient_id']}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(destination / "metrics.csv", index=False)
    pd.DataFrame(analyses).to_csv(destination / "disagreement_errors.csv", index=False)
    pd.DataFrame(diversity_rows).to_csv(destination / "expert_error_diversity.csv", index=False)
    aggregate = frame.groupby("method").mean(numeric_only=True)
    aggregate.to_csv(destination / "aggregate_metrics.csv")
    brats_means = aggregate["brats_mean_dice"].to_dict()
    means = aggregate["mean_dice"].to_dict()
    if selected_best is None:
        selected_best = max(models, key=lambda name: brats_means[name])
    report = {"split": split, "initial_comparison": initial, "expert_hashes": hashes,
              "controller_hash": controller_hash, "patients": len(set(frame.patient_id)),
              "mean_dice": means, "brats_mean_dice": brats_means,
              "best_individual_expert_selected_on_validation": selected_best,
              "mean_dice_definition": "Unweighted mean of configured regions; background excluded",
              "brats_mean_dice_definition": "Unweighted mean of WT, TC, and ET Dice only",
              "standalone_highres": "Full-volume native-resolution inference; proposal-limited highres used for fusion",
              "abstentions": "Average, vote, and controller exclude uncovered specialist voxels; vote ties select lowest class index",
              "disagreement_error_correlation": {method: cohort_correlation([r for r in analyses if r["method"] == method]) for method in means},
              "research_only": True}
    if not initial:
        report["controller_improvement_dice"] = {"best_individual": brats_means["final"] - brats_means[selected_best],
                                                  "average": brats_means["final"] - brats_means["average"],
                                                  "vote": brats_means["final"] - brats_means["vote"]}
    hd95_cols = [f"{region}_hd95_mm" for region in ("WT", "TC", "ET")]
    if all(col in aggregate.columns for col in hd95_cols):
        report["hd95_mm"] = {method: {region: (None if pd.isna(aggregate.loc[method][f"{region}_hd95_mm"])
                                               else float(aggregate.loc[method][f"{region}_hd95_mm"]))
                                      for region in ("WT", "TC", "ET")}
                             for method in brats_means}
    save_json(destination / "report.json", report)
    title = "Initial CNN/transformer comparison" if initial else f"{split.title()} comparison"
    lines = [f"# {title}", "", "Research experiment; no clinical validation.", "",
             "| Method | BraTS mean Dice (WT/TC/ET) | Configured mean Dice | WT | TC | ET |",
             "|---|---:|---:|---:|---:|---:|"]
    for method, value in brats_means.items():
        lines.append(
            f"| {method} | {value:.5f} | {means[method]:.5f} | "
            f"{aggregate.loc[method].get('WT_dice', float('nan')):.5f} | "
            f"{aggregate.loc[method].get('TC_dice', float('nan')):.5f} | "
            f"{aggregate.loc[method].get('ET_dice', float('nan')):.5f} |"
        )
    if "hd95_mm" in report:
        lines += ["", "| Method | WT HD95 (mm) | TC HD95 (mm) | ET HD95 (mm) |", "|---|---:|---:|---:|"]
        for method, values in report["hd95_mm"].items():
            def _fmt(value):
                return "n/a" if value is None else f"{value:.3f}"
            lines.append(f"| {method} | {_fmt(values['WT'])} | {_fmt(values['TC'])} | {_fmt(values['ET'])} |")
    lines += ["", "See metrics.csv for per-patient and region results, disagreement_errors.csv for error enrichment and AUROC.",
              "Cohort correlation treats patients as observations. Voxel statistics are descriptive, not independent-sample significance tests."]
    (destination / "report.md").write_text("\n".join(lines) + "\n")
    return report
