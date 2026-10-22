"""Validation-set expert comparison. Does not train and does not read the test split.

Mahalanobis statistics are fit on one tumor-centered patch per expert_train case.
Disagreement, segmentation error, SVD, and novelty are scored on the 191
validation cases with full MRI-only sliding windows.
"""
import faulthandler
import gc
import json
import os
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
from scipy.stats import spearmanr
from brats_debate.analysis.correlation import error_correlation_matrix, prediction_similarity_matrix
from brats_debate.analysis.difficulty import novelty_disagreement_quadrant
from brats_debate.analysis.error_maps import segmentation_error_maps
from brats_debate.analysis.extract import representation_spec
from brats_debate.analysis.mahalanobis import ExpertMahalanobis
from brats_debate.analysis.pooling import pool_patch, pool_region
from brats_debate.analysis.svd import latent_svd
from brats_debate.config import device_for, load_config
from brats_debate.data.brats_dataset import discover_patients, load_patient, make_splits, validate_training_audit
from brats_debate.data.preprocessing import crop_pad
from brats_debate.debate.disagreement import analyze_disagreement
from brats_debate.evaluation.metrics import segmentation_metrics
from brats_debate.experts import build_expert
from brats_debate.inference.patches import sliding_probabilities

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "brats2023_gli_validation_analysis"
CHECKPOINTS = {
    "cnn": (ROOT / "outputs/brats2023_gli_cnn_curve4/checkpoints/cnn/best.pt", 15),
    "transformer": (ROOT / "outputs/brats2023_gli_transformer_curve3/checkpoints/transformer/best.pt", 6),
    "boundary": (ROOT / "outputs/brats2023_gli_boundary_curve2/checkpoints/boundary/best.pt", 16),
    "highres": (ROOT / "outputs/brats2023_gli_highres_curve5/checkpoints/highres/best.pt", 20),
}
EXPERTS = tuple(CHECKPOINTS)


def log(message):
    print(message, flush=True)


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def tumor_center(label):
    coords = np.argwhere(label > 0)
    if len(coords) == 0:
        return tuple(int(size // 2) for size in label.shape)
    return tuple(int(np.median(coords[:, axis])) for axis in range(3))


def patch_size_for(name, cfg):
    return tuple(int(value) for value in (cfg["highres_patch_size"] if name == "highres" else cfg["crop_size"]))


def cosine(left, right):
    if left is None or right is None:
        return None
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0:
        return None
    return float(np.dot(left, right) / denom)


def phi(left, right):
    left = np.asarray(left, dtype=np.float64).ravel()
    right = np.asarray(right, dtype=np.float64).ravel()
    if left.size < 2 or left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def capture(model, batch):
    spec = representation_spec(model)
    box = {}

    def hook(_module, inputs, output):
        box["tensor"] = (inputs[0] if spec.capture == "input" else output).detach()

    handle = spec.module.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            logits = model.segmentation_logits(batch)
    finally:
        handle.remove()
    if "tensor" not in box or box["tensor"].ndim != 5:
        raise RuntimeError(f"{spec.name} did not yield a 5D representation")
    return logits, box["tensor"], spec.name


def tumor_patch(model, patient, size, device):
    center = tumor_center(patient["label"])
    image, _meta = crop_pad(patient["image"], size, center)
    label, _meta = crop_pad(patient["label"].astype(np.int16), size, center)
    batch = torch.from_numpy(np.ascontiguousarray(image))[None].to(device)
    logits, spatial, name = capture(model, batch)
    spatial = spatial.float().cpu()
    vector = pool_patch(spatial).numpy()[0]
    prediction = logits.float().softmax(1).argmax(1)[0].cpu().numpy().astype(np.int16)
    del logits, batch
    if prediction.shape != label.shape:
        raise RuntimeError(f"Tumor-patch prediction shape {prediction.shape} != {label.shape}")
    maps = segmentation_error_maps(prediction, label)
    whole_tumor = np.isin(label, [1, 2, 3])

    def pooled(mask):
        if int(np.asarray(mask).sum()) == 0:
            return None
        return pool_region(spatial, mask).numpy()[0]

    correct_tumor = maps["correct"] & whole_tumor
    contrast = {
        "error_vs_correct_tumor_cosine": cosine(pooled(maps["error"]), pooled(correct_tumor)),
        "boundary_vs_correct_tumor_cosine": cosine(pooled(maps["boundary"]), pooled(correct_tumor)),
    }
    del spatial
    return vector, name, contrast


def sliding_prediction(model, image, patch, device, cfg):
    chunks = []
    spec = representation_spec(model)

    def hook(_module, inputs, output):
        tensor = (inputs[0] if spec.capture == "input" else output).detach()
        chunks.append(pool_patch(tensor).cpu())

    handle = spec.module.register_forward_hook(hook)
    try:
        probabilities = sliding_probabilities(
            model, image, patch, device, cfg["inference"]["overlap"],
            **cfg.get("validation_performance", {}))
    finally:
        handle.remove()
    if not chunks:
        raise RuntimeError(f"No windows captured for {spec.name}")
    return probabilities, torch.cat(chunks, 0).mean(0).numpy(), spec.name, len(chunks)


def load_models(cfg, device):
    models, meta = {}, {}
    for name, (path, epoch) in CHECKPOINTS.items():
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.get("name") != name or int(data.get("completed_epoch", -1)) != epoch:
            raise RuntimeError(f"{path} is not frozen {name} epoch {epoch}")
        model = build_expert(name, cfg).to(device)
        model.load_state_dict(data["state_dict"])
        model.eval().requires_grad_(False)
        models[name] = model
        meta[name] = {
            "path": str(path.relative_to(ROOT)),
            "completed_epoch": epoch,
            "validation_dice": data.get("validation_dice"),
        }
        del data
    return models, meta


def fit_mahalanobis(models, train_records, cfg, device):
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "expert_train_representations.npz"
    if cache.exists():
        saved = np.load(cache, allow_pickle=False)
        ids = saved["patient_id"].tolist()
        vectors = {name: saved[name] for name in EXPERTS}
        log(f"loaded expert_train representations {len(ids)}")
    else:
        vectors = {name: [] for name in EXPERTS}
        ids = []
        for index, record in enumerate(train_records, 1):
            patient = load_patient(record, cfg)
            ids.append(patient["patient_id"])
            for name, model in models.items():
                vector, _rep_name, _contrast = tumor_patch(model, patient, patch_size_for(name, cfg), device)
                vectors[name].append(vector)
            del patient
            if index % 25 == 0 or index == len(train_records):
                log(f"expert_train representation {index}/{len(train_records)}")
                gc.collect()
        np.savez_compressed(cache, patient_id=np.array(ids), **{name: np.stack(vectors[name]) for name in EXPERTS})
        vectors = {name: np.stack(vectors[name]) for name in EXPERTS}
    lookup = {patient_id: "expert_train" for patient_id in ids}
    fitted = {}
    for name in EXPERTS:
        fitted[name] = ExpertMahalanobis(name).fit(vectors[name], "expert_train", ids, lookup)
        log(f"mahalanobis {name} {fitted[name].summary()}")
    return fitted


def analyze_patient(patient, models, cfg, device, mahalanobis):
    label = patient["label"]
    image = torch.from_numpy(patient["image"])[None]
    probabilities, representations, tumor_vectors = {}, {}, {}
    rep_names, contrasts, windows = {}, {}, {}
    for name, model in models.items():
        probabilities[name], representations[name], rep_names[name], windows[name] = sliding_prediction(
            model, image, patch_size_for(name, cfg), device, cfg)
        tumor_vectors[name], _rep_name, contrasts[name] = tumor_patch(
            model, patient, patch_size_for(name, cfg), device)
    predictions = {name: probabilities[name][0].numpy().argmax(0).astype(np.int16) for name in EXPERTS}
    outputs = {
        name: {
            "probabilities": probabilities[name],
            "confidence": probabilities[name].amax(dim=1, keepdim=True),
            "extra": {},
        }
        for name in EXPERTS
    }
    debate = analyze_disagreement(outputs)
    js = debate["js_divergence"].detach().cpu().numpy()
    while js.ndim > 3:
        js = js[0]
    hard_disagree = np.zeros(label.shape, dtype=bool)
    for left, right in combinations(EXPERTS, 2):
        hard_disagree |= predictions[left] != predictions[right]
    whole_tumor = np.isin(label, [1, 2, 3])
    js_high = js >= float(cfg["evaluation"]["high_disagreement_threshold"])
    record = {
        "patient_id": patient["patient_id"],
        "experts": {},
        "representations": {},
        "tumor_patch_representations": {},
        "novelty": {},
        "tumor_patch_novelty": {},
        "representation_contrast": contrasts,
        "representation_names": rep_names,
        "windows": windows,
    }
    boundary = None
    for name in EXPERTS:
        metrics = segmentation_metrics(predictions[name], label, cfg["regions"], len(cfg["label_mapping"]), hd95=False)
        errors = predictions[name] != label
        if boundary is None:
            boundary = segmentation_error_maps(predictions[name], label, cfg["regions"])["boundary"]

        def rate(mask):
            selected = mask & whole_tumor
            return float(errors[selected].mean()) if selected.any() else None

        record["experts"][name] = {
            "WT_dice": metrics["WT"]["dice"],
            "TC_dice": metrics["TC"]["dice"],
            "ET_dice": metrics["ET"]["dice"],
            "brats_mean_dice": metrics["brats_mean_dice"],
            "error_fraction": float(errors.mean()),
            "wt_error_fraction": float(errors[whole_tumor].mean()) if whole_tumor.any() else None,
            "wt_error_where_experts_disagree": rate(hard_disagree),
            "wt_error_where_experts_agree": rate(~hard_disagree),
            "wt_error_where_js_high": rate(js_high),
            "wt_error_where_js_low": rate(~js_high),
            "boundary_error_fraction": float(errors[boundary].mean()) if boundary.any() else None,
            "wt_disagree_voxels": int((whole_tumor & hard_disagree).sum()),
            "wt_agree_voxels": int((whole_tumor & ~hard_disagree).sum()),
        }
        record["representations"][name] = representations[name].tolist()
        record["tumor_patch_representations"][name] = tumor_vectors[name].tolist()
        record["novelty"][name] = float(mahalanobis[name].distance(representations[name])[0])
        record["tumor_patch_novelty"][name] = float(mahalanobis[name].distance(tumor_vectors[name])[0])
    error_payload = {name: {"segmentation": predictions[name]} for name in EXPERTS}
    record["error_correlation_pairs"] = error_correlation_matrix(
        error_payload, label, EXPERTS, cfg["regions"])["pairs"]
    record["prediction_similarity_pairs"] = prediction_similarity_matrix(error_payload, EXPERTS)["pairs"]
    tumor_pairs = []
    for left, right in combinations(EXPERTS, 2):
        left_error = predictions[left] != label
        right_error = predictions[right] != label
        tumor_pairs.append({
            "pair": f"{left}__{right}",
            "wt_error_phi": phi(left_error[whole_tumor], right_error[whole_tumor]) if whole_tumor.any() else None,
            "wt_hard_disagreement_fraction": float((predictions[left] != predictions[right])[whole_tumor].mean()) if whole_tumor.any() else None,
        })
    record["wt_error_pairs"] = tumor_pairs
    record["mean_js_disagreement"] = float(js.mean())
    record["wt_mean_js_disagreement"] = float(js[whole_tumor].mean()) if whole_tumor.any() else None
    record["wt_hard_disagreement_fraction"] = float(hard_disagree[whole_tumor].mean()) if whole_tumor.any() else None
    del probabilities, outputs, debate, image, predictions
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return record


def mean(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    return None if not values else float(np.mean(values))


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return None
    coef = spearmanr(x[valid], y[valid]).statistic
    return None if not np.isfinite(coef) else float(coef)


def compact_svd(matrix, **kwargs):
    result = latent_svd(matrix, n_components=5, **kwargs)
    energy = result["explained_energy"]
    return {
        "kind": result["kind"],
        "n_rows": result["n_rows"],
        "n_columns": result["n_columns"],
        "explained_energy_top5": energy[:5],
        "cumulative_energy_top5": result["cumulative_energy"][:5],
        "normalized_singular_values_top5": result["normalized_singular_values"][:5],
    }


def aggregate(rows, mahalanobis):
    summary = {"n_validation_cases": len(rows), "experts": {}}
    for name in EXPERTS:
        summary["experts"][name] = {
            "WT_dice": mean(row["experts"][name]["WT_dice"] for row in rows),
            "TC_dice": mean(row["experts"][name]["TC_dice"] for row in rows),
            "ET_dice": mean(row["experts"][name]["ET_dice"] for row in rows),
            "brats_mean_dice": mean(row["experts"][name]["brats_mean_dice"] for row in rows),
            "wt_error_fraction": mean(row["experts"][name]["wt_error_fraction"] for row in rows),
            "wt_error_where_experts_disagree": mean(row["experts"][name]["wt_error_where_experts_disagree"] for row in rows),
            "wt_error_where_experts_agree": mean(row["experts"][name]["wt_error_where_experts_agree"] for row in rows),
            "wt_error_where_js_high": mean(row["experts"][name]["wt_error_where_js_high"] for row in rows),
            "wt_error_where_js_low": mean(row["experts"][name]["wt_error_where_js_low"] for row in rows),
            "boundary_error_fraction": mean(row["experts"][name]["boundary_error_fraction"] for row in rows),
            "error_vs_correct_tumor_cosine": mean(row["representation_contrast"][name]["error_vs_correct_tumor_cosine"] for row in rows),
            "boundary_vs_correct_tumor_cosine": mean(row["representation_contrast"][name]["boundary_vs_correct_tumor_cosine"] for row in rows),
            "tumor_patch_novelty_vs_wt_error_spearman": spearman(
                [row["tumor_patch_novelty"][name] for row in rows],
                [row["experts"][name]["wt_error_fraction"] for row in rows]),
            "volume_novelty_vs_wt_error_spearman": spearman(
                [row["novelty"][name] for row in rows],
                [row["experts"][name]["wt_error_fraction"] for row in rows]),
            "tumor_patch_novelty_vs_one_minus_dice_spearman": spearman(
                [row["tumor_patch_novelty"][name] for row in rows],
                [1.0 - row["experts"][name]["brats_mean_dice"] for row in rows]),
            "mahalanobis": mahalanobis[name].summary(),
            "validation_tumor_patch_svd": compact_svd(
                np.stack([row["tumor_patch_representations"][name] for row in rows]), row_kind="validation_patient"),
            "validation_volume_svd": compact_svd(
                np.stack([row["representations"][name] for row in rows]), row_kind="validation_patient"),
        }
    summary["disagreement_vs_mean_wt_error_spearman"] = spearman(
        [row["wt_mean_js_disagreement"] for row in rows],
        [mean(row["experts"][name]["wt_error_fraction"] for name in EXPERTS) for row in rows])
    summary["hard_disagreement_vs_mean_wt_error_spearman"] = spearman(
        [row["wt_hard_disagreement_fraction"] for row in rows],
        [mean(row["experts"][name]["wt_error_fraction"] for name in EXPERTS) for row in rows])

    def average_pairs(key):
        grouped = {}
        for row in rows:
            for pair in row[key]:
                bucket = grouped.setdefault(pair["pair"], {})
                for field, value in pair.items():
                    if field in ("pair", "kind") or value is None:
                        continue
                    bucket.setdefault(field, []).append(value)
        return {pair: {field: mean(values) for field, values in fields.items()} for pair, fields in grouped.items()}

    summary["error_correlation"] = average_pairs("error_correlation_pairs")
    summary["prediction_similarity"] = average_pairs("prediction_similarity_pairs")
    summary["wt_error_correlation"] = average_pairs("wt_error_pairs")
    disagreement = np.array([row["wt_mean_js_disagreement"] for row in rows], dtype=np.float64)
    novelty = np.array([mean(row["tumor_patch_novelty"][name] for name in EXPERTS) for row in rows], dtype=np.float64)
    error = np.array([mean(row["experts"][name]["wt_error_fraction"] for name in EXPERTS) for row in rows], dtype=np.float64)
    quadrants = novelty_disagreement_quadrant(
        disagreement, novelty, float(np.nanmedian(disagreement)), float(np.nanmedian(novelty)))
    labels = quadrants["quadrant"]
    summary["quadrants"] = {
        "thresholds": quadrants["thresholds"],
        "counts": quadrants["counts"],
        "mean_wt_error": {name: mean(error[labels == name]) for name in quadrants["counts"]},
        "note": quadrants["note"],
    }
    difficulty = np.stack([[1.0 - row["experts"][name]["brats_mean_dice"] for name in EXPERTS] for row in rows])
    shared = latent_svd(difficulty, n_components=4, column_names=list(EXPERTS), row_kind="validation_patient")
    summary["shared_difficulty_svd"] = {
        "explained_energy": shared["explained_energy"],
        "cumulative_energy": shared["cumulative_energy"],
        "loadings": shared["loadings"],
    }
    return summary


def load_done(path):
    done = {}
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        done[row["patient_id"]] = row
    return done


def main():
    faulthandler.dump_traceback_later(600, repeat=True)
    cfg = load_config(ROOT / "configs" / "brats_highres_curve5.yaml")
    if cfg["evaluation"].get("hd95"):
        raise RuntimeError("HD95 must stay off")
    validate_training_audit(cfg)
    device = device_for(cfg["device"])
    if device.type != "mps":
        raise RuntimeError("Analysis must run on MPS")
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    by_id = {record["patient_id"]: record for record in records}
    test_ids = set(splits["test"])
    train_records = [by_id[patient_id] for patient_id in splits["expert_train"]]
    validation_records = [by_id[patient_id] for patient_id in splits["validation"]]
    if set(splits["expert_train"]) & test_ids or set(splits["validation"]) & test_ids:
        raise RuntimeError("Test cases leaked into analysis splits")
    if len(validation_records) != 191 or len(train_records) != 682:
        raise RuntimeError(f"Unexpected split sizes train={len(train_records)} val={len(validation_records)}")
    log(f"splits expert_train={len(train_records)} validation={len(validation_records)} test_unused={len(test_ids)}")
    models, meta = load_models(cfg, device)
    mahalanobis = fit_mahalanobis(models, train_records, cfg, device)
    OUT.mkdir(parents=True, exist_ok=True)
    patient_path = OUT / "validation_patients.jsonl"
    done = load_done(patient_path)
    with patient_path.open("a") as stream:
        for record in validation_records:
            if record["patient_id"] in done:
                continue
            if record["patient_id"] in test_ids:
                raise RuntimeError("Refusing a test patient")
            patient = load_patient(record, cfg)
            row = analyze_patient(patient, models, cfg, device, mahalanobis)
            stream.write(json.dumps(jsonable(row)) + "\n")
            stream.flush()
            done[row["patient_id"]] = row
            dice = {name: round(row["experts"][name]["brats_mean_dice"], 3) for name in EXPERTS}
            log(f"validation {len(done)}/{len(validation_records)} {row['patient_id']} dice={dice}")
            del patient, row
    rows = [done[record["patient_id"]] for record in validation_records]
    summary = aggregate(rows, mahalanobis)
    summary["checkpoints"] = meta
    summary["test_cases_used"] = 0
    summary["controller_train_cases_used"] = 0
    summary["mahalanobis_fit_split"] = "expert_train"
    summary["mahalanobis_fit_definition"] = "one deterministic tumor-centered patch per expert_train case; PCA then Ledoit-Wolf"
    summary["validation_probability_definition"] = "full MRI-only sliding windows; highres proposal gating was not used"
    summary["disagreement_definition"] = "expert disagreement among the four experts; not ground-truth error"
    summary["svd_definition"] = "exploratory latent structure, not biological labels"
    (OUT / "summary.json").write_text(json.dumps(jsonable(summary), indent=2) + "\n")
    log("VALIDATION_ANALYSIS_COMPLETE")
    faulthandler.cancel_dump_traceback_later()
    os._exit(0)


if __name__ == "__main__":
    main()
