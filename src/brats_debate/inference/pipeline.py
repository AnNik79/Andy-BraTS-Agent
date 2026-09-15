from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from ..config import REASONING_EXPERT, SEGMENTATION_EXPERTS, file_hash, fingerprint, protocol, save_json, device_for
from ..experts import build_expert
from ..experts.base import uncertainty
from ..debate.disagreement import analyze_disagreement
from ..debate.features import controller_features, feature_channels, stack_evidence
from ..controller.gating_network import GatingNetwork
from ..reasoning.reasoning_engine import LLMReasoningExpert, case_summary, as_volume
from ..visualization.visualize import visualize
from .patches import sliding_predict


def validate_checkpoint(checkpoint, cfg, splits, role, name=None):
    if checkpoint["protocol_hash"] != fingerprint(protocol(cfg)):
        raise ValueError("Checkpoint protocol differs from configuration; use the original model/data/inference settings")
    if checkpoint["split_hash"] != fingerprint(splits):
        raise ValueError("Checkpoint patient splits differ from current splits")
    expected = splits["expert_train" if role == "expert" else "controller_train"]
    if sorted(checkpoint["training_patients"]) != sorted(expected):
        raise ValueError("Checkpoint trained on unexpected patients; possible leakage")
    if checkpoint["role"] != role or (name is not None and checkpoint["name"] != name):
        raise ValueError("Checkpoint role/expert mismatch")


def load_experts(cfg, splits, names=None):
    names = SEGMENTATION_EXPERTS if names is None else tuple(names)
    if REASONING_EXPERT in names:
        raise ValueError("The LLM is the reasoning expert and has no voxel checkpoint; use LLMReasoningExpert")
    models, hashes = {}, {}
    for name in names:
        path = Path(cfg["output_dir"]) / "checkpoints" / name / "best.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        validate_checkpoint(checkpoint, cfg, splits, "expert", name)
        model = build_expert(name, cfg)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval().requires_grad_(False)
        models[name], hashes[name] = model, file_hash(path)
    return models, hashes


def load_controller(cfg, splits, expert_hashes):
    path = Path(cfg["output_dir"]) / "checkpoints" / "controller" / "best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint, cfg, splits, "controller")
    if checkpoint["expert_hashes"] != expert_hashes:
        raise ValueError("Experts changed since controller training; regenerate caches and retrain controller")
    model = GatingNetwork(feature_channels(cfg), cfg["controller_width"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), file_hash(path)


def predict_experts(patient, models, cfg, device):
    image = torch.from_numpy(patient["image"])[None]
    outputs = {}
    inference = cfg["inference"]
    for name, model in models.items():
        proposal = None
        if name == "highres":
            # Fusion-time coverage limiter only. HighResolutionExpert itself maps MRI →
            # probabilities with no other-expert input. Training and standalone evaluation
            # run full sliding windows. Do not treat this proposal as a scientific dependency.
            if len(outputs) != 3:
                raise ValueError("Highres proposal inference requires CNN, transformer, boundary first")
            votes = torch.stack([(o["segmentation"] > 0) for o in outputs.values()]).any(0)[:, None]
            uncertain = torch.stack([o["uncertainty"] for o in outputs.values()]).mean(0) > inference["uncertainty_threshold"]
            proposal = (votes | uncertain).float()
        model.to(device)
        outputs[name] = sliding_predict(model, image,
            cfg["highres_patch_size"] if name == "highres" else cfg["crop_size"], device,
            inference["overlap"], inference["mc_samples"], proposal,
            inference["highres_max_patches"] if name == "highres" else None)
        model.cpu()
    return outputs


@torch.no_grad()
def gate_volume(controller, image, outputs, debate, cfg, device):
    # Gate is pointwise: process slabs to avoid moving whole-volume features to VRAM.
    shape = image.shape[2:]
    final = torch.zeros((1, len(cfg["label_mapping"]), *shape))
    weights = torch.zeros((1, len(SEGMENTATION_EXPERTS), *shape))
    controller.to(device)
    for start in range(0, shape[0], 8):
        sl = (slice(start, start + 8), slice(None), slice(None))
        local = {name: {**{k: v[(..., *sl)] for k, v in out.items() if k != "extra"},
                         "extra": {k: v[(..., *sl)] for k, v in out["extra"].items()}}
                 for name, out in outputs.items()}
        local_debate = {k: v[(..., *sl)] for k, v in debate.items() if isinstance(v, torch.Tensor)}
        features = controller_features(image[(..., *sl)], local, local_debate)
        p, availability = stack_evidence(local)
        result = controller(features.to(device), p.to(device), availability.to(device))
        final[(..., *sl)] = result["probabilities"].cpu()
        weights[(..., *sl)] = result["weights"].cpu()
    controller.cpu()
    return final, weights


def save_nifti(path, array, patient, segmentation=False, cfg=None):
    if segmentation:
        decoded = np.zeros(array.shape, dtype=np.int16)
        for original, internal in cfg["label_mapping"].items():
            decoded[array == internal] = original
        array = decoded
    else:
        array = array.astype(np.float32)
    header = patient["header"].copy()
    header.set_data_dtype(array.dtype)
    image = nib.Nifti1Image(array, patient["affine"], header)
    nib.save(image, str(path))


def run_patient(patient, models, controller, cfg, device, reasoning_engine=None):
    outputs = predict_experts(patient, models, cfg, device)
    debate = analyze_disagreement(outputs)
    image = torch.from_numpy(patient["image"])[None]
    final, weights = gate_volume(controller, image, outputs, debate, cfg, device)
    predictions = {name: as_volume(out["segmentation"]) for name, out in outputs.items()}
    predictions.update(average=as_volume(debate["average_probabilities"].argmax(1)),
                       vote=as_volume(debate["majority_segmentation"]), final=as_volume(final.argmax(1)))
    summary = case_summary(patient, outputs, debate, final, weights, cfg)
    destination = Path(cfg["output_dir"]) / "predictions" / patient["patient_id"]
    destination.mkdir(parents=True, exist_ok=True)
    for name, segmentation in predictions.items():
        save_nifti(destination / f"{name}_seg.nii.gz", segmentation, patient, True, cfg)
    save_nifti(destination / "disagreement.nii.gz", as_volume(debate["map"]), patient)
    save_nifti(destination / "highres_coverage.nii.gz", as_volume(outputs["highres"]["extra"]["availability"]), patient)
    weight_arrays = {name: weights[0, i].numpy() for i, name in enumerate(SEGMENTATION_EXPERTS)}
    if cfg["inference"]["save_weights"]:
        for name, array in weight_arrays.items():
            save_nifti(destination / f"weight_{name}.nii.gz", array, patient)
    summary["highres_export_note"] = "Uncovered voxels in highres_seg are background placeholders; consult highres_coverage. Gate and baselines mask abstentions."
    engine = reasoning_engine or LLMReasoningExpert.from_config(cfg)
    result = engine.reason(summary)
    if result.can_modify_segmentation:
        raise RuntimeError("LLM reasoning must not modify the segmentation")
    summary["llm_reasoning"] = result.to_dict()
    save_json(destination / "expert_summary.json", summary)
    save_json(destination / "llm_reasoning.json", result.to_dict())
    (destination / "reasoning.txt").write_text(result.explanation)
    flair = next((name for name in ("flair", "t2f") if name in cfg["modalities"]), cfg["modalities"][0])
    modality = cfg["modalities"].index(flair)
    visualize(patient, predictions, as_volume(debate["map"]), destination / "visualization.png", modality,
              weight_arrays if cfg["inference"]["save_weights"] else None)
    return predictions, outputs, debate, final, summary
