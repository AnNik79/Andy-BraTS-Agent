"""Create the one-case Beta 1 professor demo from frozen checkpoints.

Runs BraTS-GLI-00006-000 only. Saves proposal-limited expert and controller
masks, then a separate full-volume HighRes mask. Does not train or read the
historical test split.
"""

import json
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import nibabel as nib
import numpy as np
from PIL import Image
from scipy import ndimage
import torch

from brats_debate.config import device_for, file_hash, load_config, save_json, seed_everything
from brats_debate.data.brats_dataset import discover_patients, load_patient
from brats_debate.inference.patches import sliding_predict
from brats_debate.inference.pipeline import (
    load_controller,
    load_experts,
    run_patient,
    save_nifti,
)
from brats_debate.reasoning.reasoning_engine import as_volume


ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "BraTS-GLI-00006-000"
DEMO_ROOT = ROOT / "outputs" / "beta1_professor_demo"
CASE_DIR = DEMO_ROOT / "predictions" / CASE_ID
WEB_DIR = DEMO_ROOT / "web"
EXPECTED_EXPERT_HASHES = {
    "cnn": "e8fbde5084353dc5aa499626d208fad7c7c4595e23376daa7d98a4890259697c",
    "transformer": "dd77255376fb0831e7b91795dff3dbf1bba647b3e305df9ef5dfa21d6d55b64b",
    "boundary": "dd360d8d6430acd4eff116b181ed3b83b2144fffd16c0bf8c0baaa913f00b79c",
    "highres": "15bd9b270ff3ca160e7a4f07a93809e1e5ad9c7e25b352966df5940d639b83c0",
}
EXPECTED_CONTROLLER_HASH = "86b604082a934a1d4f4a68467a88d16d268ae7653c552dcfc6c3087ba1d79fe2"
COLORS = {
    1: np.array([230, 97, 1], dtype=np.float32),
    2: np.array([94, 172, 67], dtype=np.float32),
    3: np.array([213, 40, 155], dtype=np.float32),
}
PANEL_ORDER = (
    ("mri", "T2-FLAIR"),
    ("cnn", "CNN"),
    ("transformer", "Transformer"),
    ("boundary", "Boundary"),
    ("highres", "HighRes, proposal-limited"),
    ("final", "Controller"),
    ("ground_truth", "Ground truth"),
    ("highres_full", "HighRes, full volume"),
)
TEST_METRICS = {
    "source": "Frozen Beta 1 held-out test aggregate. Not this patient's score.",
    "cases": 130,
    "mean_dice": {
        "cnn": 0.86844,
        "transformer": 0.86602,
        "boundary": 0.86344,
        "highres_full_volume": 0.81344,
        "controller": 0.87344,
        "majority_vote": 0.87312,
    },
}


def display_plane(volume, index):
    """Match the project visualizer: transposed native plane, origin lower."""
    return np.flipud(np.swapaxes(np.asarray(volume)[:, :, index], 0, 1))


def window_uint8(image):
    foreground = image[image != 0]
    low, high = np.percentile(foreground, [1, 99])
    scaled = np.clip((image - low) / max(float(high - low), 1e-6), 0, 1)
    return (scaled * 255).astype(np.uint8)


def label_rgba(plane, reference=None):
    rgba = np.zeros((*plane.shape, 4), dtype=np.uint8)
    for value, color in COLORS.items():
        selected = plane == value
        rgba[selected, :3] = color
        rgba[selected, 3] = 255
    if reference is not None:
        differ = plane != reference
        if differ.any():
            thick = ndimage.binary_dilation(differ, iterations=2)
            rgba[thick, 0] = 0
            rgba[thick, 1] = 229
            rgba[thick, 2] = 255
            rgba[thick, 3] = 255
    return rgba


def composite(mri, labels, alpha=0.58):
    rgb = np.repeat(mri[..., None], 3, axis=-1).astype(np.float32)
    for value, color in COLORS.items():
        mask = labels == value
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * color
    return rgb.astype(np.uint8)


def checkpoint_paths(cfg):
    paths = {name: Path(cfg["expert_checkpoints"][name]) for name in EXPECTED_EXPERT_HASHES}
    paths["controller"] = Path(cfg["output_dir"]) / "checkpoints" / "controller" / "best.pt"
    return paths


def verify_checkpoint_hashes(paths):
    hashes = {name: file_hash(path) for name, path in paths.items()}
    experts = {name: hashes[name] for name in EXPECTED_EXPERT_HASHES}
    if experts != EXPECTED_EXPERT_HASHES or hashes["controller"] != EXPECTED_CONTROLLER_HASH:
        raise RuntimeError(f"Frozen checkpoint hash mismatch: {hashes}")
    return hashes


def selected_record(cfg):
    splits = json.loads(Path(cfg["split_file"]).read_text())
    if CASE_ID not in splits["validation"] or CASE_ID in splits["test"]:
        raise RuntimeError(f"{CASE_ID} is not an eligible validation case")
    if splits["validation"][0] != CASE_ID:
        raise RuntimeError("The frozen validation order no longer starts with the approved case")
    records = [record for record in discover_patients(cfg) if record["patient_id"] == CASE_ID]
    if len(records) != 1:
        raise RuntimeError(f"Expected one local MRI record for {CASE_ID}")
    return records[0]


def load_saved_segmentation(path):
    import nibabel as nib
    array = np.rint(nib.load(str(path)).get_fdata(dtype=np.float32)).astype(np.uint8)
    if not np.isin(array, [0, 1, 2, 3]).all():
        raise RuntimeError(f"Unexpected labels in {path}")
    return array


def run_case(cfg, paths):
    if shutil.disk_usage(ROOT).free < 15 * 2**30:
        raise RuntimeError("Internal free space is below 15 GiB")
    before = verify_checkpoint_hashes(paths)
    splits = json.loads(Path(cfg["split_file"]).read_text())
    seed_everything(cfg["seed"])
    device = device_for(cfg["device"])
    if device.type != "mps" or not torch.backends.mps.is_available():
        raise RuntimeError("This approved demo inference requires MPS")
    load_cfg = {**cfg, "patient_cache_dir": None}
    patient = load_patient(selected_record(cfg), load_cfg)
    if patient["patient_id"] != CASE_ID or patient["label"] is None:
        raise RuntimeError("Loaded patient does not match the approved validation case")
    print(f"Loaded {CASE_ID} shape={patient['image'].shape[1:]}", flush=True)
    models, expert_hashes = load_experts(cfg, splits)
    controller, controller_hash = load_controller(cfg, splits, expert_hashes)
    if expert_hashes != EXPECTED_EXPERT_HASHES or controller_hash != EXPECTED_CONTROLLER_HASH:
        raise RuntimeError("Loaded checkpoint hashes differ from the approved frozen hashes")
    demo_cfg = {**cfg, "output_dir": str(DEMO_ROOT), "patient_cache_dir": None}
    if CASE_DIR.exists():
        raise RuntimeError("Demo prediction directory already exists; refusing to overwrite it")
    started = time.perf_counter()
    with torch.inference_mode():
        predictions, outputs, debate, _final, _summary = run_patient(
            patient, models, controller, demo_cfg, device
        )
        fusion_seconds = time.perf_counter() - started
        print(f"Proposal-limited fusion finished in {fusion_seconds:.1f}s", flush=True)
        highres_model = models["highres"].to(device)
        image = torch.from_numpy(patient["image"])[None]
        full_started = time.perf_counter()
        full = sliding_predict(
            highres_model,
            image,
            cfg["highres_patch_size"],
            device,
            cfg["inference"]["overlap"],
            cfg["inference"]["mc_samples"],
        )
        highres_model.cpu()
        full_seconds = time.perf_counter() - full_started
    full_segmentation = as_volume(full["segmentation"]).astype(np.uint8)
    save_nifti(CASE_DIR / "highres_full_seg.nii.gz", full_segmentation, patient, True, cfg)
    limited = predictions["highres"].astype(np.uint8)
    if limited.shape != full_segmentation.shape or limited.shape != patient["label"].shape:
        raise RuntimeError("Prediction, full HighRes, and ground-truth shapes differ")
    coverage = float((outputs["highres"]["extra"]["availability"] > 0).float().mean())
    changed = int((limited != full_segmentation).sum())
    after = verify_checkpoint_hashes(paths)
    if before != after:
        raise RuntimeError("Checkpoint files changed during demo inference")
    provenance = {
        "format": "beta1_professor_demo_case_v1",
        "patient_id": CASE_ID,
        "source_split": "validation",
        "test_data_accessed": False,
        "precomputed": True,
        "research_only": True,
        "expert_hashes": expert_hashes,
        "controller_hash": controller_hash,
        "checkpoint_hashes_unchanged": True,
        "inference": {
            "overlap": cfg["inference"]["overlap"],
            "mc_samples": cfg["inference"]["mc_samples"],
            "uncertainty_threshold": cfg["inference"]["uncertainty_threshold"],
            "highres_max_patches": cfg["inference"]["highres_max_patches"],
        },
        "highres_proposal_limited": {
            "file": "highres_seg.nii.gz",
            "coverage": coverage,
            "meaning": "Mask actually available to the frozen controller. Uncovered voxels are background placeholders.",
        },
        "highres_full_volume": {
            "file": "highres_full_seg.nii.gz",
            "voxels_different_from_proposal_limited": changed,
            "meaning": "Separate full sliding-window HighRes pass. The published 0.81344 test Dice describes this protocol, not the proposal-limited mask.",
        },
        "fusion_seconds": fusion_seconds,
        "full_highres_seconds": full_seconds,
        "total_seconds": time.perf_counter() - started,
        "shape": list(patient["label"].shape),
        "spacing_mm": patient["spacing"],
    }
    save_json(CASE_DIR / "demo_provenance.json", provenance)
    arrays = {
        "mri": {name: patient["image"][index] for index, name in enumerate(cfg["modalities"])},
        "labels": {
            "cnn": predictions["cnn"].astype(np.uint8),
            "transformer": predictions["transformer"].astype(np.uint8),
            "boundary": predictions["boundary"].astype(np.uint8),
            "highres": limited,
            "final": predictions["final"].astype(np.uint8),
            "vote": predictions["vote"].astype(np.uint8),
            "ground_truth": patient["label"].astype(np.uint8),
            "highres_full": full_segmentation,
        },
        "disagreement": as_volume(debate["map"]).astype(np.float32),
    }
    for name, array in arrays["labels"].items():
        saved_name = {
            "highres": "highres_seg.nii.gz",
            "highres_full": "highres_full_seg.nii.gz",
            "ground_truth": None,
        }.get(name, f"{name}_seg.nii.gz")
        if saved_name is None:
            continue
        reloaded = load_saved_segmentation(CASE_DIR / saved_name)
        if not np.array_equal(reloaded, array):
            raise RuntimeError(f"Saved segmentation does not match memory for {name}")
    print(json.dumps({"coverage": coverage, "highres_voxels_changed": changed, "seconds": provenance["total_seconds"]}), flush=True)
    return arrays, provenance


def render_assets(cfg, arrays, provenance):
    modalities = {name: window_uint8(image) for name, image in arrays["mri"].items()}
    labels = arrays["labels"]
    reference = labels["ground_truth"]
    slice_index = int((reference > 0).sum(axis=(0, 1)).argmax())
    flair = modalities["t2f"]
    panels = []
    for key, title in PANEL_ORDER:
        plane = display_plane(flair, slice_index)
        if key == "mri":
            image = np.repeat(plane[..., None], 3, axis=-1)
        else:
            image = composite(plane, display_plane(labels[key], slice_index))
        panels.append((title, image))
    figure, axes = plt.subplots(2, 4, figsize=(16, 8.6), dpi=200)
    for ax, (title, image) in zip(axes.flat, panels):
        ax.imshow(image, interpolation="nearest")
        ax.set_title(title, fontsize=11, pad=6)
        ax.axis("off")
    figure.legend(
        handles=[
            Patch(facecolor=COLORS[3] / 255, label="Enhancing tumor (ET, label 3)"),
            Patch(facecolor=COLORS[1] / 255, label="Non-enhancing core (label 1; tumor core = 1+3)"),
            Patch(facecolor=COLORS[2] / 255, label="Edema (label 2; whole tumor = 1+2+3)"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    figure.suptitle(
        f"CancerAgents Beta 1 — {CASE_ID} — T2-FLAIR, native axis 2, slice {slice_index}",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.035,
        "Validation case selected as the first frozen validation ID. Research use only; not a diagnosis.\n"
        "The controller used proposal-limited HighRes. The separate full-volume HighRes panel is the protocol behind the published 0.81344 test Dice.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    figure_path = CASE_DIR / "comparison.png"
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)
    shutil.copyfile(figure_path, DEMO_ROOT / "comparison.png")

    if WEB_DIR.exists():
        shutil.rmtree(WEB_DIR)
    for name, volume in modalities.items():
        destination = WEB_DIR / "slices" / "mri" / name
        destination.mkdir(parents=True)
        for index in range(volume.shape[2]):
            Image.fromarray(display_plane(volume, index), mode="L").save(destination / f"{index:03d}.png")
    reference_volume = labels["ground_truth"]
    slice_differences = {}
    for name, volume in labels.items():
        destination = WEB_DIR / "slices" / "seg" / name
        destination.mkdir(parents=True)
        counts = []
        for index in range(volume.shape[2]):
            plane = display_plane(volume, index)
            reference = None if name == "ground_truth" else display_plane(reference_volume, index)
            if reference is not None:
                counts.append(int((plane != reference).sum()))
            Image.fromarray(label_rgba(plane, reference), mode="RGBA").save(destination / f"{index:03d}.png")
        if counts:
            slice_differences[name] = counts
    disagreement = arrays["disagreement"]
    destination = WEB_DIR / "slices" / "seg" / "disagreement"
    destination.mkdir(parents=True)
    peak = float(disagreement.max())
    for index in range(disagreement.shape[2]):
        plane = display_plane(disagreement, index) / max(peak, 1e-6)
        rgba = np.zeros((*plane.shape, 4), dtype=np.uint8)
        active = plane > 0
        rgba[active, 0] = 255
        rgba[active, 1] = (255 * (1 - plane[active])).astype(np.uint8)
        rgba[active, 3] = (180 * plane[active]).astype(np.uint8)
        Image.fromarray(rgba, mode="RGBA").save(destination / f"{index:03d}.png")
    shape = labels["ground_truth"].shape
    metadata = {
        "patient_id": CASE_ID,
        "slice_count": int(shape[2]),
        "default_slice": slice_index,
        "display": "Native axis 2, transposed and vertically flipped to match origin-lower project figures.",
        "modalities": [
            {"id": "t1n", "label": "T1n"},
            {"id": "t1c", "label": "T1c"},
            {"id": "t2w", "label": "T2w"},
            {"id": "t2f", "label": "T2-FLAIR"},
        ],
        "overlays": [
            {"id": "cnn", "label": "CNN"},
            {"id": "transformer", "label": "Transformer"},
            {"id": "boundary", "label": "Boundary"},
            {"id": "highres", "label": "HighRes, proposal-limited"},
            {"id": "final", "label": "Controller"},
            {"id": "vote", "label": "Majority vote"},
            {"id": "ground_truth", "label": "Ground truth"},
            {"id": "highres_full", "label": "HighRes, full volume"},
            {"id": "disagreement", "label": "Expert disagreement"},
        ],
        "test_metrics": TEST_METRICS,
        "highres_note": provenance["highres_full_volume"]["meaning"],
        "proposal_coverage": provenance["highres_proposal_limited"]["coverage"],
        "highres_voxels_different": provenance["highres_full_volume"]["voxels_different_from_proposal_limited"],
        "slice_differences": slice_differences,
        "asset_version": 3,
        "provenance": provenance,
        "research_only": True,
        "precomputed": True,
    }
    save_json(WEB_DIR / "meta.json", metadata)
    print(f"Figure: {figure_path}", flush=True)


def load_rendered_case(cfg):
    provenance = json.loads((CASE_DIR / "demo_provenance.json").read_text())
    patient = load_patient(selected_record({**cfg, "patient_cache_dir": None}), {**cfg, "patient_cache_dir": None})
    labels = {"ground_truth": patient["label"].astype(np.uint8)}
    for name in ("cnn", "transformer", "boundary", "highres", "final", "vote", "highres_full"):
        filename = "highres_full_seg.nii.gz" if name == "highres_full" else f"{name}_seg.nii.gz" if name != "highres" else "highres_seg.nii.gz"
        labels[name] = load_saved_segmentation(CASE_DIR / filename)
    disagreement = nib.load(str(CASE_DIR / "disagreement.nii.gz")).get_fdata(dtype=np.float32)
    return {
        "mri": {name: patient["image"][index] for index, name in enumerate(cfg["modalities"])},
        "labels": labels,
        "disagreement": disagreement,
    }, provenance


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(ROOT / "configs" / "brats_controller.yaml")
    if args.render_only:
        arrays, provenance = load_rendered_case(cfg)
        render_assets(cfg, arrays, provenance)
        print(json.dumps({"status": "rendered"}, indent=2), flush=True)
        return
    if cfg["reasoning"]["mode"] != "deterministic":
        raise RuntimeError("Demo inference must keep deterministic local reasoning")
    paths = checkpoint_paths(cfg)
    arrays, provenance = run_case(cfg, paths)
    render_assets(cfg, arrays, provenance)
    print(json.dumps({"status": "complete", "seconds": provenance["total_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
