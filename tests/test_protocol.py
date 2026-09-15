from pathlib import Path
import torch
import pytest
from brats_debate.config import EXPERTS, SEGMENTATION_EXPERTS, SYSTEM_EXPERTS, REASONING_EXPERT
from brats_debate.data.brats_dataset import discover_patients, load_patient
from brats_debate.controller.controller_dataset import load_cache
from brats_debate.controller.gating_network import GatingNetwork
from brats_debate.debate.features import feature_channels
from brats_debate.experts import build_expert


def test_modern_aliases_and_manifest(cfg, tmp_path):
    records = discover_patients(cfg)
    first = records[0]
    for modality, alias in {"t1": "t1n", "t1ce": "t1c", "t2": "t2w", "flair": "t2f"}.items():
        path = Path(first["paths"][modality])
        path.rename(path.with_name(f"{first['patient_id']}-{alias}.nii.gz"))
    records = discover_patients(cfg)
    assert load_patient(records[0], cfg)["image"].shape[0] == 4
    import csv
    manifest = tmp_path / "manifest.csv"
    with open(manifest, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["patient_id", *cfg["modalities"], "seg"])
        writer.writeheader()
        for record in records:
            writer.writerow({"patient_id": record["patient_id"], **record["paths"]})
    cfg["manifest"] = str(manifest)
    assert discover_patients(cfg) == records


def test_unlabeled_patient(cfg):
    record = discover_patients(cfg)[0]
    Path(record["paths"]["seg"]).unlink()
    records = discover_patients(cfg, require_seg=False)
    assert load_patient(records[0], cfg)["label"] is None
    with pytest.raises(ValueError, match="missing seg"):
        discover_patients(cfg)


def test_cache_identity_and_expert_hashes(tmp_path):
    path = tmp_path / "patient.pt"
    torch.save({"patient_id": "test_patient", "provenance": {"expert_hashes": "old"}}, path)
    with pytest.raises(ValueError, match="Stale"):
        load_cache(path, {"expert_hashes": "new"}, ["controller_patient"])
    with pytest.raises(ValueError, match="outside"):
        load_cache(path, {"expert_hashes": "old"}, ["controller_patient"])


def test_gate_smaller_than_every_expert(cfg):
    gate_count = sum(p.numel() for p in GatingNetwork(feature_channels(cfg), cfg["controller_width"]).parameters())
    assert all(gate_count < sum(p.numel() for p in build_expert(name, cfg).parameters()) for name in EXPERTS)


def test_five_expert_count_does_not_change_segmentation_tuple():
    assert EXPERTS == SEGMENTATION_EXPERTS == ("cnn", "transformer", "boundary", "highres")
    assert REASONING_EXPERT == "llm"
    assert SYSTEM_EXPERTS == ("cnn", "transformer", "boundary", "highres", "llm")
    assert "llm" not in EXPERTS


def test_build_expert_rejects_llm(cfg):
    with pytest.raises(ValueError, match="reasoning expert"):
        build_expert("llm", cfg)
