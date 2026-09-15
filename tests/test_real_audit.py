from pathlib import Path
import json
import zipfile
import nibabel as nib
import numpy as np
import pytest
from brats_debate.data.real_audit import inspect_case, archive_inventory, source_stamp
from brats_debate.data.brats_dataset import discover_patients, load_patient, make_splits, subject_id, validate_training_audit


def test_archive_loading_and_raw_audit(cfg, tmp_path):
    archive = tmp_path / "training.zip"
    pid = "BraTS-GLI-00001-000"
    with zipfile.ZipFile(archive, "w") as z:
        for name in ("t1n", "t1c", "t2w", "t2f", "seg"):
            path = tmp_path / f"{pid}-{name}.nii.gz"
            array = np.random.default_rng(7).normal(size=(8, 8, 8)).astype(np.float32)
            if name == "seg":
                array = np.zeros((8, 8, 8), np.int16)
                array[2:6, 2:6, 2:6] = 3
            nib.save(nib.Nifti1Image(array, np.eye(4)), path)
            z.write(path, f"Training/{pid}/{path.name}")
    cases, unexpected = archive_inventory(archive)
    assert not unexpected
    row = inspect_case(archive, pid, cases[pid])
    assert row["usable"] and row["files"]["seg"]["labels"] == [0, 3]
    cfg.update(dataset_archive=str(archive), modalities=["t1n", "t1c", "t2w", "t2f"], label_mapping={0: 0, 1: 1, 2: 2, 3: 3})
    records = discover_patients(cfg)
    patient = load_patient(records[0], cfg)
    assert patient["image"].shape == (4, 8, 8, 8)
    assert set(np.unique(patient["label"])) == {0, 3}
    report = tmp_path / "audit.json"
    report.write_text(json.dumps({"passed": True, "sanity_visualizations_passed": False}))
    cfg.update(require_dataset_audit=True, audit_report=str(report))
    with pytest.raises(ValueError, match="visual"):
        validate_training_audit(cfg)


def test_subject_group_splits_and_leak_rejection(cfg):
    cfg["patient_group_regex"] = r"^(BraTS-GLI-\d+)-\d+$"
    records = [{"patient_id": f"BraTS-GLI-{i:05d}-{scan:03d}"} for i in range(12) for scan in range(2)]
    splits = make_splits(records, cfg)
    owners = {}
    for split, ids in splits.items():
        for pid in ids:
            group = subject_id({"patient_id": pid}, cfg)
            assert group not in owners or owners[group] == split
            owners[group] = split
    assert len(owners) == 12
    assert splits == make_splits(list(reversed(records)), cfg)
    moved = splits["expert_train"].pop()
    splits["test"].append(moved)
    (Path(cfg["output_dir"]) / "splits.json").write_text(json.dumps(splits))
    with pytest.raises(ValueError, match="Subject leakage"):
        make_splits(records, cfg)


def test_corrupt_archive_case_is_not_usable(tmp_path):
    archive = tmp_path / "broken.zip"
    pid = "BraTS-GLI-00001-000"
    member = f"Training/{pid}/{pid}-t1c.nii.gz"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(member, b"not gzip")
    row = inspect_case(archive, pid, {"t1c": member})
    assert not row["usable"] and len(row["missing"]) == 4 and row["errors"]
