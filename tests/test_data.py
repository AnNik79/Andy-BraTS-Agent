import numpy as np
import nibabel as nib
import pytest
import torch
from brats_debate.data.brats_dataset import discover_patients, load_patient, convert_labels, make_splits, BraTSPatches
from brats_debate.data.preprocessing import crop_pad, restore_crop, normalize_nonzero, geometry_targets
from brats_debate.experts import build_expert
from brats_debate.training.train_expert import expert_loss


def test_load_and_patient_splits(cfg):
    records = discover_patients(cfg)
    patient = load_patient(records[0], cfg)
    assert patient["image"].shape == (4, 16, 16, 16)
    assert set(np.unique(patient["label"])) == {0, 1, 2, 3}
    splits = make_splits(records, cfg)
    assert len({pid for ids in splits.values() for pid in ids}) == 4
    assert splits == make_splits(records, cfg)


def test_alignment_rejected(cfg):
    record = discover_patients(cfg)[0]
    path = record["paths"]["t2"]
    img = nib.load(path)
    affine = img.affine.copy()
    affine[0, 3] += 2
    nib.save(nib.Nifti1Image(img.get_fdata(), affine), path)
    with pytest.raises(ValueError, match="alignment"):
        load_patient(record, cfg)


def test_conversion_normalization_and_crop():
    assert convert_labels(np.array([0, 1, 2, 4]), {0: 0, 1: 1, 2: 2, 4: 3}).tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="Unknown"):
        convert_labels(np.array([3]), {0: 0, 4: 1})
    with pytest.raises(ValueError, match="integer"):
        convert_labels(np.array([1.5]), {0: 0, 1: 1})
    arr = np.array([[[[0., 2., 4.]]]])
    np.testing.assert_allclose(normalize_nonzero(arr), [[[[0, -1, 1]]]])
    x = np.arange(5 * 6 * 7).reshape(5, 6, 7)
    padded, meta = crop_pad(x, (8, 9, 10))
    np.testing.assert_array_equal(restore_crop(padded, meta), x)
    cropped, meta = crop_pad(x, (3, 4, 5))
    restored = restore_crop(cropped, meta)
    assert np.count_nonzero(restored) == np.count_nonzero(cropped)


def test_boundary_geometry():
    labels = np.zeros((8, 8, 8), np.int64)
    labels[2:6, 2:6, 2:6] = 1
    boundary, distance = geometry_targets(labels)
    assert boundary.any() and not boundary[3, 3, 3]
    assert distance[boundary.astype(bool)].max() == 0


def test_boundary_geometry_is_precomputed_on_cpu_patch(cfg):
    records = discover_patients(cfg)
    dataset = BraTSPatches(records, cfg, geometry=True)
    image, label, boundary, distance = dataset.sample(load_patient(records[0], cfg), 0)
    expected_b, expected_d = geometry_targets(np.ascontiguousarray(label.numpy()))
    np.testing.assert_array_equal(boundary[0].numpy(), expected_b)
    np.testing.assert_allclose(distance[0].numpy(), expected_d)
    assert image.device.type == "cpu" and label.device.type == "cpu"
    model = build_expert("boundary", cfg)
    output = model(image[None])
    precomputed = expert_loss(output, label[None], (boundary[None], distance[None]))
    fallback = expert_loss(output, label[None])
    torch.testing.assert_close(precomputed, fallback)
    if torch.backends.mps.is_available():
        with pytest.raises(ValueError, match="precomputed"):
            expert_loss(output, label[None].to("mps"))


def test_split_overlap_rejected(cfg):
    import json
    from pathlib import Path
    records = discover_patients(cfg)
    splits = make_splits(records, cfg)
    splits["test"] = splits["expert_train"]
    (Path(cfg["output_dir"]) / "splits.json").write_text(json.dumps(splits))
    with pytest.raises(ValueError, match="overlap"):
        make_splits(records, cfg)
