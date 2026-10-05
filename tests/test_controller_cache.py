from collections import namedtuple
from pathlib import Path
import numpy as np
import pytest
import torch
from brats_debate.config import SEGMENTATION_EXPERTS, fingerprint, load_config, protocol
from brats_debate.controller.controller_dataset import (
    ControllerDataset, assert_cache_space, cache_provenance, choose_patch_center, extract_training_patches,
    save_training_cache, training_cache_estimate, training_cache_is_complete, training_cache_path,
)
from brats_debate.debate.features import feature_channels
from brats_debate.experts.base import pack_output
from brats_debate.training.train_controller import (
    generate_controller_data, records_for, score_controller_volume, validate_controller,
)

ROOT = Path(__file__).resolve().parents[1]
Usage = namedtuple("Usage", "total used free")


def splits():
    return {"expert_train": ["expert_case"], "controller_train": ["train_case"],
            "validation": ["val_case"], "test": ["test_case"]}


def controller_cfg(tmp_path, crop=8, patches=2):
    cfg = load_config(ROOT / "configs" / "brats_controller.yaml")
    cfg.update(output_dir=str(tmp_path), device="cpu", seed=7, patches_per_patient=patches,
               crop_size=[crop, crop, crop], mixed_precision=False, num_workers=0, batch_size=1)
    return cfg


def records():
    return [{"patient_id": "train_case"}, {"patient_id": "val_case"}, {"patient_id": "test_case"},
            {"patient_id": "expert_case"}]


def patient(patient_id, spatial=(8, 8, 8)):
    label = np.zeros(spatial, np.int64)
    label[2:5, 2:5, 2:5] = 1
    return {"patient_id": patient_id, "image": np.zeros((4, *spatial), np.float32), "label": label}


def fake_outputs(spatial=(8, 8, 8)):
    outputs = {}
    for name in SEGMENTATION_EXPERTS:
        extra = {"availability": torch.ones(1, 1, *spatial)}
        if name == "boundary":
            extra["boundary_probability"] = torch.zeros(1, 1, *spatial)
            extra["distance_to_boundary"] = torch.zeros(1, 1, *spatial)
        outputs[name] = pack_output(torch.zeros(1, 4, *spatial), torch.zeros(1, 4, *spatial), extra)
    return outputs


def test_controller_config_keeps_frozen_protocol_and_checkpoints():
    cfg = load_config(ROOT / "configs" / "brats_controller.yaml")
    expert = load_config(ROOT / "configs" / "brats_highres_curve5.yaml")
    assert fingerprint(protocol(cfg)) == fingerprint(protocol(expert))
    assert feature_channels(cfg) == 65
    assert cfg["controller_epochs"] == 20
    assert cfg["learning_rate"] == 0.0003
    assert cfg["controller_width"] == 8
    assert cfg["batch_size"] == 1
    assert cfg["patches_per_patient"] == 4
    assert cfg["crop_size"] == [64, 64, 64]
    assert cfg["device"] == "mps"
    assert cfg["evaluation"]["hd95"] is False
    assert set(cfg["expert_checkpoints"]) == set(SEGMENTATION_EXPERTS)
    assert "llm" not in cfg["expert_checkpoints"]
    for path in cfg["expert_checkpoints"].values():
        assert Path(path).is_file()


def test_training_cache_stores_sampled_patches_only(tmp_path):
    cfg = controller_cfg(tmp_path)
    image = torch.zeros(1, 4, 8, 8, 8)
    outputs = fake_outputs()
    from brats_debate.debate.disagreement import analyze_disagreement
    from brats_debate.debate.features import controller_features, stack_evidence
    debate = analyze_disagreement(outputs)
    features = controller_features(image, outputs, debate)
    probabilities, availability = stack_evidence(outputs)
    label = patient("train_case")["label"]
    patches = extract_training_patches(features, probabilities, availability, label, debate["map"][0, 0],
                                       cfg, "train_case")
    assert len(patches) == 2
    assert patches[0]["features"].shape == (65, 8, 8, 8)
    assert patches[0]["probabilities"].shape == (4, 4, 8, 8, 8)
    assert patches[0]["availability"].shape == (4, 8, 8, 8)
    assert patches[0]["label"].shape == (8, 8, 8)
    assert set(np.unique(patches[0]["label"].numpy())).issubset({0, 1})
    provenance = cache_provenance(cfg, splits(), {name: "hash" for name in SEGMENTATION_EXPERTS})
    path = training_cache_path(cfg, "train_case")
    save_training_cache(path, "train_case", provenance, patches)
    blob = path.read_bytes()
    assert b"test_case" not in blob
    saved = torch.load(path, map_location="cpu", weights_only=True)
    assert set(saved) == {"format", "complete", "split", "patient_id", "provenance", "patches"}
    assert saved["patches"][0]["features"].shape == (65, 8, 8, 8)
    dataset = ControllerDataset(cfg, splits(), {name: "hash" for name in SEGMENTATION_EXPERTS})
    batch = dataset[0]
    assert batch["features"].shape == (65, 8, 8, 8)
    assert batch["label"].shape == (8, 8, 8)
    full_volume_bytes = 65 * 240 * 240 * 155 * 2
    assert path.stat().st_size < full_volume_bytes / 10


def test_disagreement_center_rule_is_unchanged():
    label = np.zeros((6, 6, 6), np.int64)
    disagreement = np.zeros((6, 6, 6), np.float32)
    disagreement[1, 2, 3] = 0.5

    class Early:
        def random(self):
            return 0.1

        def integers(self, count):
            assert count == 1
            return 0

    assert choose_patch_center(label, disagreement, Early()) == (1, 2, 3)


def test_resume_accepts_only_complete_caches(tmp_path):
    cfg = controller_cfg(tmp_path)
    provenance = cache_provenance(cfg, splits(), {name: "hash" for name in SEGMENTATION_EXPERTS})
    path = training_cache_path(cfg, "train_case")
    path.parent.mkdir(parents=True)
    partial = path.with_suffix(".partial")
    partial.write_bytes(b"incomplete")
    path.write_bytes(b"not a torch cache")
    assert training_cache_is_complete(path, cfg, "train_case", provenance) is False
    assert training_cache_is_complete(partial, cfg, "train_case", provenance) is False
    patches = extract_training_patches(
        torch.zeros(1, 65, 8, 8, 8), torch.zeros(1, 4, 4, 8, 8, 8), torch.ones(1, 4, 8, 8, 8),
        np.zeros((8, 8, 8), np.int64), np.zeros((8, 8, 8), np.float32), cfg, "train_case")
    save_training_cache(path, "train_case", provenance, patches)
    assert training_cache_is_complete(path, cfg, "train_case", provenance) is True
    assert training_cache_is_complete(partial, cfg, "train_case", provenance) is False


def test_preflight_refuses_insufficient_space_without_writing(tmp_path, monkeypatch):
    cfg = controller_cfg(tmp_path)
    monkeypatch.setattr("brats_debate.controller.controller_dataset.shutil.disk_usage",
                        lambda path: Usage(100, 100, 0))
    estimate = training_cache_estimate(1, cfg)
    with pytest.raises(RuntimeError, match="No files were deleted"):
        assert_cache_space(tmp_path, estimate)
    loaded = []
    monkeypatch.setattr("brats_debate.training.train_controller.load_patient",
                        lambda record, cfg: loaded.append(record["patient_id"]))
    monkeypatch.setattr("brats_debate.training.train_controller.load_experts",
                        lambda cfg, splits: ({}, {name: "hash" for name in SEGMENTATION_EXPERTS}))
    with pytest.raises(RuntimeError, match="No files were deleted"):
        generate_controller_data(records(), splits(), cfg)
    assert loaded == []
    assert list(tmp_path.rglob("*.pt")) == []


def test_generate_skips_completed_cases_and_never_loads_test(tmp_path, monkeypatch):
    cfg = controller_cfg(tmp_path)
    loaded, predicted = [], []

    def load_patient(record, _cfg):
        loaded.append(record["patient_id"])
        if record["patient_id"] == "test_case":
            raise AssertionError("test patient was loaded")
        return patient(record["patient_id"])

    def predict(_patient, _models, _cfg):
        predicted.append(_patient["patient_id"])
        return fake_outputs()

    monkeypatch.setattr("brats_debate.training.train_controller.load_patient", load_patient)
    monkeypatch.setattr("brats_debate.training.train_controller.load_experts",
                        lambda cfg, splits: ({}, {name: "hash" for name in SEGMENTATION_EXPERTS}))
    monkeypatch.setattr("brats_debate.training.train_controller._expert_outputs", predict)
    monkeypatch.setattr("brats_debate.controller.controller_dataset.shutil.disk_usage",
                        lambda path: Usage(10**15, 0, 10**15))
    generate_controller_data(records(), splits(), cfg)
    assert loaded == ["train_case"]
    assert predicted == ["train_case"]
    cache = training_cache_path(cfg, "train_case")
    assert cache.is_file()
    assert not (tmp_path / "controller_cache" / "test_case.pt").exists()
    assert not (tmp_path / "controller_cache" / "val_case.pt").exists()
    generate_controller_data(records(), splits(), cfg)
    assert loaded == ["train_case"]
    assert predicted == ["train_case"]
    cache.write_bytes(b"corrupt")
    generate_controller_data(records(), splits(), cfg)
    assert predicted == ["train_case", "train_case"]
    assert training_cache_is_complete(
        cache, cfg, "train_case", cache_provenance(cfg, splits(), {name: "hash" for name in SEGMENTATION_EXPERTS}))


def test_validation_streams_full_volumes_without_storing_features(tmp_path, monkeypatch):
    cfg = controller_cfg(tmp_path, crop=8)
    seen = []

    def load_patient(record, _cfg):
        if record["patient_id"] == "test_case":
            raise AssertionError("test patient was loaded")
        seen.append(record["patient_id"])
        label = np.zeros((10, 6, 6), np.int64)
        label[8:10, 1:4, 1:4] = 1
        return {"patient_id": record["patient_id"], "image": np.zeros((4, 10, 6, 6), np.float32), "label": label}

    def predict(volume, _models, _cfg):
        outputs = fake_outputs(volume["label"].shape)
        probabilities = outputs["cnn"]["probabilities"]
        probabilities.zero_()
        labels = torch.as_tensor(volume["label"])
        for class_id in range(4):
            probabilities[0, class_id][labels == class_id] = 1
        return outputs

    sentinel = tmp_path / "controller_cache" / "val_case.pt"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"do not read")
    monkeypatch.setattr("brats_debate.training.train_controller.load_patient", load_patient)
    monkeypatch.setattr("brats_debate.training.train_controller._expert_outputs", predict)

    class Gate(torch.nn.Module):
        def forward(self, features, probabilities, availability):
            return {"probabilities": probabilities[:, 0], "weights": availability}

    loss, dice = validate_controller(Gate(), {}, records(), cfg, splits(), torch.device("cpu"))
    assert seen == ["val_case"]
    assert sentinel.read_bytes() == b"do not read"
    assert list(sentinel.parent.glob("*.pt")) == [sentinel]
    assert loss >= 0
    assert dice == pytest.approx(1.0)


def test_score_uses_the_tail_slab():
    cfg = controller_cfg(Path("/tmp"), crop=6)
    depth, height, width = 10, 6, 6
    probabilities = torch.zeros(1, 4, 4, depth, height, width)
    probabilities[:, 0, 1] = 1
    features = torch.zeros(1, feature_channels(cfg), depth, height, width)
    availability = torch.ones(1, 4, depth, height, width)
    label = np.ones((depth, height, width), np.int64)

    class Gate(torch.nn.Module):
        def forward(self, features, probabilities, availability):
            return {"probabilities": probabilities[:, 0], "weights": availability}

    _loss, dice = score_controller_volume(Gate(), features, probabilities, availability, label, cfg,
                                          torch.device("cpu"))
    assert dice == pytest.approx(1.0)
    label_miss = label.copy()
    label_miss[8:] = 0
    _loss, missed = score_controller_volume(Gate(), features, probabilities, availability, label_miss, cfg,
                                            torch.device("cpu"))
    assert missed < 0.99


def test_records_for_never_returns_test_cases():
    chosen = records_for(records(), ["train_case"], ["test_case"])
    assert [record["patient_id"] for record in chosen] == ["train_case"]
    with pytest.raises(RuntimeError, match="forbidden"):
        records_for(records(), ["test_case"], ["test_case"])


def test_real_training_cache_estimate_excludes_validation_volumes():
    cfg = load_config(ROOT / "configs" / "brats_controller.yaml")
    estimate = training_cache_estimate(248, cfg)
    assert estimate["stores_validation_volumes"] is False
    assert estimate["feature_channels"] == 65
    assert estimate["patches_per_case"] == 4
    assert estimate["crop_size"] == [64, 64, 64]
    per_case = estimate["tensor_bytes"] / 248
    assert per_case < 250 * 1024 ** 2
    assert estimate["required_bytes"] < 60 * 1024 ** 3
