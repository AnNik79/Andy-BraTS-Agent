from pathlib import Path
import json
import numpy as np
import pytest
import torch
from brats_debate.config import SEGMENTATION_EXPERTS, load_config
from brats_debate.controller.gating_network import GatingNetwork
from brats_debate.debate.features import feature_channels
from brats_debate.experts.base import pack_output
from brats_debate.training.train_controller import (
    _refuse_existing_run, _save_controller_epoch, aggregate_validation_scores, controller_run_dir,
    epoch_checkpoint_path, generate_controller_data, score_controller_metrics, score_controller_volume,
    sorted_epoch_checkpoints, train_controller, validate_controller, validate_saved_controllers,
)

ROOT = Path(__file__).resolve().parents[1]
HASHES = {name: "hash" for name in SEGMENTATION_EXPERTS}


def controller_cfg(tmp_path, crop=8, patches=2):
    cfg = load_config(ROOT / "configs" / "brats_controller.yaml")
    cfg.update(output_dir=str(tmp_path), device="cpu", seed=7, patches_per_patient=patches,
               crop_size=[crop, crop, crop], mixed_precision=False, num_workers=0, batch_size=1)
    return cfg


def cache_splits():
    return {"expert_train": ["expert_case"], "controller_train": ["train_case"],
            "validation": ["val_case"], "test": ["test_case"]}


def validation_splits():
    return {"expert_train": ["expert_case"], "controller_train": ["train_case"],
            "validation": ["val_a", "val_b"], "test": ["test_case"]}


def records():
    return [{"patient_id": patient_id} for patient_id in
            ("train_case", "val_a", "val_b", "test_case", "expert_case")]


def labeled_patient(patient_id):
    label = np.zeros((10, 6, 6), np.int64)
    if patient_id != "val_b":
        label[8:10, 1:4, 1:4] = 1
    return {"patient_id": patient_id, "image": np.zeros((4, 10, 6, 6), np.float32), "label": label}


def fake_outputs(spatial):
    outputs = {}
    for name in SEGMENTATION_EXPERTS:
        extra = {"availability": torch.ones(1, 1, *spatial)}
        if name == "boundary":
            extra["boundary_probability"] = torch.zeros(1, 1, *spatial)
            extra["distance_to_boundary"] = torch.zeros(1, 1, *spatial)
        outputs[name] = pack_output(torch.zeros(1, 4, *spatial), torch.zeros(1, 4, *spatial), extra)
    return outputs


def expert_outputs(patient):
    outputs = fake_outputs(patient["label"].shape)
    labels = torch.as_tensor(patient["label"])
    cnn = outputs["cnn"]["probabilities"]
    cnn.zero_()
    for class_id in range(4):
        cnn[0, class_id][labels == class_id] = 1
    transformer = outputs["transformer"]["probabilities"]
    transformer.zero_()
    transformer[0, 0] = 1
    return outputs


def one_hot_gate(expert_index, cfg):
    model = GatingNetwork(feature_channels(cfg), cfg["controller_width"])
    with torch.no_grad():
        model.network[2].weight.zero_()
        model.network[2].bias.fill_(-20)
        model.network[2].bias[expert_index] = 20
    return model


def save_gate(cfg, split_ids, model, epoch):
    device = torch.device("cpu")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator().manual_seed(cfg["seed"])
    return _save_controller_epoch(cfg, model, optimizer, scaler, generator, split_ids, HASHES,
                                  epoch, 0.0, device)


def install_experts(monkeypatch, loaded, predicted):
    def load_patient(record, _cfg):
        if record["patient_id"] == "test_case":
            raise AssertionError("test patient was loaded")
        loaded.append(record["patient_id"])
        return labeled_patient(record["patient_id"])

    def predict(patient, _models, cfg, device):
        assert cfg["inference"]["highres_max_patches"] == 128
        assert cfg["inference"]["overlap"] == 0.5
        assert cfg["evaluation"]["hd95"] is False
        assert device.type == "cpu"
        predicted.append(patient["patient_id"])
        return expert_outputs(patient)

    monkeypatch.setattr("brats_debate.training.train_controller.load_patient", load_patient)
    monkeypatch.setattr("brats_debate.training.train_controller.load_experts",
                        lambda cfg, splits: ({"kept": object()}, dict(HASHES)))
    monkeypatch.setattr("brats_debate.inference.pipeline.predict_experts", predict)


def test_appledouble_sidecars_do_not_block_resume(tmp_path):
    cfg = controller_cfg(tmp_path)
    directory = controller_run_dir(cfg)
    directory.mkdir(parents=True)
    (directory / "epoch_001.pt").write_bytes(b"real")
    (directory / "._epoch_001.pt").write_bytes(b"sidecar")
    _refuse_existing_run(cfg)
    (directory / "notes.pt").write_bytes(b"other")
    with pytest.raises(FileExistsError, match="unexpected checkpoints"):
        _refuse_existing_run(cfg)


def test_epoch_files_sort_numerically(tmp_path):
    (tmp_path / "epoch_10.pt").write_bytes(b"later")
    (tmp_path / "epoch_2.pt").write_bytes(b"earlier")
    ordered = [path.name for path in sorted_epoch_checkpoints(tmp_path)]
    assert ordered == ["epoch_2.pt", "epoch_10.pt"]
    (tmp_path / "epoch_002.pt").write_bytes(b"duplicate")
    with pytest.raises(FileExistsError, match="Duplicate"):
        sorted_epoch_checkpoints(tmp_path)


def test_tie_keeps_the_earliest_epoch(tmp_path):
    early = tmp_path / "epoch_2.pt"
    late = tmp_path / "epoch_10.pt"
    early.write_bytes(b"early")
    late.write_bytes(b"late")
    score = {"loss": 0.2, "brats_mean_dice": 0.5, "WT": 0.4, "TC": 0.5, "ET": 0.6}
    cases = [{"scores": [{"epoch": 10, **score}, {"epoch": 2, **score}]}]
    per_epoch, chosen = aggregate_validation_scores(cases, [late, early])
    assert [row["epoch"] for row in per_epoch] == [2, 10]
    assert chosen["epoch"] == 2


def test_training_resume_matches_an_uninterrupted_run(tmp_path, monkeypatch):
    loaded, predicted = [], []

    def load_patient(record, _cfg):
        if record["patient_id"] == "test_case":
            raise AssertionError("test patient was loaded")
        loaded.append(record["patient_id"])
        label = np.zeros((8, 8, 8), np.int64)
        label[2:5, 2:5, 2:5] = 1
        return {"patient_id": record["patient_id"], "image": np.zeros((4, 8, 8, 8), np.float32), "label": label}

    def predict(_patient, _models, _cfg):
        predicted.append(_patient["patient_id"])
        return fake_outputs((8, 8, 8))

    def refuse_inference(*_args, **_kwargs):
        raise AssertionError("expert inference ran during controller training")

    monkeypatch.setattr("brats_debate.training.train_controller.load_patient", load_patient)
    monkeypatch.setattr("brats_debate.training.train_controller.load_experts",
                        lambda cfg, splits: ({}, dict(HASHES)))
    monkeypatch.setattr("brats_debate.training.train_controller.validate_controller",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("validation ran during training")))

    def prepare(directory):
        monkeypatch.setattr("brats_debate.training.train_controller._expert_outputs", predict)
        cfg = controller_cfg(directory)
        cfg["controller_epochs"] = 2
        generate_controller_data(records(), cache_splits(), cfg)
        monkeypatch.setattr("brats_debate.training.train_controller._expert_outputs", refuse_inference)
        return cfg

    resumed_cfg = prepare(tmp_path / "resumed")
    train_controller(records(), cache_splits(), resumed_cfg, stop_after_epoch=1)
    first_epoch = epoch_checkpoint_path(resumed_cfg, 1).read_bytes()
    assert not (controller_run_dir(resumed_cfg) / "best.pt").exists()
    train_controller(records(), cache_splits(), resumed_cfg)
    assert epoch_checkpoint_path(resumed_cfg, 1).read_bytes() == first_epoch
    assert epoch_checkpoint_path(resumed_cfg, 2).is_file()
    assert not (controller_run_dir(resumed_cfg) / "best.pt").exists()
    history = (controller_run_dir(resumed_cfg) / "history.csv").read_text().strip().splitlines()
    assert [line.split(",")[0] for line in history[1:]] == ["1", "2"]

    full_cfg = prepare(tmp_path / "full")
    train_controller(records(), cache_splits(), full_cfg)
    resumed_weights = torch.load(epoch_checkpoint_path(resumed_cfg, 2), map_location="cpu", weights_only=True)
    full_weights = torch.load(epoch_checkpoint_path(full_cfg, 2), map_location="cpu", weights_only=True)
    assert resumed_weights["state_dict"].keys() == full_weights["state_dict"].keys()
    assert all(torch.equal(resumed_weights["state_dict"][key], full_weights["state_dict"][key])
               for key in resumed_weights["state_dict"])
    finished = epoch_checkpoint_path(full_cfg, 2).read_bytes()
    train_controller(records(), cache_splits(), full_cfg)
    assert epoch_checkpoint_path(full_cfg, 2).read_bytes() == finished
    (controller_run_dir(full_cfg) / "best.pt").write_bytes(b"previous run")
    with pytest.raises(FileExistsError, match="best.pt"):
        train_controller(records(), cache_splits(), full_cfg)
    assert epoch_checkpoint_path(full_cfg, 2).read_bytes() == finished
    assert loaded == ["train_case", "train_case"]
    assert predicted == ["train_case", "train_case"]


def test_corrupt_or_gapped_epoch_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr("brats_debate.training.train_controller.load_experts",
                        lambda cfg, splits: ({}, dict(HASHES)))
    cfg = controller_cfg(tmp_path)
    cfg["controller_epochs"] = 2
    directory = controller_run_dir(cfg)
    directory.mkdir(parents=True)
    broken = epoch_checkpoint_path(cfg, 1)
    broken.write_bytes(b"not a checkpoint")
    with pytest.raises(FileExistsError, match="not treated as done"):
        train_controller(records(), cache_splits(), cfg)
    assert broken.read_bytes() == b"not a checkpoint"
    broken.unlink()
    gap = directory / "epoch_002.pt"
    gap.write_bytes(b"gap")
    with pytest.raises(FileExistsError, match="contiguous"):
        train_controller(records(), cache_splits(), cfg)
    assert gap.read_bytes() == b"gap"


def test_deferred_validation_matches_single_checkpoint_scoring(tmp_path, monkeypatch, capsys):
    loaded, predicted = [], []
    install_experts(monkeypatch, loaded, predicted)
    cfg = controller_cfg(tmp_path, crop=6)
    cfg["controller_epochs"] = 2
    split_ids = validation_splits()
    cnn_gate = one_hot_gate(0, cfg)
    transformer_gate = one_hot_gate(1, cfg)
    save_gate(cfg, split_ids, cnn_gate, 1)
    save_gate(cfg, split_ids, transformer_gate, 2)
    device = torch.device("cpu")
    reference = {}
    per_patient = {}
    for epoch, model in ((1, cnn_gate), (2, transformer_gate)):
        _loss, dice = validate_controller(model, {}, records(), cfg, split_ids, device)
        per_patient[epoch] = {}
        for patient_id in ("val_a", "val_b"):
            patient = labeled_patient(patient_id)
            from brats_debate.training.train_controller import _controller_volumes
            features, probabilities, availability, _disagreement = _controller_volumes(
                patient, expert_outputs(patient))
            metrics = score_controller_metrics(model, features, probabilities, availability,
                                               patient["label"], cfg, device)
            volume_loss, volume_dice = score_controller_volume(
                model, features, probabilities, availability, patient["label"], cfg, device)
            assert volume_loss == metrics["loss"]
            assert volume_dice == metrics["brats_mean_dice"]
            per_patient[epoch][patient_id] = metrics
        mean = {key: sum(per_patient[epoch][patient_id][key] for patient_id in ("val_a", "val_b")) / 2
                for key in ("loss", "brats_mean_dice", "WT", "TC", "ET")}
        assert dice == mean["brats_mean_dice"]
        reference[epoch] = mean
    assert reference[1]["brats_mean_dice"] == pytest.approx(1.0)
    assert per_patient[2]["val_a"]["brats_mean_dice"] < 0.99
    assert reference[2]["brats_mean_dice"] != per_patient[2]["val_a"]["brats_mean_dice"]
    predicted.clear()
    loaded.clear()
    summary = validate_saved_controllers(records(), split_ids, cfg)
    assert predicted == ["val_a", "val_b"]
    assert loaded == ["val_a", "val_b"]
    assert "test_case" not in predicted
    assert summary["selected_epoch"] == 1
    assert summary["metadata"]["feature_channels"] == 65
    assert summary["metadata"]["hd95"] is False
    assert summary["metadata"]["slab"] == 8
    assert summary["provenance"]["highres_max_patches"] == 128
    for row in summary["per_epoch"]:
        assert row["brats_mean_dice"] == reference[row["epoch"]]["brats_mean_dice"]
        assert row["WT"] == reference[row["epoch"]]["WT"]
        assert row["TC"] == reference[row["epoch"]]["TC"]
        assert row["ET"] == reference[row["epoch"]]["ET"]
    for patient_id in ("val_a", "val_b"):
        saved_case = json.loads((tmp_path / "controller_validation" / "cases" / f"{patient_id}.json").read_text())
        for row in saved_case["scores"]:
            expected = per_patient[row["epoch"]][patient_id]
            assert row["brats_mean_dice"] == expected["brats_mean_dice"]
            assert row["WT"] == expected["WT"]
            assert row["TC"] == expected["TC"]
            assert row["ET"] == expected["ET"]
    best = controller_run_dir(cfg) / "best.pt"
    selected = epoch_checkpoint_path(cfg, 1)
    assert best.read_bytes() == selected.read_bytes()
    case_path = tmp_path / "controller_validation" / "cases" / "val_a.json"
    case_bytes = case_path.read_bytes()
    best.unlink()
    predicted.clear()
    validate_saved_controllers(records(), split_ids, cfg)
    assert predicted == []
    assert case_path.read_bytes() == case_bytes
    assert best.read_bytes() == selected.read_bytes()
    case_path.write_bytes(b"incomplete")
    predicted.clear()
    validate_saved_controllers(records(), split_ids, cfg)
    assert predicted == ["val_a"]
    assert "will be recomputed" in capsys.readouterr().out
    assert (tmp_path / "controller_validation" / "summary.json").is_file()
    assert (tmp_path / "controller_validation" / "summary.csv").is_file()
    assert not list((tmp_path / "controller_validation").rglob("*.pt"))


def test_validation_does_not_promote_best_when_a_case_fails(tmp_path, monkeypatch):
    loaded, predicted = [], []
    install_experts(monkeypatch, loaded, predicted)

    def fail_second(patient, _models, cfg, device):
        predicted.append(patient["patient_id"])
        if patient["patient_id"] == "val_b":
            raise RuntimeError("interrupted validation case")
        return expert_outputs(patient)

    monkeypatch.setattr("brats_debate.inference.pipeline.predict_experts", fail_second)
    cfg = controller_cfg(tmp_path, crop=6)
    cfg["controller_epochs"] = 2
    split_ids = validation_splits()
    save_gate(cfg, split_ids, one_hot_gate(0, cfg), 1)
    save_gate(cfg, split_ids, one_hot_gate(1, cfg), 2)
    with pytest.raises(RuntimeError, match="interrupted validation case"):
        validate_saved_controllers(records(), split_ids, cfg)
    assert not (controller_run_dir(cfg) / "best.pt").exists()
    assert not (tmp_path / "controller_validation" / "summary.json").exists()
    assert (tmp_path / "controller_validation" / "cases" / "val_a.json").is_file()
    saved = json.loads((tmp_path / "controller_validation" / "cases" / "val_a.json").read_text())
    assert saved["complete"] is True
    assert [row["epoch"] for row in saved["scores"]] == [1, 2]


def test_missing_epochs_do_not_load_patients(tmp_path, monkeypatch):
    loaded = []
    monkeypatch.setattr("brats_debate.training.train_controller.load_patient",
                        lambda record, cfg: loaded.append(record["patient_id"]))
    monkeypatch.setattr("brats_debate.training.train_controller.load_experts",
                        lambda cfg, splits: (_ for _ in ()).throw(AssertionError("experts loaded before checkpoints")))
    cfg = controller_cfg(tmp_path)
    cfg["controller_epochs"] = 2
    with pytest.raises(FileNotFoundError, match="Incomplete checkpoints"):
        validate_saved_controllers(records(), validation_splits(), cfg)
    assert loaded == []


def test_controller_cli_cannot_select_the_test_split(capsys):
    import sys
    from brats_debate.cli import main
    argv = sys.argv
    try:
        sys.argv = ["validate_controller", "--help"]
        with pytest.raises(SystemExit) as exited:
            main("validate_controller")
        assert exited.value.code == 0
        assert "test split" in capsys.readouterr().out.lower()
        sys.argv = ["train_controller", "--help"]
        with pytest.raises(SystemExit) as exited:
            main("train_controller")
        text = capsys.readouterr().out.lower()
        assert exited.value.code == 0
        assert "test split" in text
        assert "full-volume validation" in text
        sys.argv = ["validate_controller", "--config", "configs/brats_controller.yaml", "--split", "test"]
        with pytest.raises(SystemExit) as exited:
            main("validate_controller")
        assert exited.value.code != 0
    finally:
        sys.argv = argv


def test_sigint_saves_the_current_epoch_and_stops(tmp_path, monkeypatch):
    import os
    import signal
    from brats_debate.training import train_controller as module

    def load_patient(record, _cfg):
        if record["patient_id"] == "test_case":
            raise AssertionError("test patient was loaded")
        return {"patient_id": record["patient_id"], "image": np.zeros((4, 8, 8, 8), np.float32),
                "label": np.zeros((8, 8, 8), np.int64)}

    def predict(_patient, _models, _cfg):
        return fake_outputs((8, 8, 8))

    monkeypatch.setattr(module, "load_patient", load_patient)
    monkeypatch.setattr(module, "load_experts", lambda cfg, splits: ({}, dict(HASHES)))
    monkeypatch.setattr(module, "_expert_outputs", predict)
    cfg = controller_cfg(tmp_path)
    cfg["controller_epochs"] = 3
    generate_controller_data(records(), cache_splits(), cfg)
    monkeypatch.setattr(module, "_expert_outputs",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("expert inference ran during training")))
    original = module._save_controller_epoch

    def save_then_interrupt(*args, **kwargs):
        path = original(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGINT)
        return path

    monkeypatch.setattr(module, "_save_controller_epoch", save_then_interrupt)
    train_controller(records(), cache_splits(), cfg)
    payload = torch.load(epoch_checkpoint_path(cfg, 1), map_location="cpu", weights_only=True)
    assert payload["complete"] is True and payload["epoch"] == 1
    for key in ("state_dict", "optimizer_state_dict", "scaler_state_dict", "rng_state", "loader_generator_state"):
        assert key in payload
    assert not epoch_checkpoint_path(cfg, 2).exists()
    assert not (controller_run_dir(cfg) / "best.pt").exists()


def test_real_config_keeps_controller_inference_settings():
    cfg = load_config(ROOT / "configs" / "brats_controller.yaml")
    model = GatingNetwork(feature_channels(cfg), cfg["controller_width"])
    assert feature_channels(cfg) == 65
    assert model.network[0].in_channels == 65
    assert model.network[0].out_channels == 8
    assert model.network[2].out_channels == 4
    assert cfg["controller_epochs"] == 20
    assert cfg["inference"]["highres_max_patches"] == 128
    assert cfg["inference"]["overlap"] == 0.5
    assert cfg["evaluation"]["hd95"] is False
