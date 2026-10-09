import numpy as np
import pytest
from brats_debate.evaluation.metrics import binary_metrics, segmentation_metrics
from brats_debate.evaluation.disagreement_analysis import case_error_analysis
from brats_debate.inference.pipeline import validate_checkpoint, save_nifti
from brats_debate.config import fingerprint, protocol
from brats_debate.data.brats_dataset import discover_patients, load_patient


def test_metrics_empty_perfect_and_distance():
    empty = np.zeros((6, 6, 6), bool)
    assert binary_metrics(empty, empty, hd95=True)["dice"] == 1
    a, b = empty.copy(), empty.copy()
    a[2, 2, 2] = True
    b[3, 2, 2] = True
    assert binary_metrics(a, b, spacing=(2, 1, 1), hd95=True)["hd95_mm"] == 2
    assert binary_metrics(a, empty, hd95=True)["hd95_mm"] is None
    assert binary_metrics(a, a)["precision"] == 1
    b[2, 2, 2] = True
    assert binary_metrics(a, b)["dice"] == pytest.approx(2 / 3)


def test_brats_et_tc_wt_conversion_and_composite():
    # Internal BraTS-style labels: 1=NCR/NET, 2=edema, 3=ET.
    target = np.zeros((6, 6, 6), np.int16)
    target[1:5, 1:5, 1:5] = 2
    target[2:4, 2:4, 2:4] = 1
    target[2:4, 2:4, 2] = 3
    pred = target.copy()
    regions = {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3], "edema": [2]}
    perfect = segmentation_metrics(pred, target, regions, 4, hd95=True)
    assert perfect["ET"]["dice"] == 1
    assert perfect["TC"]["dice"] == 1
    assert perfect["WT"]["dice"] == 1
    assert perfect["brats_mean_dice"] == 1
    assert perfect["ET"]["hd95_mm"] == 0
    # Predict edema only: WT can remain high while ET/TC collapse.
    edema_only = np.where(target > 0, 2, 0)
    collapsed = segmentation_metrics(edema_only, target, regions, 4)
    edema_true = int((target == 2).sum())
    edema_pred = int((edema_only == 2).sum())
    expected_edema = 2 * edema_true / (edema_pred + edema_true)
    assert collapsed["WT"]["dice"] == 1
    assert collapsed["ET"]["dice"] == 0
    assert collapsed["TC"]["dice"] == 0
    assert collapsed["edema"]["dice"] == pytest.approx(expected_edema)
    assert collapsed["brats_mean_dice"] == pytest.approx((1 + 0 + 0) / 3)
    assert collapsed["mean_dice"] == pytest.approx((1 + 0 + 0 + expected_edema) / 4)
    with pytest.raises(ValueError, match="BraTS region"):
        segmentation_metrics(pred, target, {"edema": [2]}, 4)


def test_error_enrichment():
    target = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 0, 0])
    stats = case_error_analysis(np.array([0., .1, .8, .9]), pred, target, np.ones(4))
    assert stats["error_detection_auroc"] == 1
    assert stats["high_disagreement_error_rate"] == 1
    assert stats["low_disagreement_error_rate"] == 0


def test_provenance_rejects_leakage(cfg):
    splits = {"expert_train": ["a"], "controller_train": ["b"], "validation": ["c"], "test": ["d"]}
    checkpoint = {"protocol_hash": fingerprint(protocol(cfg)), "split_hash": fingerprint(splits),
                  "training_patients": ["d"], "role": "expert", "name": "cnn"}
    with pytest.raises(ValueError, match="leakage"):
        validate_checkpoint(checkpoint, cfg, splits, "expert", "cnn")


def test_final_evaluation_resumes_without_double_counting(tmp_path, monkeypatch):
    import json
    import torch
    from brats_debate.evaluation.evaluate import (
        CASE_FORMAT, _case_is_complete, _case_path, evaluate,
    )

    patient_id = "BraTS-GLI-00001-001"
    other_id = "BraTS-GLI-00002-001"
    shape = (4, 4, 4)
    image = np.zeros((4, *shape), np.float32)
    image[0, 0, 0, 0] = 1
    label = np.zeros(shape, np.int16)
    import nibabel as nib
    patient = {"patient_id": patient_id, "image": image, "label": label,
               "spacing": (1.0, 1.0, 1.0), "affine": np.eye(4), "header": nib.Nifti1Header()}
    probabilities = torch.full((1, 4, *shape), 0.25)
    segmentation = torch.zeros((1, *shape), dtype=torch.int64)
    calls = {"load": 0, "run": 0}

    class _Model:
        def to(self, device):
            return self

        def cpu(self):
            return self

    models = {name: _Model() for name in ("cnn", "transformer", "boundary", "highres")}
    hashes = {name: f"hash-{name}" for name in models}

    def load_experts(cfg, splits, names=None):
        return models, hashes

    def load_controller(cfg, splits, expert_hashes):
        assert expert_hashes == hashes
        return _Model(), "controller-hash"

    def load_one(record, cfg):
        calls["load"] += 1
        assert record["patient_id"] == patient_id
        return patient

    def run_one(patient, models, controller, cfg, device, reasoning_engine=None):
        calls["run"] += 1
        predictions = {name: label.copy() for name in
                       ("cnn", "transformer", "boundary", "highres", "average", "vote", "final")}
        outputs = {name: {"probabilities": probabilities} for name in ("cnn", "transformer", "boundary")}
        debate = {"map": torch.zeros((1, 1, *shape)), "average_probabilities": probabilities}
        return predictions, outputs, debate, probabilities, {}

    def sliding(model, image, patch, device, overlap, mc_samples, proposal=None, max_patches=None):
        return {"probabilities": probabilities, "segmentation": segmentation}

    monkeypatch.setattr("brats_debate.evaluation.evaluate.load_experts", load_experts)
    monkeypatch.setattr("brats_debate.evaluation.evaluate.load_controller", load_controller)
    monkeypatch.setattr("brats_debate.evaluation.evaluate.load_patient", load_one)
    monkeypatch.setattr("brats_debate.evaluation.evaluate.run_patient", run_one)
    monkeypatch.setattr("brats_debate.evaluation.evaluate.sliding_predict", sliding)
    monkeypatch.setattr("brats_debate.evaluation.evaluate.save_nifti", lambda *args, **kwargs: None)
    cfg = {
        "seed": 1, "device": "cpu", "output_dir": str(tmp_path),
        "modalities": ["t1n"], "label_mapping": {0: 0, 1: 1, 2: 2, 3: 4},
        "regions": {"WT": [1, 2, 3], "TC": [1, 3], "ET": [3], "edema": [2]},
        "model": {"feature_channels": 4}, "crop_size": [4, 4, 4], "highres_patch_size": [2, 2, 2],
        "inference": {"overlap": 0.5, "mc_samples": 1, "uncertainty_threshold": 0.55,
                      "highres_max_patches": 128, "save_weights": False},
        "evaluation": {"hd95": True, "high_disagreement_threshold": 0.25},
    }
    splits = {"expert_train": [], "controller_train": [], "validation": [other_id], "test": [patient_id]}
    records = [{"patient_id": patient_id}, {"patient_id": other_id}]
    report_path = tmp_path / "evaluation" / "validation" / "report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps({
        "expert_hashes": hashes,
        "controller_hash": "controller-hash",
        "best_individual_expert_selected_on_validation": "boundary",
    }))
    first = evaluate(records, splits, cfg, split="test")
    assert calls == {"load": 1, "run": 1}
    assert first["patients"] == 1
    assert first["best_individual_expert_selected_on_validation"] == "boundary"
    case_path = tmp_path / "evaluation" / "test" / "cases" / f"{patient_id}.json"
    saved = json.loads(case_path.read_text())
    assert saved["format"] == CASE_FORMAT and saved["complete"] is True
    assert len(saved["rows"]) == 7
    second = evaluate(records, splits, cfg, split="test")
    assert calls == {"load": 1, "run": 1}
    assert second["patients"] == 1
    metrics = (tmp_path / "evaluation" / "test" / "metrics.csv").read_text().strip().splitlines()
    assert len(metrics) == 8
    saved["provenance"]["controller_hash"] = "stale"
    case_path.write_text(json.dumps(saved))
    evaluate(records, splits, cfg, split="test")
    assert calls == {"load": 2, "run": 2}
    incomplete = json.loads(case_path.read_text())
    incomplete["complete"] = False
    case_path.write_text(json.dumps(incomplete))
    assert _case_is_complete(case_path, patient_id, incomplete["provenance"]) is False
    with pytest.raises(ValueError):
        _case_path(tmp_path, "not-a-case")


def test_original_space_export(cfg, tmp_path):
    import nibabel as nib
    patient = load_patient(discover_patients(cfg)[0], cfg)
    output = tmp_path / "seg.nii.gz"
    save_nifti(output, patient["label"], patient, True, cfg)
    image = nib.load(output)
    np.testing.assert_allclose(image.affine, patient["affine"])
    assert image.shape == patient["label"].shape
    assert set(np.unique(image.get_fdata())) == {0, 1, 2, 4}
