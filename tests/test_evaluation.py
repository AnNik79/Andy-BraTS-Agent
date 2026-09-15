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


def test_original_space_export(cfg, tmp_path):
    import nibabel as nib
    patient = load_patient(discover_patients(cfg)[0], cfg)
    output = tmp_path / "seg.nii.gz"
    save_nifti(output, patient["label"], patient, True, cfg)
    image = nib.load(output)
    np.testing.assert_allclose(image.affine, patient["affine"])
    assert image.shape == patient["label"].shape
    assert set(np.unique(image.get_fdata())) == {0, 1, 2, 4}
