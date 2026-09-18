import json
from pathlib import Path
import numpy as np
import pytest
import torch
from brats_debate.analysis.active_learning import unlabeled_region_priority
from brats_debate.analysis.correlation import error_correlation_matrix, export_matrices, prediction_similarity_matrix
from brats_debate.analysis.difficulty import QUADRANT_D, novelty_disagreement_quadrant
from brats_debate.analysis.error_maps import segmentation_error_maps
from brats_debate.analysis.extract import FROZEN_CNN_BEST, FROZEN_CNN_EPOCH, FROZEN_CNN_MEAN_DICE, analyze_segmentation_expert, load_frozen_cnn, representation_spec
from brats_debate.analysis.mahalanobis import ExpertMahalanobis
from brats_debate.analysis.pairwise import SEGMENTATION_PAIRS, pairwise_expert_analysis
from brats_debate.analysis.pooling import pool_patch, pool_patient, pool_region
from brats_debate.analysis.schema import REGION_RECORD_FIELDS, attach_spatial_identity, empty_region_record, modality_statistics
from brats_debate.analysis.svd import latent_svd
from brats_debate.analysis.tensor_layout import ANALYSIS_AXES, tagged_feature_vector
from brats_debate.config import SEGMENTATION_EXPERTS, load_config
from brats_debate.controller.gating_network import GatingNetwork
from brats_debate.debate.disagreement import analyze_disagreement
from brats_debate.debate.features import controller_features, feature_channels, stack_evidence
from brats_debate.experts import build_expert
from brats_debate.experts.base import pack_output
from brats_debate.reasoning.reasoning_engine import LLMReasoningExpert

REPO = Path(__file__).resolve().parents[1]
CURVE4 = REPO / "configs/brats_cnn_curve4.yaml"
CASES = REPO / "configs/splits/brats2023_gli_cases.json"
SPLIT_KEYS = ("expert_train", "controller_train", "validation", "test")


def _fake_output(segmentation, classes=4):
    volume = torch.as_tensor(segmentation)[None]
    logits = torch.zeros(1, classes, *volume.shape[-3:])
    logits[0].scatter_(0, volume.long(), 10.0)
    compact = torch.zeros(1, 2, *volume.shape[-3:])
    return pack_output(logits, compact)


def test_every_expert_exposes_analysis_representation(cfg):
    image = torch.randn(1, 4, 8, 8, 8)
    for name in SEGMENTATION_EXPERTS:
        model = build_expert(name, cfg).eval()
        spec = representation_spec(model)
        assert spec.name
        assert spec.capture in {"input", "output"}
        result = model.analyze(image)
        assert result["logits"].shape == (1, 4, 8, 8, 8)
        assert result["probabilities"].shape == (1, 4, 8, 8, 8)
        assert result["segmentation"].shape == (1, 8, 8, 8)
        assert result["confidence"].shape[1] == 1
        assert result["uncertainty"].shape[1] == 1
        assert result["spatial_representation"].ndim == 5
        assert result["spatial_shape"] == tuple(result["spatial_representation"].shape)
        assert result["pooled_patch"].shape == (1, result["pooled_dim"])
        assert result["pooled_dim"] == result["spatial_representation"].shape[1]


def test_pooled_dimensions_are_deterministic(cfg):
    image = torch.randn(1, 4, 8, 8, 8)
    for name in SEGMENTATION_EXPERTS:
        model = build_expert(name, cfg).eval()
        first = analyze_segmentation_expert(model, image)
        second = analyze_segmentation_expert(model, image)
        torch.testing.assert_close(first["pooled_patch"], second["pooled_patch"])
        torch.testing.assert_close(first["spatial_representation"], second["spatial_representation"])


def test_spatial_representations_keep_coordinates(cfg):
    image = torch.randn(1, 4, 16, 16, 16)
    shapes = {}
    for name in SEGMENTATION_EXPERTS:
        result = build_expert(name, cfg).eval().analyze(image)
        spatial = result["spatial_representation"]
        assert spatial.shape[0] == 1
        assert len(spatial.shape) == 5
        shapes[name] = result["spatial_shape"]
        pooled = pool_region(spatial, torch.ones(16, 16, 16))
        assert pooled.shape == (1, spatial.shape[1])
    assert shapes["cnn"][2:] != (16, 16, 16)  # bottleneck is downsampled
    assert shapes["transformer"][2:] != (16, 16, 16)
    assert shapes["boundary"][2:] == (16, 16, 16)
    assert shapes["highres"][2:] == (16, 16, 16)


def test_representation_extraction_does_not_change_prediction(cfg):
    image = torch.randn(1, 4, 8, 8, 8)
    for name in SEGMENTATION_EXPERTS:
        model = build_expert(name, cfg).eval()
        with torch.no_grad():
            before = model(image)
            analyzed = model.analyze(image)
            after = model(image)
        torch.testing.assert_close(before["logits"], after["logits"])
        torch.testing.assert_close(before["logits"], analyzed["logits"])
        torch.testing.assert_close(before["probabilities"], analyzed["probabilities"])
        torch.testing.assert_close(before["segmentation"].float(), analyzed["segmentation"].float())


@pytest.mark.skipif(not FROZEN_CNN_BEST.exists(), reason="frozen CNN checkpoint not present")
def test_frozen_cnn_epoch15_loads_and_is_observational():
    cfg = load_config(CURVE4)
    model, checkpoint = load_frozen_cnn(cfg, FROZEN_CNN_BEST)
    assert checkpoint["name"] == "cnn"
    assert checkpoint["completed_epoch"] == FROZEN_CNN_EPOCH
    assert checkpoint["best_validation_metric"] == pytest.approx(FROZEN_CNN_MEAN_DICE)
    image = torch.randn(1, 4, 16, 16, 16)
    with torch.no_grad():
        before = model(image)
        analyzed = model.analyze(image)
        after = model(image)
    torch.testing.assert_close(before["logits"], after["logits"])
    torch.testing.assert_close(before["logits"], analyzed["logits"])
    assert analyzed["representation_name"] == "encoder_bottleneck"
    assert analyzed["spatial_shape"][1] == 192


def test_segmentation_error_maps_identify_fp_fn_and_brats_regions():
    target = np.zeros((6, 6, 6), np.int16)
    target[1:5, 1:5, 1:5] = 2
    target[2:4, 2:4, 2:4] = 1
    target[2:4, 2:4, 2] = 3
    pred = target.copy()
    pred[0, 0, 0] = 2
    pred[2, 2, 2] = 0
    maps = segmentation_error_maps(pred, target)
    assert maps["correct"][1, 1, 1]
    assert maps["error"][0, 0, 0]
    assert maps["false_positive"][0, 0, 0]
    assert not maps["false_negative"][0, 0, 0]
    assert maps["false_negative"][2, 2, 2]
    assert maps["class_error"][2][0, 0, 0]
    assert maps["class_false_positive"][2][0, 0, 0]
    assert maps["region_error"]["ET"][2, 2, 2]
    assert maps["region_error"]["TC"][2, 2, 2]
    assert maps["region_error"]["WT"][2, 2, 2]
    assert maps["boundary_error"].shape == target.shape
    assert maps["patient_summary"]["false_positive_voxels"] == 1


def test_pairwise_disagreement_helper_and_required_pairs(cfg):
    image = torch.randn(1, 4, 8, 8, 8)
    outputs = {name: build_expert(name, cfg).eval()(image) for name in SEGMENTATION_EXPERTS}
    debate = analyze_disagreement(outputs)
    analysis = pairwise_expert_analysis(outputs, debate=debate)
    assert set(analysis["pairs"]) == {f"{a}__{b}" for a, b in SEGMENTATION_PAIRS}
    assert len(SEGMENTATION_PAIRS) == 6
    pair = analysis["pairs"]["cnn__transformer"]
    assert pair["hard_disagreement"].shape == (8, 8, 8)
    assert pair["pairwise_tv"].shape[1] == 1
    assert pair["pairwise_js"].shape[1] == 1
    assert "WT" in pair["region_hard_disagreement"]
    torch.testing.assert_close(debate["js_divergence"], analysis["production"]["js_divergence"])


def test_error_correlation_is_not_prediction_similarity():
    target = np.zeros((4, 4, 4), np.int16)
    target[:, :, 2:] = 1
    missed = np.zeros_like(target)
    extra = np.ones_like(target)
    predictions = {
        "cnn": _fake_output(missed),
        "transformer": _fake_output(missed),
        "boundary": _fake_output(extra),
        "highres": _fake_output(extra),
    }
    errors = error_correlation_matrix(predictions, target)
    preds = prediction_similarity_matrix(predictions)
    assert errors["kind"] == "error_correlation"
    assert preds["kind"] == "prediction_similarity"
    assert errors["matrices"]["error_phi"]["cnn"]["transformer"] == pytest.approx(1.0)
    assert errors["matrices"]["error_phi"]["cnn"]["boundary"] < 1
    assert preds["matrices"]["hard_agreement_fraction"]["cnn"]["transformer"] == pytest.approx(1.0)
    assert errors["matrices"]["error_jaccard"]["cnn"]["cnn"] == 1.0


def test_export_matrices_writes_csv_and_json(tmp_path):
    target = np.zeros((3, 3, 3), np.int16)
    predictions = {name: _fake_output(target) for name in SEGMENTATION_EXPERTS}
    payload = error_correlation_matrix(predictions, target)
    paths = export_matrices(payload, tmp_path, "error")
    assert paths["json"].exists()
    assert (tmp_path / "error_error_phi.csv").exists()
    rows = json.loads(paths["long_form"].read_text())
    assert rows[0]["kind"] == "error_correlation"


def test_svd_synthetic_matrix():
    matrix = np.array([[1.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.9, 0.0]], dtype=np.float64)
    result = latent_svd(matrix, column_names=["cnn", "transformer", "boundary"])
    assert len(result["singular_values"]) == 3
    assert result["explained_energy"][0] == pytest.approx(max(result["explained_energy"]))
    assert pytest.approx(sum(result["explained_energy"])) == 1
    assert "cnn" in result["loadings"][0]
    assert 1 in result["reconstruction_mse_by_rank"]
    assert result["kind"] == "exploratory_latent_svd"


def test_mahalanobis_inliers_outliers_and_split_guard():
    rng = np.random.default_rng(0)
    train = rng.normal(size=(60, 6))
    model = ExpertMahalanobis("cnn").fit(train, "expert_train")
    inliers = model.distance(train[:12])
    outliers = model.distance(train[:12] + 10)
    assert outliers.mean() > inliers.mean()
    with pytest.raises(ValueError, match="expert_train"):
        ExpertMahalanobis("cnn").fit(train, "validation")
    with pytest.raises(ValueError, match="expert_train"):
        ExpertMahalanobis("cnn").fit(train, "test")
    with pytest.raises(ValueError, match="expert_train"):
        ExpertMahalanobis("cnn").fit(train, "controller_train")
    with pytest.raises(ValueError, match="not expert_train"):
        ExpertMahalanobis("cnn").fit(train, "expert_train", patient_ids=["p"],
                                     patient_split_lookup={"p": "validation"})


def test_region_records_preserve_identity_and_coordinates():
    record = empty_region_record("BraTS-GLI-00000-000", "r0", expert_identity="cnn")
    attach_spatial_identity(record, origin=(2, 4, 6), shape=(3, 5, 7), spacing=(1, 1, 1))
    image = np.zeros((4, 8, 8, 8))
    image[0] = 1
    stats = modality_statistics(image, np.ones((8, 8, 8), bool))
    record["T1n_region_statistics"] = stats["t1n"]
    record["mri_modality_identity"] = "t1n"
    assert record["patient_id"] == "BraTS-GLI-00000-000"
    assert record["expert_identity"] == "cnn"
    assert record["x_start"] == 6 and record["x_end"] == 13
    assert record["spatial_coordinates"]["origin_zyx"] == [2, 4, 6]
    assert "t1n" in stats and "t2f" in stats
    tagged = tagged_feature_vector([0.1, 0.2], patient_id=record["patient_id"],
                                   region_id="r0", expert="cnn", modality="t1c")
    assert tagged["axes"] == ANALYSIS_AXES
    assert tagged["expert"] == "cnn"
    assert tagged["mri_modality"] == "t1c"
    assert set(REGION_RECORD_FIELDS) >= {"CNN_mahalanobis", "active_learning_priority"}


def test_low_disagreement_high_novelty_is_preserved():
    result = novelty_disagreement_quadrant(np.array([0.0, 0.9]), np.array([0.9, 0.9]), 0.5, 0.5)
    assert result["quadrant"][0] == QUADRANT_D
    assert "not imply" in result["note"]


def test_active_learning_refuses_ground_truth():
    score = unlabeled_region_priority(disagreement=[0.1, 0.9], uncertainty=[0.2, 0.8], novelty=[0.0, 1.0])
    assert score["uses_ground_truth"] is False
    assert score["priority"][1] > score["priority"][0]
    with pytest.raises(ValueError, match="ground truth"):
        unlabeled_region_priority(disagreement=[0.1], uncertainty=[0.1], novelty=[0.1],
                                  ground_truth=np.array([1]))


def test_canonical_splits_untouched():
    cases = json.loads(CASES.read_text())
    assert tuple(cases) == SPLIT_KEYS
    assert len(cases["expert_train"]) == 682
    assert len(cases["controller_train"]) == 248
    assert len(cases["validation"]) == 191
    assert len(cases["test"]) == 130


def test_production_controller_still_has_65_channels():
    cfg = load_config(REPO / "configs/brats.yaml")
    assert feature_channels(cfg) == 65


def test_controller_still_four_expert_weights(cfg):
    image = torch.randn(1, 4, 8, 8, 8)
    outputs = {name: build_expert(name, cfg).eval()(image) for name in SEGMENTATION_EXPERTS}
    debate = analyze_disagreement(outputs)
    features = controller_features(image, outputs, debate)
    assert features.shape[1] == feature_channels(cfg)
    p, availability = stack_evidence(outputs)
    result = GatingNetwork(features.shape[1], 4)(features, p, availability)
    assert result["weights"].shape[1] == 4
    fused = (p * result["weights"].unsqueeze(2)).sum(1)
    torch.testing.assert_close(result["probabilities"], fused, atol=1e-5, rtol=1e-5)


def test_llm_still_cannot_modify_segmentation(cfg):
    result = LLMReasoningExpert.from_config(cfg).reason({
        "patient_id": "synthetic_000",
        "disagreement_regions": [],
        "interpretation_limit": "not medical advice",
    })
    assert result.can_modify_segmentation is False


def test_pool_patient_mean():
    vectors = [torch.tensor([1.0, 3.0]), torch.tensor([3.0, 1.0])]
    torch.testing.assert_close(pool_patient(vectors), torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(pool_patch(torch.ones(2, 3, 4, 4, 4)), torch.ones(2, 3))
