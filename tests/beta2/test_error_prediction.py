import json

import numpy as np
import pytest

from brats_debate.uncertainty.error_prediction import (
    LogisticErrorModel,
    PlattCalibrator,
    construct_error_target,
    error_prediction_metrics,
    evaluate_error_model,
    inverse_probability_weights,
    load_error_model_bundle,
    predict_risk_map,
    save_error_model_bundle,
)
from brats_debate.uncertainty.partitions import (
    make_subject_partitions,
    validate_subject_partitions,
)


FEATURE_NAMES = (
    "js_disagreement",
    "predictive_entropy",
    "boundary_probability",
    "boundary_distance",
    "mri_t1n",
    "mri_t1c",
    "mri_t2w",
    "mri_t2f",
)


def _synthetic(seed, samples, subject_prefix):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(samples, len(FEATURE_NAMES)))
    logit = 2.5 * X[:, 0] + 1.5 * X[:, 1] + 0.8 * X[:, 2] - 0.4
    probability = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, probability).astype(np.uint8)
    subjects = np.asarray(
        [f"{subject_prefix}_{index // 20}" for index in range(samples)]
    )
    return X, y, subjects


def test_voxel_error_target_and_alignment():
    truth = np.asarray([[[0, 1], [2, 3]]])
    prediction = truth.copy()
    prediction[0, 0, 1] = 0
    error = construct_error_target(prediction, truth)
    assert error.dtype == np.uint8
    assert error.sum() == 1
    assert error[0, 0, 1] == 1
    with pytest.raises(ValueError, match="align"):
        construct_error_target(prediction[..., :1], truth)
    with pytest.raises(ValueError, match="integer"):
        construct_error_target(prediction + 0.25, truth)


@pytest.mark.parametrize(
    "kind", ["disagreement_only", "entropy_only", "combined"]
)
def test_logistic_baselines_are_finite_and_respond_to_features(kind):
    X, y, subjects = _synthetic(1, 400, "fit")
    model = LogisticErrorModel(kind).fit(
        X,
        y,
        feature_names=FEATURE_NAMES,
        subject_ids=subjects,
    )
    probability = model.predict_proba(X)
    assert probability.shape == (400,)
    assert np.isfinite(probability).all()
    assert ((probability > 0) & (probability < 1)).all()
    if kind != "entropy_only":
        assert probability[X[:, 0] > 1].mean() > probability[X[:, 0] < -1].mean()


def test_platt_calibration_and_subject_isolation():
    X_fit, y_fit, fit_subjects = _synthetic(2, 300, "fit")
    X_cal, y_cal, calibration_subjects = _synthetic(3, 160, "cal")
    X_eval, y_eval, evaluation_subjects = _synthetic(4, 160, "eval")
    model = LogisticErrorModel("combined").fit(
        X_fit,
        y_fit,
        feature_names=FEATURE_NAMES,
        subject_ids=fit_subjects,
    )
    calibrator = PlattCalibrator().fit(
        model.decision_function(X_cal),
        y_cal,
        subject_ids=calibration_subjects,
        forbidden_subject_ids=model.fit_subjects_,
    )
    result = evaluate_error_model(
        model,
        X_eval,
        y_eval,
        subject_ids=evaluation_subjects,
        calibrator=calibrator,
    )
    assert result["calibrated"] is True
    assert 0 <= result["brier_score"] <= 1
    assert result["auroc"] is not None
    with pytest.raises(RuntimeError, match="leakage"):
        PlattCalibrator().fit(
            model.decision_function(X_fit),
            y_fit,
            subject_ids=fit_subjects,
            forbidden_subject_ids=model.fit_subjects_,
        )
    with pytest.raises(RuntimeError, match="leakage"):
        evaluate_error_model(
            model,
            X_cal,
            y_cal,
            subject_ids=calibration_subjects,
            calibrator=calibrator,
        )
    with pytest.raises(RuntimeError, match="never test"):
        LogisticErrorModel("combined").fit(
            X_fit,
            y_fit,
            feature_names=FEATURE_NAMES,
            subject_ids=fit_subjects,
            source_split="test",
        )
    with pytest.raises(ValueError, match="required"):
        PlattCalibrator().fit(
            model.decision_function(X_cal),
            y_cal,
            subject_ids=calibration_subjects,
            forbidden_subject_ids=[],
        )


def test_sampling_probability_weights_restore_population_prevalence():
    # Five of ten errors and ten of ninety correct voxels were sampled.
    y = np.r_[np.ones(5), np.zeros(10)]
    inclusion_probability = np.r_[
        np.full(5, 0.5), np.full(10, 10 / 90)
    ]
    weights = inverse_probability_weights(inclusion_probability)
    assert np.average(y, weights=weights) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="in \\(0,1\\]"):
        inverse_probability_weights([0, 1])


def test_metrics_handle_zero_rare_and_all_error_conditions():
    no_error = error_prediction_metrics(
        np.zeros(20, dtype=np.uint8), np.linspace(0.01, 0.2, 20)
    )
    assert no_error["auroc"] is None
    assert no_error["auprc"] is None
    assert np.isfinite(no_error["log_loss"])
    rare = np.zeros(20, dtype=np.uint8)
    rare[-1] = 1
    rare_result = error_prediction_metrics(rare, np.linspace(0.01, 0.9, 20))
    assert rare_result["auroc"] == pytest.approx(1.0)
    assert rare_result["auprc"] == pytest.approx(1.0)
    all_error = error_prediction_metrics(
        np.ones(20, dtype=np.uint8), np.linspace(0.8, 0.99, 20)
    )
    assert all_error["auroc"] is None
    assert all_error["auprc"] is None
    assert len(no_error["reliability"]["bins"]) == 10
    with pytest.raises(ValueError, match="binary"):
        error_prediction_metrics(np.full(20, 0.5), np.linspace(0.1, 0.9, 20))


def test_subject_grouped_partitions_keep_repeated_scans_together():
    case_to_subject = {
        "a0": "a",
        "a1": "a",
        "b0": "b",
        "c0": "c",
        "d0": "d",
        "e0": "e",
        "f0": "f",
    }
    manifest = make_subject_partitions(
        case_to_subject,
        list(case_to_subject),
        forbidden_case_ids=["test_case"],
        seed=7,
    )
    validate_subject_partitions(
        manifest, case_to_subject, forbidden_case_ids=["test_case"]
    )
    owners = {
        case: role
        for role, cases in manifest["partitions"].items()
        for case in cases
    }
    assert owners["a0"] == owners["a1"]
    assert sum(manifest["case_counts"].values()) == len(case_to_subject)
    with pytest.raises(RuntimeError, match="test cases"):
        make_subject_partitions(
            {**case_to_subject, "test_case": "t"},
            [*case_to_subject, "test_case"],
            forbidden_case_ids=["test_case"],
        )
    with pytest.raises(RuntimeError, match="test subjects"):
        make_subject_partitions(
            {**case_to_subject, "test_scan": "a"},
            list(case_to_subject),
            forbidden_case_ids=["test_scan"],
        )


def test_model_serialization_and_risk_map_round_trip(tmp_path):
    X, y, subjects = _synthetic(5, 300, "fit")
    model = LogisticErrorModel("combined").fit(
        X, y, feature_names=FEATURE_NAMES, subject_ids=subjects
    )
    with pytest.raises(ValueError, match="column count"):
        LogisticErrorModel("combined").fit(
            np.c_[X, np.ones(len(X))],
            y,
            feature_names=FEATURE_NAMES,
            subject_ids=subjects,
        )
    provenance = {"split_hash": "abc", "source_split": "validation"}
    path = tmp_path / "bundle.json"
    save_error_model_bundle(
        path, {"combined": model}, provenance=provenance
    )
    loaded = load_error_model_bundle(
        path, expected_provenance=provenance
    )
    np.testing.assert_allclose(
        loaded["models"]["combined"].predict_proba(X),
        model.predict_proba(X),
    )
    payload = json.loads(path.read_text())
    payload["provenance"]["split_hash"] = "tampered"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="fingerprint"):
        load_error_model_bundle(path)

    shape = (3, 4, 5)
    columns = {
        name: X[: np.prod(shape), index].reshape(shape)
        for index, name in enumerate(FEATURE_NAMES)
    }
    risk = predict_risk_map(model, columns)
    assert risk["error_risk"].shape == shape
    assert np.isfinite(risk["error_risk"]).all()
    np.testing.assert_allclose(
        risk["error_risk"] + risk["reliability"], 1, atol=1e-6
    )
    assert risk["calibrated"] is False
