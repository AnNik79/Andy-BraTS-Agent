"""Small, calibrated logistic models for frozen-controller voxel errors."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from ..config import fingerprint


ERROR_MODEL_FORMAT = "beta2_logistic_error_model_v1"
ERROR_BUNDLE_FORMAT = "beta2_error_model_bundle_v1"
BASELINE_KINDS = ("disagreement_only", "entropy_only", "combined")
DISAGREEMENT_FEATURE = "js_disagreement"
ENTROPY_FEATURE = "predictive_entropy"


def _numpy(value, *, dtype=None) -> np.ndarray:
    try:
        import torch

        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value, dtype=dtype)


def construct_error_target(prediction, target) -> np.ndarray:
    """Return ``E(v)=1[prediction(v) != target(v)]`` as uint8."""

    pred, truth = _numpy(prediction), _numpy(target)
    while pred.ndim > truth.ndim and pred.shape[0] == 1:
        pred = pred[0]
    while truth.ndim > pred.ndim and truth.shape[0] == 1:
        truth = truth[0]
    if pred.shape != truth.shape:
        raise ValueError("controller prediction and ground truth must align")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("prediction and target labels must be finite")
    if not np.array_equal(pred, np.rint(pred)) or not np.array_equal(
        truth, np.rint(truth)
    ):
        raise ValueError("prediction and target must contain integer labels")
    return (pred.astype(np.int64) != truth.astype(np.int64)).astype(np.uint8)


def feature_matrix(
    columns: Mapping[str, object],
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[int, ...]]:
    """Flatten aligned scalar feature maps into ``[voxels, features]``."""

    names = tuple(columns) if feature_names is None else tuple(feature_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("feature names must be unique and nonempty")
    missing = [name for name in names if name not in columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing}")
    arrays, shape = [], None
    for name in names:
        value = _numpy(columns[name], dtype=np.float32)
        while value.ndim > 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 3:
            raise ValueError(f"feature {name} must be a scalar 3D map")
        if shape is None:
            shape = value.shape
        elif value.shape != shape:
            raise ValueError("all feature maps must share the voxel grid")
        if not np.isfinite(value).all():
            raise ValueError(f"feature {name} contains non-finite values")
        arrays.append(value.reshape(-1))
    return np.stack(arrays, axis=1), names, tuple(shape)


def inverse_probability_weights(
    inclusion_probability, *, normalize: bool = True
) -> np.ndarray:
    probability = _numpy(inclusion_probability, dtype=np.float64).reshape(-1)
    if (
        probability.size == 0
        or not np.isfinite(probability).all()
        or (probability <= 0).any()
        or (probability > 1).any()
    ):
        raise ValueError("inclusion probabilities must be finite and in (0,1]")
    weights = 1.0 / probability
    if normalize:
        weights /= weights.mean()
    return weights


def _validate_binary_data(
    X,
    y,
    sample_weight,
    subject_ids,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    matrix = _numpy(X, dtype=np.float64)
    target = _numpy(y).reshape(-1)
    subjects = _numpy(subject_ids).astype(str).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != target.size:
        raise ValueError("X must be [samples,features] and align with y")
    if subjects.size != target.size:
        raise ValueError("one subject ID is required per sample")
    if not np.isfinite(matrix).all():
        raise ValueError("model features must be finite")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("error targets must be binary")
    weights = (
        np.ones(target.size, dtype=np.float64)
        if sample_weight is None
        else _numpy(sample_weight, dtype=np.float64).reshape(-1)
    )
    if (
        weights.size != target.size
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("sample weights must be positive, finite, and aligned")
    return matrix, target.astype(np.uint8), weights, subjects


def _selected_features(
    feature_names: Sequence[str], kind: str
) -> tuple[tuple[str, ...], np.ndarray]:
    names = tuple(feature_names)
    if kind not in BASELINE_KINDS:
        raise ValueError(f"kind must be one of {BASELINE_KINDS}")
    if kind == "disagreement_only":
        selected = (DISAGREEMENT_FEATURE,)
    elif kind == "entropy_only":
        selected = (ENTROPY_FEATURE,)
    else:
        required = {
            DISAGREEMENT_FEATURE,
            ENTROPY_FEATURE,
            "boundary_probability",
            "boundary_distance",
        }
        missing = required - set(names)
        if missing or not any(name.startswith("mri_") for name in names):
            raise ValueError(
                "combined model requires D, H, two boundary features, and MRI"
            )
        selected = names
    missing = [name for name in selected if name not in names]
    if missing:
        raise ValueError(f"missing baseline features: {missing}")
    return selected, np.asarray([names.index(name) for name in selected], dtype=int)


@dataclass
class LogisticErrorModel:
    """Regularized logistic baseline with a versioned feature schema."""

    kind: str
    regularization_c: float = 1.0
    max_iter: int = 500

    def fit(
        self,
        X,
        y,
        *,
        feature_names: Sequence[str],
        subject_ids,
        sample_weight=None,
        source_split: str = "validation",
        partition_role: str = "fit",
    ) -> "LogisticErrorModel":
        if source_split == "test" or partition_role != "fit":
            raise RuntimeError("error models may fit only the fit partition, never test")
        matrix, target, weights, subjects = _validate_binary_data(
            X, y, sample_weight, subject_ids
        )
        if matrix.shape[1] != len(feature_names):
            raise ValueError("X column count must match feature_names")
        if np.unique(target).size != 2:
            raise ValueError("logistic fitting requires both error classes")
        if self.regularization_c <= 0:
            raise ValueError("regularization_c must be positive")
        selected_names, indices = _selected_features(feature_names, self.kind)
        selected = matrix[:, indices]
        scaler = StandardScaler().fit(selected, sample_weight=weights)
        scaled = scaler.transform(selected)
        estimator = LogisticRegression(
            C=self.regularization_c,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=0,
        ).fit(scaled, target, sample_weight=weights)
        self.feature_names_ = tuple(feature_names)
        self.selected_feature_names_ = selected_names
        self.selected_indices_ = indices
        self.mean_ = np.asarray(scaler.mean_, dtype=np.float64)
        self.scale_ = np.asarray(scaler.scale_, dtype=np.float64)
        self.coef_ = np.asarray(estimator.coef_[0], dtype=np.float64)
        self.intercept_ = float(estimator.intercept_[0])
        self.fit_subjects_ = tuple(sorted(set(subjects.tolist())))
        self.source_split_ = source_split
        self.partition_role_ = partition_role
        self.weighted_error_prevalence_ = float(
            np.average(target, weights=weights)
        )
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "coef_"):
            raise RuntimeError("error model is not fitted")

    def decision_function(self, X) -> np.ndarray:
        self._check_fitted()
        matrix = _numpy(X, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names_):
            raise ValueError("prediction features do not match the fitted schema")
        if not np.isfinite(matrix).all():
            raise ValueError("prediction features must be finite")
        selected = matrix[:, self.selected_indices_]
        scaled = (selected - self.mean_) / self.scale_
        return scaled @ self.coef_ + self.intercept_

    def predict_proba(self, X) -> np.ndarray:
        return expit(self.decision_function(X))

    def to_dict(self) -> dict[str, object]:
        self._check_fitted()
        return {
            "format": ERROR_MODEL_FORMAT,
            "kind": self.kind,
            "regularization_c": self.regularization_c,
            "max_iter": self.max_iter,
            "feature_names": list(self.feature_names_),
            "selected_feature_names": list(self.selected_feature_names_),
            "selected_indices": self.selected_indices_.tolist(),
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "coef": self.coef_.tolist(),
            "intercept": self.intercept_,
            "fit_subjects": list(self.fit_subjects_),
            "source_split": self.source_split_,
            "partition_role": self.partition_role_,
            "weighted_error_prevalence": self.weighted_error_prevalence_,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LogisticErrorModel":
        if payload.get("format") != ERROR_MODEL_FORMAT:
            raise ValueError("unknown logistic error-model format")
        model = cls(
            str(payload["kind"]),
            float(payload["regularization_c"]),
            int(payload["max_iter"]),
        )
        model.feature_names_ = tuple(payload["feature_names"])
        model.selected_feature_names_ = tuple(payload["selected_feature_names"])
        model.selected_indices_ = np.asarray(payload["selected_indices"], dtype=int)
        model.mean_ = np.asarray(payload["mean"], dtype=np.float64)
        model.scale_ = np.asarray(payload["scale"], dtype=np.float64)
        model.coef_ = np.asarray(payload["coef"], dtype=np.float64)
        model.intercept_ = float(payload["intercept"])
        model.fit_subjects_ = tuple(payload["fit_subjects"])
        model.source_split_ = str(payload["source_split"])
        model.partition_role_ = str(payload["partition_role"])
        model.weighted_error_prevalence_ = float(
            payload["weighted_error_prevalence"]
        )
        return model


@dataclass
class PlattCalibrator:
    """Logistic calibration of a frozen model's scalar logit."""

    def fit(
        self,
        scores,
        y,
        *,
        subject_ids,
        forbidden_subject_ids: Sequence[str],
        sample_weight=None,
        source_split: str = "validation",
        partition_role: str = "calibration",
    ) -> "PlattCalibrator":
        if source_split == "test" or partition_role != "calibration":
            raise RuntimeError("calibration may use only the calibration partition")
        matrix, target, weights, subjects = _validate_binary_data(
            _numpy(scores, dtype=np.float64).reshape(-1, 1),
            y,
            sample_weight,
            subject_ids,
        )
        if np.unique(target).size != 2:
            raise ValueError("Platt calibration requires both error classes")
        forbidden = set(str(value) for value in forbidden_subject_ids)
        if not forbidden:
            raise ValueError("fit subject IDs are required for calibration isolation")
        overlap = set(subjects.tolist()) & forbidden
        if overlap:
            raise RuntimeError("subject leakage between model fit and calibration")
        estimator = LogisticRegression(
            C=1e6, solver="lbfgs", max_iter=500, random_state=0
        ).fit(matrix, target, sample_weight=weights)
        self.coef_ = float(estimator.coef_[0, 0])
        self.intercept_ = float(estimator.intercept_[0])
        self.calibration_subjects_ = tuple(sorted(set(subjects.tolist())))
        self.checked_fit_subjects_ = tuple(sorted(forbidden))
        self.source_split_ = source_split
        return self

    def predict_proba(self, scores) -> np.ndarray:
        if not hasattr(self, "coef_"):
            raise RuntimeError("calibrator is not fitted")
        values = _numpy(scores, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("calibration scores must be finite")
        return expit(self.coef_ * values + self.intercept_)

    def to_dict(self) -> dict[str, object]:
        if not hasattr(self, "coef_"):
            raise RuntimeError("calibrator is not fitted")
        return {
            "method": "platt",
            "coef": self.coef_,
            "intercept": self.intercept_,
            "calibration_subjects": list(self.calibration_subjects_),
            "checked_fit_subjects": list(self.checked_fit_subjects_),
            "source_split": self.source_split_,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PlattCalibrator":
        if payload.get("method") != "platt":
            raise ValueError("unknown calibration method")
        calibrator = cls()
        calibrator.coef_ = float(payload["coef"])
        calibrator.intercept_ = float(payload["intercept"])
        calibrator.calibration_subjects_ = tuple(payload["calibration_subjects"])
        calibrator.checked_fit_subjects_ = tuple(payload["checked_fit_subjects"])
        calibrator.source_split_ = str(payload["source_split"])
        return calibrator


def reliability_summary(
    y_true,
    probability,
    *,
    sample_weight=None,
    bins: int = 10,
) -> dict[str, object]:
    target = _numpy(y_true).reshape(-1)
    predicted = _numpy(probability, dtype=np.float64).reshape(-1)
    if target.size != predicted.size or target.size == 0:
        raise ValueError("truth and probability must be aligned and nonempty")
    if not np.isin(target, [0, 1]).all():
        raise ValueError("truth must be binary")
    if (
        not np.isfinite(predicted).all()
        or (predicted < 0).any()
        or (predicted > 1).any()
    ):
        raise ValueError("probabilities must be finite and in [0,1]")
    weights = (
        np.ones(target.size, dtype=np.float64)
        if sample_weight is None
        else _numpy(sample_weight, dtype=np.float64).reshape(-1)
    )
    if (
        bins < 1
        or weights.size != target.size
        or not np.isfinite(weights).all()
        or (weights <= 0).any()
    ):
        raise ValueError("reliability weights must be positive and aligned")
    edges = np.linspace(0, 1, bins + 1)
    assignment = np.minimum(np.searchsorted(edges, predicted, side="right") - 1, bins - 1)
    assignment = np.maximum(assignment, 0)
    rows, total_weight, ece, mce = [], float(weights.sum()), 0.0, 0.0
    for index in range(bins):
        mask = assignment == index
        weight = float(weights[mask].sum())
        mean_probability = (
            float(np.average(predicted[mask], weights=weights[mask]))
            if mask.any()
            else None
        )
        observed_error = (
            float(np.average(target[mask], weights=weights[mask]))
            if mask.any()
            else None
        )
        gap = (
            None
            if mean_probability is None
            else abs(mean_probability - observed_error)
        )
        if gap is not None:
            ece += weight / total_weight * gap
            mce = max(mce, gap)
        rows.append(
            {
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(mask.sum()),
                "weight": weight,
                "mean_probability": mean_probability,
                "observed_error_fraction": observed_error,
                "absolute_gap": gap,
            }
        )
    return {
        "bins": rows,
        "expected_calibration_error": float(ece),
        "maximum_calibration_error": float(mce),
    }


def error_prediction_metrics(
    y_true,
    probability,
    *,
    sample_weight=None,
    reliability_bins: int = 10,
) -> dict[str, object]:
    raw_target = _numpy(y_true).reshape(-1)
    if (
        raw_target.size == 0
        or not np.isfinite(raw_target).all()
        or not np.isin(raw_target, [0, 1]).all()
    ):
        raise ValueError("error targets must be finite, nonempty, and binary")
    target = raw_target.astype(np.uint8)
    predicted = _numpy(probability, dtype=np.float64).reshape(-1)
    weights = (
        None
        if sample_weight is None
        else _numpy(sample_weight, dtype=np.float64).reshape(-1)
    )
    reliability = reliability_summary(
        target, predicted, sample_weight=weights, bins=reliability_bins
    )
    classes = np.unique(target)
    clipped = np.clip(predicted, 1e-7, 1 - 1e-7)
    return {
        "samples": int(target.size),
        "weighted_error_prevalence": float(
            np.average(
                target,
                weights=(
                    np.ones(target.size, dtype=np.float64)
                    if weights is None
                    else weights
                ),
            )
        ),
        "auroc": (
            float(roc_auc_score(target, predicted, sample_weight=weights))
            if classes.size == 2
            else None
        ),
        "auprc": (
            float(average_precision_score(target, predicted, sample_weight=weights))
            if classes.size == 2
            else None
        ),
        "brier_score": float(
            brier_score_loss(target, predicted, sample_weight=weights)
        ),
        "log_loss": float(
            log_loss(target, clipped, sample_weight=weights, labels=[0, 1])
        ),
        "reliability": reliability,
    }


def evaluate_error_model(
    model: LogisticErrorModel,
    X,
    y,
    *,
    subject_ids,
    sample_weight=None,
    calibrator: PlattCalibrator | None = None,
    source_split: str = "validation",
    partition_role: str = "evaluation",
) -> dict[str, object]:
    if source_split == "test" or partition_role != "evaluation":
        raise RuntimeError("evaluation must use the evaluation partition, never test")
    subjects = set(_numpy(subject_ids).astype(str).reshape(-1).tolist())
    forbidden = set(model.fit_subjects_)
    if calibrator is not None:
        if set(calibrator.checked_fit_subjects_) != set(model.fit_subjects_):
            raise RuntimeError("calibrator was not isolated from this model's fit subjects")
        if set(calibrator.calibration_subjects_) & set(model.fit_subjects_):
            raise RuntimeError("subject leakage between model fit and calibration")
        forbidden |= set(calibrator.calibration_subjects_)
    if subjects & forbidden:
        raise RuntimeError("subject leakage into error-model evaluation")
    scores = model.decision_function(X)
    probability = (
        model.predict_proba(X)
        if calibrator is None
        else calibrator.predict_proba(scores)
    )
    result = error_prediction_metrics(
        y, probability, sample_weight=sample_weight
    )
    result.update(
        {
            "model_kind": model.kind,
            "calibrated": calibrator is not None,
            "source_split": source_split,
            "partition_role": partition_role,
            "evaluation_subjects": sorted(subjects),
        }
    )
    return result


def predict_risk_map(
    model: LogisticErrorModel,
    feature_columns: Mapping[str, object],
    *,
    calibrator: PlattCalibrator | None = None,
) -> dict[str, object]:
    matrix, names, shape = feature_matrix(
        feature_columns, model.feature_names_
    )
    if names != model.feature_names_:
        raise ValueError("risk-map features do not match the fitted model schema")
    scores = model.decision_function(matrix)
    risk = (
        model.predict_proba(matrix)
        if calibrator is None
        else calibrator.predict_proba(scores)
    ).reshape(shape)
    return {
        "error_risk": risk.astype(np.float32),
        "reliability": (1 - risk).astype(np.float32),
        "calibrated": calibrator is not None,
        "label": (
            "calibrated_error_probability"
            if calibrator is not None
            else "uncalibrated_logistic_error_score"
        ),
    }


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def save_error_model_bundle(
    path,
    models: Mapping[str, LogisticErrorModel],
    *,
    calibrators: Mapping[str, PlattCalibrator | None] | None = None,
    provenance: Mapping[str, object],
) -> None:
    if set(models) - set(BASELINE_KINDS):
        raise ValueError("model bundle contains an unknown baseline")
    calibrators = calibrators or {}
    payload = {
        "format": ERROR_BUNDLE_FORMAT,
        "complete": True,
        "provenance": dict(provenance),
        "provenance_hash": fingerprint(provenance),
        "models": {name: model.to_dict() for name, model in models.items()},
        "calibrators": {
            name: None if value is None else value.to_dict()
            for name, value in calibrators.items()
        },
    }
    _atomic_json(Path(path), payload)


def load_error_model_bundle(
    path,
    *,
    expected_provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != ERROR_BUNDLE_FORMAT or not payload.get("complete"):
        raise ValueError("incomplete or unknown error-model bundle")
    if payload.get("provenance_hash") != fingerprint(payload["provenance"]):
        raise ValueError("error-model provenance fingerprint is invalid")
    if expected_provenance is not None and payload["provenance"] != dict(
        expected_provenance
    ):
        raise ValueError("error-model provenance does not match")
    return {
        "provenance": payload["provenance"],
        "models": {
            name: LogisticErrorModel.from_dict(value)
            for name, value in payload["models"].items()
        },
        "calibrators": {
            name: None if value is None else PlattCalibrator.from_dict(value)
            for name, value in payload["calibrators"].items()
        },
    }
