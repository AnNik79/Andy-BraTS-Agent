"""Mahalanobis novelty relative to an expert's expert_train representation.

Distance from the training representation distribution is not automatically
tumor, pathology, segmentation error, or OOD disease. Those links need later
empirical tests.

Default covariance: PCA to at most 32 components, then Ledoit-Wolf shrinkage
in the reduced space. Full unregularized covariance is unstable when the
pooled feature dimension approaches or exceeds the number of fit samples.
"""
import numpy as np
from sklearn.covariance import LedoitWolf

FIT_SPLIT = "expert_train"
FORBIDDEN_FIT_SPLITS = frozenset({"validation", "test", "controller_train"})
MAX_PCA_COMPONENTS = 32


class ExpertMahalanobis:
    def __init__(self, expert_name, max_components=MAX_PCA_COMPONENTS):
        self.expert_name = expert_name
        self.max_components = max_components
        self.fitted_split = None
        self.mean_ = None
        self.components_ = None
        self.covariance_ = None

    def fit(self, vectors, split, patient_ids=None, patient_split_lookup=None):
        if split in FORBIDDEN_FIT_SPLITS or split != FIT_SPLIT:
            raise ValueError("Mahalanobis statistics may be fit only on expert_train; "
                             f"refusing split={split!r}")
        if patient_split_lookup:
            for patient_id in patient_ids or []:
                assigned = patient_split_lookup.get(patient_id)
                if assigned != FIT_SPLIT:
                    raise ValueError(f"Patient {patient_id} is in {assigned}, not expert_train")
        data = np.asarray(vectors, dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]
        if data.shape[0] < 3:
            raise ValueError("Need at least 3 expert_train vectors to fit Mahalanobis statistics")
        self.mean_ = data.mean(0)
        centered = data - self.mean_
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        rank = int(min(self.max_components, data.shape[0] - 1, data.shape[1], max(1, int((singular > 1e-8).sum()))))
        self.components_ = vt[:rank]
        reduced = centered @ self.components_.T
        self.covariance_ = LedoitWolf().fit(reduced)
        self.fitted_split = FIT_SPLIT
        return self

    def transform(self, vectors):
        if self.mean_ is None:
            raise RuntimeError("Mahalanobis model is not fit")
        data = np.asarray(vectors, dtype=np.float64)
        if data.ndim == 1:
            data = data[None, :]
        return (data - self.mean_) @ self.components_.T

    def distance(self, vectors):
        reduced = self.transform(vectors)
        return np.asarray(self.covariance_.mahalanobis(reduced), dtype=np.float64)

    def summary(self):
        return {
            "expert": self.expert_name,
            "fitted_split": self.fitted_split,
            "covariance": "pca_ledoit_wolf",
            "n_components": None if self.components_ is None else int(self.components_.shape[0]),
            "feature_dim": None if self.mean_ is None else int(self.mean_.shape[0]),
        }
