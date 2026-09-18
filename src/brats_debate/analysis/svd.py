"""Exploratory SVD of analysis matrices. Components are not biological labels."""
import numpy as np


def latent_svd(matrix, center=True, n_components=None, column_names=None, row_kind="sample"):
    """SVD of a samples-or-regions × features matrix.

    Returns singular values, energy, loadings, and optional rank-k reconstruction
    error. This is exploratory latent-structure analysis, not a causal model.
    """
    data = np.asarray(matrix, dtype=np.float64)
    if data.ndim != 2 or min(data.shape) < 1:
        raise ValueError("SVD requires a 2D matrix")
    mean = data.mean(0) if center else np.zeros(data.shape[1])
    centered = data - mean
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    energy = (singular ** 2)
    total = float(energy.sum()) if energy.size else 0.0
    explained = (energy / total).tolist() if total > 0 else [0.0] * len(singular)
    names = list(column_names) if column_names is not None else [f"feature_{i}" for i in range(data.shape[1])]
    if len(names) != data.shape[1]:
        raise ValueError("column_names must match matrix width")
    k = len(singular) if n_components is None else min(int(n_components), len(singular))
    loadings = [{name: float(vt[component, i]) for i, name in enumerate(names)} for component in range(k)]
    reconstruction = {}
    for rank in range(1, k + 1):
        approx = (u[:, :rank] * singular[:rank]) @ vt[:rank]
        reconstruction[rank] = float(np.mean((centered - approx) ** 2))
    return {
        "kind": "exploratory_latent_svd",
        "row_kind": row_kind,
        "n_rows": int(data.shape[0]),
        "n_columns": int(data.shape[1]),
        "column_names": names,
        "singular_values": singular.tolist(),
        "normalized_singular_values": (singular / singular[0]).tolist() if singular.size and singular[0] else [],
        "explained_energy": explained,
        "cumulative_energy": np.cumsum(explained).tolist() if explained else [],
        "loadings": loadings,
        "expert_or_feature_contribution": loadings,
        "reconstruction_mse_by_rank": reconstruction,
        "mean": mean.tolist(),
    }
