import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def visualize(patient, predictions, disagreement, path, modality_index=0, weights=None):
    reference = patient["label"] if patient["label"] is not None else predictions.get("final", next(iter(predictions.values())))
    counts = (reference > 0).sum(axis=(0, 1))
    index = int(counts.argmax()) if counts.any() else reference.shape[2] // 2
    panels = [("MRI", patient["image"][modality_index, :, :, index], "gray")]
    if patient["label"] is not None:
        panels.append(("Ground truth", patient["label"][:, :, index], "tab10"))
    panels += [(name, array[:, :, index], "tab10") for name, array in predictions.items()
               if name not in ("average", "vote")]
    panels.append(("Disagreement", disagreement[:, :, index], "magma"))
    if weights is not None:
        panels += [(f"Weight: {name}", array[:, :, index], "viridis") for name, array in weights.items()]
    fig, axes = plt.subplots(int(np.ceil(len(panels) / 4)), 4, figsize=(16, 4 * int(np.ceil(len(panels) / 4))))
    vmax = max(int(array.max()) for name, array in predictions.items())
    if patient["label"] is not None:
        vmax = max(vmax, int(patient["label"].max()))
    for ax, (title, array, cmap) in zip(axes.flat, panels):
        settings = {} if cmap == "gray" else {"vmin": 0, "vmax": max(vmax, 1) if cmap == "tab10" else 1}
        shown = ax.imshow(array.T, origin="lower", cmap=cmap, **settings)
        ax.set_title(title)
        ax.axis("off")
        if cmap in ("magma", "viridis"):
            fig.colorbar(shown, ax=ax, fraction=.04)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"{patient['patient_id']} — native voxel axis 2, slice {index}; research only")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
