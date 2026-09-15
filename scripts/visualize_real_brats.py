"""Save raw modalities, source labels, overlays, and preprocessing checks for five subjects."""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from brats_debate.config import save_json
from brats_debate.data.real_audit import archive_inventory, image_from_archive
from brats_debate.data.preprocessing import normalize_nonzero


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    cases, _ = archive_inventory(args.archive)
    selected, seen = [], set()
    for pid in cases:
        subject = pid.rsplit("-", 1)[0]
        if subject not in seen:
            selected.append(pid)
            seen.add(subject)
        if len(selected) == 5:
            break
    cmap = ListedColormap(["black", "#e66101", "#5eac43", "#d5289b"])
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], cmap.N)
    summaries = []
    for pid in selected:
        modalities = ["t1n", "t1c", "t2w", "t2f"]
        raw = np.stack([image_from_archive(args.archive, cases[pid][m]).get_fdata(dtype=np.float32) for m in modalities])
        label_image = image_from_archive(args.archive, cases[pid]["seg"])
        labels = label_image.get_fdata(dtype=np.float32).astype(np.int16)
        normalized = normalize_nonzero(raw)
        stats = {}
        for i, name in enumerate(modalities):
            mask = raw[i] != 0
            stats[name] = {"mean": float(normalized[i][mask].mean()), "std": float(normalized[i][mask].std()),
                           "background_preserved": bool((normalized[i][~mask] == 0).all())}
            assert abs(stats[name]["mean"]) < 1e-4 and abs(stats[name]["std"] - 1) < 1e-4
            assert stats[name]["background_preserved"]
        fig, axes = plt.subplots(3, 6, figsize=(18, 10))
        slices = {}
        for row, axis in enumerate((2, 1, 0)):
            reduced_axes = tuple(i for i in range(3) if i != axis)
            index = int((labels > 0).sum(axis=reduced_axes).argmax())
            slices[str(axis)] = index
            seg = np.take(labels, index, axis=axis).T
            for col, name in enumerate(modalities):
                img = np.take(raw[col], index, axis=axis).T
                values = raw[col][raw[col] != 0]
                low, high = np.percentile(values, [1, 99])
                axes[row, col].imshow(img, cmap="gray", vmin=low, vmax=high, origin="lower")
                axes[row, col].set_title(f"{name}: native axis {axis}, slice {index}")
            axes[row, 4].imshow(seg, cmap=cmap, norm=norm, origin="lower", interpolation="nearest")
            axes[row, 4].set_title("Ground truth: source labels")
            axes[row, 5].imshow(np.take(raw[3], index, axis=axis).T, cmap="gray", origin="lower")
            axes[row, 5].imshow(np.ma.masked_where(seg == 0, seg), cmap=cmap, norm=norm, alpha=.6,
                                origin="lower", interpolation="nearest")
            axes[row, 5].set_title("t2f + ground truth")
        for ax in axes.flat:
            ax.axis("off")
        fig.suptitle(f"{pid} — real BraTS training data; native LPS axes; 1 mm voxels", fontsize=15)
        fig.legend(handles=[Patch(color=cmap(i), label=f"Source label {i}") for i in range(4)], loc="lower center", ncol=4)
        fig.tight_layout(rect=[0, .04, 1, .95])
        path = destination / f"{pid}_sanity.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        coords = np.argwhere(labels > 0)
        summaries.append({"patient_id": pid, "visualization": str(path.resolve()), "source_labels": np.unique(labels).tolist(),
                          "slice_indices": slices, "normalization": stats,
                          "tumor_bounding_box": [coords.min(0).tolist(), coords.max(0).tolist()]})
        print(path, flush=True)
    save_json(destination / "preprocessing_checks.json", summaries)


if __name__ == "__main__":
    main()
