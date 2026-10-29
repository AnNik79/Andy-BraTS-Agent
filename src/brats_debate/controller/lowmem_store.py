"""Disk shards for one-expert-at-a-time controller pilots.

Full expert volumes stay on disk in depth slabs. A later stage reads only
the slab it is scoring, including the one-voxel halo a 3x3x3 gate needs.
Values are float32, the same dtype as the in-memory inference path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..config import SEGMENTATION_EXPERTS
from .spatial_stream import build_gate_inputs

SHARD_DEPTH = 16
EXPERT_ARRAYS = (
    "probabilities",
    "uncertainty",
    "confidence",
    "features",
    "margin",
    "availability",
)
BOUNDARY_ARRAYS = ("boundary_probability", "distance_to_boundary")


def _depth(array: np.ndarray) -> int:
    if array.ndim < 3:
        raise ValueError("a sharded volume needs spatial axes")
    return int(array.shape[-3])


def save_array_shards(directory: Path, array: np.ndarray, shard_depth: int = SHARD_DEPTH) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    values = np.ascontiguousarray(array)
    depth = _depth(values)
    if depth < 1 or shard_depth < 1:
        raise ValueError("shard depth must be positive")
    for index, start in enumerate(range(0, depth, shard_depth)):
        piece = np.ascontiguousarray(values[..., start : start + shard_depth, :, :])
        np.save(directory / f"{index:03d}.npy", piece)


def load_depth_range(directory: Path, start: int, stop: int) -> np.ndarray:
    """Load spatial axis -3 over [start, stop), copying only those slices."""

    if stop <= start:
        raise ValueError("depth range is empty")
    files = sorted(Path(directory).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no shards in {directory}")
    parts = []
    cursor = 0
    for path in files:
        shard = np.load(path, mmap_mode="r")
        depth = _depth(shard)
        shard_stop = cursor + depth
        if shard_stop > start and cursor < stop:
            lo = max(0, start - cursor)
            hi = min(depth, stop - cursor)
            parts.append(np.array(shard[..., lo:hi, :, :], dtype=np.float32, copy=True))
        cursor = shard_stop
        del shard
    if not parts or cursor < stop:
        raise ValueError(f"shards in {directory} do not cover [{start}, {stop})")
    return np.concatenate(parts, axis=-3)


def save_expert_compact(directory: Path, compact: dict) -> None:
    """Persist one expert and a tiny proposal cache. The caller owns the memory."""

    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    arrays = {
        "probabilities": compact["probabilities"],
        "uncertainty": compact["uncertainty"],
        "confidence": compact["confidence"],
        "features": compact["features"],
        "margin": compact["extra"]["margin"],
        "availability": compact["extra"]["availability"],
    }
    extra = compact["extra"]
    for name in BOUNDARY_ARRAYS:
        if name in extra:
            arrays[name] = extra[name]
    for name, value in arrays.items():
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        save_array_shards(directory / name, np.asarray(array, dtype=np.float32))
    probabilities = compact["probabilities"]
    segmentation = probabilities.argmax(1).to(torch.uint8).cpu().numpy()
    uncertainty = compact["uncertainty"].detach().cpu().numpy().astype(np.float32, copy=False)
    np.save(directory / "segmentation.npy", np.ascontiguousarray(segmentation))
    np.save(directory / "uncertainty.npy", np.ascontiguousarray(uncertainty))
    (directory / "done.json").write_text(json.dumps({"complete": True, "keys": sorted(arrays)}))


def expert_complete(directory: Path) -> bool:
    path = Path(directory) / "done.json"
    return path.is_file() and json.loads(path.read_text()).get("complete") is True


def load_expert_depth(directory: Path, start: int, stop: int) -> dict:
    directory = Path(directory)

    def grab(name: str) -> torch.Tensor:
        return torch.from_numpy(load_depth_range(directory / name, start, stop))

    extra = {"margin": grab("margin"), "availability": grab("availability")}
    if (directory / "boundary_probability").is_dir():
        extra["boundary_probability"] = grab("boundary_probability")
        extra["distance_to_boundary"] = grab("distance_to_boundary")
    return {
        "probabilities": grab("probabilities"),
        "uncertainty": grab("uncertainty"),
        "confidence": grab("confidence"),
        "features": grab("features"),
        "extra": extra,
    }


def crop_loaded(output: dict, y0: int, y1: int, x0: int, x1: int) -> dict:
    def cut(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[..., :, y0:y1, x0:x1].contiguous()

    return {
        "probabilities": cut(output["probabilities"]),
        "uncertainty": cut(output["uncertainty"]),
        "confidence": cut(output["confidence"]),
        "features": cut(output["features"]),
        "extra": {key: cut(value) for key, value in output["extra"].items()},
    }


def proposal_from_experts(case_dir: Path, names=("cnn", "transformer", "boundary"), threshold: float = 0.55) -> torch.Tensor:
    """Match compact HighRes proposal construction: foreground vote or mean uncertainty."""

    segmentations = []
    uncertainties = []
    for name in names:
        directory = Path(case_dir) / name
        segmentations.append(torch.from_numpy(np.load(directory / "segmentation.npy")))
        uncertainties.append(torch.from_numpy(np.load(directory / "uncertainty.npy")))
    votes = torch.stack([(segmentation > 0) for segmentation in segmentations]).any(0)[:, None]
    uncertain = torch.stack(uncertainties).mean(0) > threshold
    return (votes | uncertain).to(dtype=torch.float32)


def gather_probabilities(directory: Path, flat_index: np.ndarray) -> np.ndarray:
    """Gather (sample, class) probabilities. flat_index is C-order over the spatial volume."""

    files = sorted((Path(directory) / "probabilities").glob("*.npy"))
    depth = 0
    for path in files:
        shard = np.load(path, mmap_mode="r")
        depth += _depth(shard)
        del shard
    values = load_depth_range(Path(directory) / "probabilities", 0, depth)
    array = np.asarray(values)
    if array.ndim == 5:
        array = array[0]
    gathered = array.reshape(array.shape[0], -1)[:, np.asarray(flat_index, dtype=np.int64)]
    return np.asarray(gathered.T, dtype=np.float32)


@torch.inference_mode()
def stream_stored_gates(modules, image, case_dir, cfg, slab_depth=8, halo=1, capture_flat=None):
    """Score stored experts one slab at a time. Halo voxels are real neighbors."""

    if image.ndim != 5 or image.shape[0] != 1:
        raise ValueError("stored streaming expects one image batch")
    if slab_depth < 1 or halo < 0:
        raise ValueError("slab depth must be positive")
    shape = tuple(int(size) for size in image.shape[-3:])
    depth = shape[0]
    labels = {name: torch.empty(shape, dtype=torch.uint8) for name in modules}
    timing = {name: 0.0 for name in modules}
    max_prob_error = {name: 0.0 for name in modules}
    max_weight_error = {name: 0.0 for name in modules}
    max_unavailable_weight = {name: 0.0 for name in modules}
    import time

    capture_coords = None
    if capture_flat is not None:
        capture_coords = np.unravel_index(np.asarray(capture_flat, dtype=np.int64), shape)
        captured = {name: np.empty((len(capture_flat), len(cfg["label_mapping"])), dtype=np.float32) for name in modules}
    for module in modules.values():
        module.eval()
    start = 0
    while start < depth:
        stop = min(start + slab_depth, depth)
        src_start = max(0, start - halo)
        src_stop = min(depth, stop + halo)
        if stop < depth and src_stop != stop + halo:
            raise RuntimeError("an open slab face is missing its stored neighbor")
        if start > 0 and src_start != start - halo:
            raise RuntimeError("an open slab face is missing its stored neighbor")
        outputs = {
            name: load_expert_depth(Path(case_dir) / name, src_start, src_stop)
            for name in SEGMENTATION_EXPERTS
        }
        built = time.perf_counter()
        features, probabilities, availability = build_gate_inputs(image[..., src_start:src_stop, :, :], outputs)
        feature_seconds = time.perf_counter() - built
        timing["features"] = timing.get("features", 0.0) + feature_seconds
        offset = start - src_start
        length = stop - start
        dest = (slice(start, stop), slice(None), slice(None))
        src = (slice(offset, offset + length), slice(None), slice(None))
        for name, module in modules.items():
            began = time.perf_counter()
            result = module(features, probabilities, availability)
            timing[name] += time.perf_counter() - began
            written_probs = result["probabilities"][(..., *src)]
            written_weights = result["weights"][(..., *src)]
            labels[name][dest] = written_probs.argmax(1)[0].to(torch.uint8)
            if capture_coords is not None:
                chosen = np.flatnonzero((capture_coords[0] >= start) & (capture_coords[0] < stop))
                if chosen.size:
                    local_z = capture_coords[0][chosen] - start
                    captured[name][chosen] = (
                        written_probs[0, :, local_z, capture_coords[1][chosen], capture_coords[2][chosen]]
                        .T.detach()
                        .cpu()
                        .numpy()
                    )
            max_prob_error[name] = max(max_prob_error[name], float((written_probs.sum(1) - 1).abs().max()))
            max_weight_error[name] = max(max_weight_error[name], float((written_weights.sum(1) - 1).abs().max()))
            unavailable = availability[(..., *src)] <= 0
            if bool(unavailable.any()):
                max_unavailable_weight[name] = max(
                    max_unavailable_weight[name],
                    float(written_weights.masked_select(unavailable).abs().max()),
                )
            del result, written_probs, written_weights
        del outputs, features, probabilities, availability
        start = stop
    stats = {
        "forward_seconds": {name: timing[name] for name in modules},
        "feature_seconds": timing.get("features", 0.0),
        "max_probability_sum_error": max_prob_error,
        "max_weight_sum_error": max_weight_error,
        "max_unavailable_expert_weight": max_unavailable_weight,
    }
    if capture_coords is not None:
        stats["captured_probabilities"] = captured
    return labels, stats
