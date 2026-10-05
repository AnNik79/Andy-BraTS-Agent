"""Controller training patches.

Full volumes are consumed only while a case is sampled. The cache stores the
patches the gate trains on, drawn once with the existing center rule, because
keeping every controller_train and validation feature volume does not fit on disk.
"""
from pathlib import Path
import hashlib
import os
import pickle
import shutil
import tempfile
import numpy as np
import torch
from torch.utils.data import Dataset
from ..config import SEGMENTATION_EXPERTS, fingerprint, protocol
from ..data.preprocessing import crop_pad, sample_center
from ..debate.features import feature_channels

TRAINING_CACHE_FORMAT = "controller_train_patches_v1"
DISAGREEMENT_CENTER_THRESHOLD = 0.1
# Pickle and filesystem overhead beyond the raw tensors.
CACHE_MARGIN_NUMERATOR = 1
CACHE_MARGIN_DENOMINATOR = 5
CACHE_MARGIN_BYTES = 512 * 1024 * 1024


def cache_provenance(cfg, splits, expert_hashes):
    return {"protocol_hash": fingerprint(protocol(cfg)), "split_hash": fingerprint(splits),
            "expert_hashes": expert_hashes}


def load_cache(path, expected, allowed_ids):
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["provenance"] != expected:
        raise ValueError(f"Stale controller cache: {path}; regenerate with current expert checkpoints")
    if data["patient_id"] not in allowed_ids:
        raise ValueError("Controller cache patient outside permitted training/validation split")
    return data


def cache_directory(cfg):
    return Path(cfg["output_dir"]) / "controller_cache"


def training_cache_path(cfg, patient_id):
    return cache_directory(cfg) / f"{patient_id}.pt"


def patch_rng(seed, patient_id, patch_index):
    """Stable stream so a resumed case repeats the same centers instead of a new draw."""
    digest = hashlib.sha256(f"{seed}:{patient_id}:{patch_index}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def choose_patch_center(label, disagreement, rng):
    """Same rule previously applied on every training fetch.

    Half of the draws that have expert disagreement above 0.1 center there.
    The rest use the specialist tumor/boundary/background sampler.
    """
    disagreement = np.asarray(disagreement)
    if rng.random() < .5 and (disagreement > DISAGREEMENT_CENTER_THRESHOLD).any():
        coordinates = np.argwhere(disagreement > DISAGREEMENT_CENTER_THRESHOLD)
        return tuple(int(value) for value in coordinates[rng.integers(len(coordinates))])
    return sample_center(np.asarray(label), rng, specialist=True)


def _crop_size(cfg):
    return tuple(int(value) for value in cfg["crop_size"])


def training_cache_estimate(n_cases, cfg):
    """Bytes for n_cases of stored training patches, including a write margin."""
    depth, height, width = _crop_size(cfg)
    voxels = depth * height * width
    experts = len(SEGMENTATION_EXPERTS)
    classes = len(cfg["label_mapping"])
    channels = feature_channels(cfg)
    per_patch = (channels * voxels * 2 + experts * classes * voxels * 2
                 + experts * voxels * 2 + voxels * 8)
    patches = int(n_cases) * int(cfg["patches_per_patient"])
    tensor_bytes = patches * per_patch
    margin = tensor_bytes * CACHE_MARGIN_NUMERATOR // CACHE_MARGIN_DENOMINATOR + CACHE_MARGIN_BYTES
    return {
        "cases": int(n_cases),
        "patches_per_case": int(cfg["patches_per_patient"]),
        "crop_size": [depth, height, width],
        "feature_channels": channels,
        "tensor_bytes": tensor_bytes,
        "margin_bytes": margin,
        "required_bytes": tensor_bytes + margin if n_cases else 0,
        "stores_validation_volumes": False,
    }


def free_bytes(path):
    probe = Path(path)
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def assert_cache_space(path, estimate):
    required = int(estimate["required_bytes"])
    available = free_bytes(path)
    if available < required:
        raise RuntimeError(
            "Not enough free disk for the controller training-patch cache at "
            f"{path}. Need {required / 1024 ** 3:.2f} GiB "
            f"({estimate['tensor_bytes'] / 1024 ** 3:.2f} GiB of tensors plus "
            f"{estimate['margin_bytes'] / 1024 ** 3:.2f} GiB margin) for "
            f"{estimate['cases']} cases; {available / 1024 ** 3:.2f} GiB free. "
            "No files were deleted."
        )
    return available


def _payload_complete(data, cfg, patient_id, provenance):
    crop = _crop_size(cfg)
    classes = len(cfg["label_mapping"])
    channels = feature_channels(cfg)
    experts = len(SEGMENTATION_EXPERTS)
    if data.get("format") != TRAINING_CACHE_FORMAT or data.get("complete") is not True:
        return False
    if data.get("patient_id") != patient_id or data.get("provenance") != provenance:
        return False
    if data.get("split") != "controller_train":
        return False
    patches = data.get("patches")
    if not isinstance(patches, list) or len(patches) != int(cfg["patches_per_patient"]):
        return False
    for patch in patches:
        if set(patch) != {"features", "probabilities", "availability", "label"}:
            return False
        features, probabilities = patch["features"], patch["probabilities"]
        availability, label = patch["availability"], patch["label"]
        if tuple(features.shape) != (channels, *crop) or features.dtype != torch.float16:
            return False
        if tuple(probabilities.shape) != (experts, classes, *crop) or probabilities.dtype != torch.float16:
            return False
        if tuple(availability.shape) != (experts, *crop) or availability.dtype != torch.float16:
            return False
        if tuple(label.shape) != crop or label.dtype != torch.int64:
            return False
    return True


def training_cache_is_complete(path, cfg, patient_id, provenance):
    """True only for a fully written patch cache. Partial and corrupt files are incomplete."""
    path = Path(path)
    if path.suffix != ".pt" or not path.is_file():
        return False
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError):
        return False
    return _payload_complete(data, cfg, patient_id, provenance)


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".partial", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _drop_batch(volume, unbatched_ndim):
    array = volume.detach().cpu().numpy() if torch.is_tensor(volume) else np.asarray(volume)
    if array.ndim == unbatched_ndim + 1 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != unbatched_ndim:
        raise ValueError(f"Expected {unbatched_ndim} spatial-leading dimensions, found {array.shape}")
    return array


def extract_training_patches(features, probabilities, availability, label, disagreement, cfg, patient_id):
    """Crop the sampled patches and drop the full volumes from the returned payload."""
    features = _drop_batch(features, 4)
    probabilities = _drop_batch(probabilities, 5)
    availability = _drop_batch(availability, 4)
    label = np.asarray(label)
    disagreement = np.asarray(disagreement)
    if disagreement.ndim > 3:
        disagreement = np.squeeze(disagreement)
    crop = _crop_size(cfg)
    patches = []
    for index in range(int(cfg["patches_per_patient"])):
        center = choose_patch_center(label, disagreement, patch_rng(cfg["seed"], patient_id, index))
        cropped = {}
        for key, volume in (("features", features), ("probabilities", probabilities),
                            ("availability", availability), ("label", label)):
            array, _meta = crop_pad(volume, crop, center)
            cropped[key] = array
        missing = cropped["availability"].sum(0) == 0
        cropped["availability"][0, missing] = 1
        cropped["probabilities"][0, 0, missing] = 1
        patches.append({
            "features": torch.from_numpy(np.ascontiguousarray(cropped["features"])).half(),
            "probabilities": torch.from_numpy(np.ascontiguousarray(cropped["probabilities"])).half(),
            "availability": torch.from_numpy(np.ascontiguousarray(cropped["availability"])).half(),
            "label": torch.from_numpy(np.ascontiguousarray(cropped["label"])).long(),
        })
    return patches


def save_training_cache(path, patient_id, provenance, patches):
    atomic_torch_save(path, {
        "format": TRAINING_CACHE_FORMAT,
        "complete": True,
        "split": "controller_train",
        "patient_id": patient_id,
        "provenance": provenance,
        "patches": patches,
    })


class ControllerDataset(Dataset):
    def __init__(self, cfg, splits, expert_hashes):
        self.cfg = cfg
        self.ids = list(splits["controller_train"])
        if set(self.ids) & set(splits["test"]):
            raise RuntimeError("Controller training cache overlaps the test split")
        self.provenance = cache_provenance(cfg, splits, expert_hashes)
        self.count = cfg["patches_per_patient"]
        self.paths = []
        for patient_id in self.ids:
            path = training_cache_path(cfg, patient_id)
            if not training_cache_is_complete(path, cfg, patient_id, self.provenance):
                raise FileNotFoundError(f"Missing complete controller cache: {path}; run generate_controller_data.py")
            self.paths.append(path)

    def __len__(self):
        return len(self.paths) * self.count

    def __getitem__(self, index):
        data = load_cache(self.paths[index // self.count], self.provenance, self.ids)
        patch = data["patches"][index % self.count]
        return {
            "features": patch["features"].float(),
            "probabilities": patch["probabilities"].float(),
            "availability": patch["availability"].float(),
            "label": patch["label"].long(),
        }
