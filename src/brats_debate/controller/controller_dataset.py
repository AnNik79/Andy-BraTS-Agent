from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from ..config import fingerprint, protocol
from ..data.preprocessing import crop_pad, sample_center, boundary_target


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


class ControllerDataset(Dataset):
    def __init__(self, cfg, splits, expert_hashes):
        self.cfg = cfg
        self.ids = splits["controller_train"]
        self.provenance = cache_provenance(cfg, splits, expert_hashes)
        self.paths = [Path(cfg["output_dir"]) / "controller_cache" / f"{pid}.pt" for pid in self.ids]
        self.count = cfg["patches_per_patient"]
        for path in self.paths:
            if not path.is_file():
                raise FileNotFoundError(f"Missing controller cache: {path}; run generate_controller_data.py")

    def __len__(self):
        return len(self.paths) * self.count

    def __getitem__(self, index):
        data = load_cache(self.paths[index // self.count], self.provenance, self.ids)
        labels = data["label"].numpy()
        rng = np.random.default_rng(np.random.randint(2**31))
        # Include actual frozen-expert disagreement regions as well as GT tumor/background.
        if rng.random() < .5 and (data["disagreement"] > .1).any():
            coordinates = np.argwhere(data["disagreement"].numpy() > .1)
            center = coordinates[rng.integers(len(coordinates))]
        else:
            center = sample_center(labels, rng, specialist=True)
        result = {}
        for key in ("features", "probabilities", "availability", "label"):
            array, _ = crop_pad(data[key].numpy(), self.cfg["crop_size"], center)
            result[key] = torch.from_numpy(array).float() if key != "label" else torch.from_numpy(array).long()
        # Padded background has a valid background distribution from the CNN.
        missing = result["availability"].sum(0) == 0
        result["availability"][0, missing] = 1
        result["probabilities"][0, 0, missing] = 1
        return result
