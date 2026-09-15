from pathlib import Path
import hashlib
import json
import random
import numpy as np
import torch
import yaml

# Four voxel segmentation experts. Do not add "llm" here: controller channels,
# probability stacks, and checkpoint paths are sized to this tuple.
SEGMENTATION_EXPERTS = ("cnn", "transformer", "boundary", "highres")
REASONING_EXPERT = "llm"
SYSTEM_EXPERTS = SEGMENTATION_EXPERTS + (REASONING_EXPERT,)
# Backward-compatible alias used by training/inference tensor code.
EXPERTS = SEGMENTATION_EXPERTS

# Internal checkpoint key "boundary" is the professor's Sum Medical role.
SEGMENTATION_EXPERT_ROLES = {
    "cnn": "local structure / voxel-level prediction",
    "transformer": "broader spatial context / long-range relationships",
    "boundary": "Sum Medical: boundary geometry / medical-geometry specialist",
    "highres": "fine-structure / high-resolution detector",
}

REASONING_DEFAULTS = {
    "mode": "deterministic",
    "provider": "openai_compatible",
    "model": None,
    "api_key_env": "BRATS_LLM_API_KEY",
    "base_url_env": "BRATS_LLM_BASE_URL",
    "model_env": "BRATS_LLM_MODEL",
    "timeout_seconds": 60,
    "max_disagreement_regions": 8,
}


def _reasoning_config(raw):
    cfg = {**REASONING_DEFAULTS, **(raw or {})}
    if cfg["mode"] not in ("llm", "deterministic"):
        raise ValueError("reasoning.mode must be 'llm' or 'deterministic'")
    if cfg["provider"] != "openai_compatible":
        raise ValueError("reasoning.provider must be openai_compatible")
    if cfg["timeout_seconds"] < 1 or cfg["max_disagreement_regions"] < 1:
        raise ValueError("reasoning timeouts and region caps must be positive")
    return cfg


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    for key in ("dataset_root", "output_dir", "manifest", "split_file", "dataset_archive", "audit_report", "patient_cache_dir"):
        if cfg.get(key):
            cfg[key] = str((path.parent / cfg[key]).resolve())
    cfg["label_mapping"] = {int(k): int(v) for k, v in cfg["label_mapping"].items()}
    mapping = cfg["label_mapping"]
    if mapping.get(0) != 0 or sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("label_mapping must be one-to-one, contiguous from 0, with background 0->0")
    for key in ("crop_size", "highres_patch_size"):
        if len(cfg[key]) != 3 or min(cfg[key]) < 4:
            raise ValueError(f"{key} must contain three sizes >=4")
    for key in ("batch_size", "epochs", "controller_epochs", "gradient_accumulation", "patches_per_patient"):
        if cfg[key] < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 <= cfg["inference"]["overlap"] < 1:
        raise ValueError("overlap must be in [0,1)")
    if cfg["inference"]["mc_samples"] < 1:
        raise ValueError("mc_samples must be >=1")
    cap = cfg["inference"]["highres_max_patches"]
    if cap is not None and cap < 1:
        raise ValueError("highres_max_patches must be positive or null")
    for threshold in (cfg["inference"]["uncertainty_threshold"], cfg["evaluation"]["high_disagreement_threshold"]):
        if not 0 <= threshold <= 1:
            raise ValueError("Uncertainty/disagreement thresholds must be in [0,1]")
    for values in cfg["regions"].values():
        if not set(values) <= set(mapping.values()):
            raise ValueError("Regions must use mapped class indices")
    if not {"WT", "TC", "ET"} <= set(cfg["regions"]):
        raise ValueError("regions must include WT, TC, and ET for BraTS evaluation")
    cfg["model"] = {"unet_depth": 3, "transformer_window": 4, **cfg["model"]}
    if cfg["model"]["unet_depth"] < 2:
        raise ValueError("unet_depth must be >=2")
    if cfg["model"]["transformer_dim"] % cfg["model"]["transformer_heads"]:
        raise ValueError("transformer_dim must be divisible by transformer_heads")
    if cfg.get("validation_every_epochs", 1) < 1:
        raise ValueError("validation_every_epochs must be positive")
    if cfg.get("early_stopping_patience", 0) < 0:
        raise ValueError("early_stopping_patience must be >=0")
    cfg.setdefault("validation_every_epochs", 1)
    cfg.setdefault("early_stopping_patience", 0)
    cfg.setdefault("validate_first_epoch", True)
    cfg["reasoning"] = _reasoning_config(cfg.get("reasoning"))
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def protocol(cfg):
    return {k: cfg[k] for k in ("modalities", "label_mapping", "regions", "model", "crop_size",
                                "highres_patch_size", "inference")}
