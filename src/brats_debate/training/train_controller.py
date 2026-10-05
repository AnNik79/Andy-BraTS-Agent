from itertools import combinations
from pathlib import Path
import csv
import gc
import json
import os
import pickle
import re
import shutil
import signal
import tempfile
import torch
from torch.utils.data import DataLoader
from ..config import seed_everything, device_for, save_json, file_hash, fingerprint, protocol
from ..data.brats_dataset import load_patient
from ..inference.pipeline import load_experts
from ..debate.disagreement import analyze_disagreement
from ..debate.features import controller_features, feature_channels, stack_evidence
from ..controller.controller_dataset import (
    ControllerDataset, assert_cache_space, atomic_torch_save, cache_directory, cache_provenance,
    extract_training_patches, save_training_cache, training_cache_estimate, training_cache_is_complete,
    training_cache_path,
)
from ..controller.gating_network import GatingNetwork
from ..evaluation.metrics import BRATS_REGIONS, segmentation_metrics
from .checkpoint import cpu_snapshot, restore_rng, rng_state
from .train_expert import segmentation_loss, append_log

SLAB = 8
EPOCH_FORMAT = "controller_epoch_v1"
CASE_FORMAT = "controller_validation_case_v1"
SUMMARY_FORMAT = "controller_validation_summary_v1"
EPOCH_FILE = re.compile(r"^epoch_(\d+)\.pt$")
PATIENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def assert_split_isolation(splits):
    keys = ("expert_train", "controller_train", "validation", "test")
    missing = [key for key in keys if key not in splits]
    if missing:
        raise ValueError(f"Controller splits are missing {missing}")
    for left, right in combinations(keys, 2):
        overlap = set(splits[left]) & set(splits[right])
        if overlap:
            raise RuntimeError(f"Refusing overlapping {left} and {right} splits")


def records_for(records, wanted_ids, forbidden_ids):
    """Select cases by id. Forbidden ids are never returned and are not loaded."""
    wanted = list(wanted_ids)
    forbidden = set(forbidden_ids)
    if set(wanted) & forbidden:
        raise RuntimeError("Refusing a split that overlaps the forbidden patient ids")
    found = {}
    for record in records:
        patient_id = record["patient_id"]
        if patient_id in forbidden:
            continue
        if patient_id in set(wanted):
            found[patient_id] = record
    missing = [patient_id for patient_id in wanted if patient_id not in found]
    if missing:
        raise FileNotFoundError(f"Missing controller cases, including {missing[:3]}")
    return [found[patient_id] for patient_id in wanted]


def _freeze_experts(models):
    for expert in models.values():
        if isinstance(expert, torch.nn.Module):
            expert.eval().requires_grad_(False)


def _expert_outputs(patient, models, cfg):
    from ..inference.pipeline import predict_experts
    return predict_experts(patient, models, cfg, device_for(cfg["device"]))


def _controller_volumes(patient, outputs):
    image = torch.from_numpy(patient["image"])[None]
    debate = analyze_disagreement(outputs)
    features = controller_features(image, outputs, debate)
    probabilities, availability = stack_evidence(outputs)
    disagreement = debate["map"][0, 0]
    return features, probabilities, availability, disagreement


def generate_controller_data(records, splits, cfg):
    """Cache controller_train patches only. Validation volumes are not stored."""
    assert_split_isolation(splits)
    seed_everything(cfg["seed"])
    models, hashes = load_experts(cfg, splits)
    _freeze_experts(models)
    provenance = cache_provenance(cfg, splits, hashes)
    selected = records_for(records, splits["controller_train"], splits["test"])
    destination = cache_directory(cfg)
    pending = [record for record in selected
               if not training_cache_is_complete(training_cache_path(cfg, record["patient_id"]), cfg,
                                                 record["patient_id"], provenance)]
    estimate = training_cache_estimate(len(pending), cfg)
    assert_cache_space(destination, estimate)
    destination.mkdir(parents=True, exist_ok=True)
    print(f"controller cache: {len(pending)} training cases pending, "
          f"{estimate['required_bytes'] / 1024 ** 3:.2f} GiB reserved, validation volumes not stored",
          flush=True)
    for record in pending:
        if record["patient_id"] in set(splits["test"]):
            raise RuntimeError("Refusing to cache a test case")
        patient = load_patient(record, cfg)
        outputs = _expert_outputs(patient, models, cfg)
        features, probabilities, availability, disagreement = _controller_volumes(patient, outputs)
        patches = extract_training_patches(features, probabilities, availability, patient["label"],
                                           disagreement, cfg, patient["patient_id"])
        save_training_cache(training_cache_path(cfg, patient["patient_id"]), patient["patient_id"],
                            provenance, patches)
        print(f"Cached {len(patches)} patches for {patient['patient_id']}", flush=True)
        del patient, outputs, features, probabilities, availability, disagreement, patches
        gc.collect()
    save_json(destination / "provenance.json", provenance)


def controller_run_dir(cfg):
    return Path(cfg["output_dir"]) / "checkpoints" / "controller"


def validation_root(cfg):
    return Path(cfg["output_dir"]) / "controller_validation"


def epoch_checkpoint_path(cfg, epoch):
    return controller_run_dir(cfg) / f"epoch_{int(epoch):03d}.pt"


def parse_epoch_filename(path):
    match = EPOCH_FILE.fullmatch(Path(path).name)
    return int(match.group(1)) if match else None


def sorted_epoch_checkpoints(directory):
    """Epoch files in numeric epoch order. epoch_2.pt sorts before epoch_10.pt."""
    grouped = {}
    for path in Path(directory).glob("epoch_*.pt"):
        epoch = parse_epoch_filename(path)
        if epoch is None or epoch < 1:
            continue
        grouped.setdefault(epoch, []).append(path)
    duplicates = {epoch: [path.name for path in paths] for epoch, paths in grouped.items() if len(paths) > 1}
    if duplicates:
        raise FileExistsError(
            f"Duplicate controller epoch files {duplicates}. No files were overwritten. Use a new output_dir."
        )
    return [grouped[epoch][0] for epoch in sorted(grouped)]


def _run_signature(cfg, splits, expert_hashes):
    return {
        "protocol_hash": fingerprint(protocol(cfg)),
        "split_hash": fingerprint(splits),
        "expert_hashes": dict(expert_hashes),
        "seed": cfg["seed"],
        "learning_rate": cfg["learning_rate"],
        "controller_width": cfg["controller_width"],
        "batch_size": cfg["batch_size"],
        "patches_per_patient": cfg["patches_per_patient"],
        "gradient_accumulation": cfg["gradient_accumulation"],
        "training_patients": list(splits["controller_train"]),
        "controller_epochs": int(cfg["controller_epochs"]),
    }


def _signature_matches(payload, signature):
    return all(payload.get(key) == value for key, value in signature.items())


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".partial", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_epoch_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError):
        return None


def _epoch_is_resumable(payload, epoch, signature, device_type):
    if not isinstance(payload, dict) or payload.get("format") != EPOCH_FORMAT or payload.get("complete") is not True:
        return False
    if payload.get("role") != "controller" or payload.get("epoch") != epoch:
        return False
    if payload.get("device_type") != device_type or not _signature_matches(payload, signature):
        return False
    required = ("state_dict", "optimizer_state_dict", "scaler_state_dict", "rng_state", "loader_generator_state")
    return all(key in payload for key in required)


def _refuse_existing_run(cfg):
    directory = controller_run_dir(cfg)
    if (directory / "best.pt").exists():
        raise FileExistsError(
            "Controller best.pt already exists in this output directory. "
            "Refusing to overwrite a selected or previous run. Use a new output_dir. "
            "No files were overwritten."
        )
    unexpected = [path.name for path in directory.glob("*.pt") if parse_epoch_filename(path) is None] if directory.exists() else []
    if unexpected:
        raise FileExistsError(
            f"Controller directory contains unexpected checkpoints {unexpected}. "
            "Use a new output_dir. No files were overwritten."
        )
    root = validation_root(cfg)
    if root.exists() and any(path.is_file() for path in root.rglob("*")):
        raise FileExistsError(
            "Controller validation results already exist in this output directory. "
            "Refusing to modify the epoch checkpoints they score. Use a new output_dir. "
            "No files were overwritten."
        )


def _saved_epochs(cfg, splits, expert_hashes, device):
    """Contiguous completed epochs. Incomplete files are not treated as done and are not overwritten."""
    directory = controller_run_dir(cfg)
    if not directory.exists():
        return []
    paths = sorted_epoch_checkpoints(directory)
    epochs = [parse_epoch_filename(path) for path in paths]
    if epochs and epochs != list(range(1, epochs[-1] + 1)):
        raise FileExistsError(
            f"Controller epochs are not a contiguous prefix starting at 1: {epochs}. "
            "No files were overwritten. Use a new output_dir."
        )
    signature = _run_signature(cfg, splits, expert_hashes)
    saved = []
    for path, epoch in zip(paths, epochs):
        payload = _read_epoch_payload(path)
        if not _epoch_is_resumable(payload, epoch, signature, device.type):
            raise FileExistsError(
                f"{path.name} is incomplete, corrupt, or from a different run and was not treated as done. "
                "No files were overwritten. Use a new output_dir, or remove that incomplete file before resuming."
            )
        saved.append((path, payload))
    return saved


def _history_has_epoch(path, epoch):
    path = Path(path)
    if not path.is_file():
        return False
    with path.open(newline="") as stream:
        return any(int(row["epoch"]) == int(epoch) for row in csv.DictReader(stream))


def _append_training_history(cfg, epoch, train_loss, checkpoint_path):
    path = controller_run_dir(cfg) / "history.csv"
    if _history_has_epoch(path, epoch):
        return
    append_log(path, {"epoch": epoch, "seed": cfg["seed"], "train_loss": train_loss,
                      "checkpoint_path": str(checkpoint_path)})


def _save_controller_epoch(cfg, model, optimizer, scaler, generator, splits, expert_hashes, epoch, train_loss, device):
    path = epoch_checkpoint_path(cfg, epoch)
    if path.exists():
        raise FileExistsError(f"{path} already exists. Refusing to overwrite it.")
    payload = {
        "format": EPOCH_FORMAT,
        "complete": True,
        "state_dict": cpu_snapshot(model.state_dict()),
        "optimizer_state_dict": cpu_snapshot(optimizer.state_dict()),
        "scaler_state_dict": cpu_snapshot(scaler.state_dict()),
        "rng_state": rng_state(device),
        "loader_generator_state": generator.get_state().clone(),
        "role": "controller",
        "name": "controller",
        "device_type": device.type,
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "validation_dice": None,
        "architecture": str(model),
        **_run_signature(cfg, splits, expert_hashes),
    }
    atomic_torch_save(path, payload)
    _append_training_history(cfg, epoch, train_loss, path)
    return path


def _restore_training(model, optimizer, scaler, generator, payload, device):
    model.load_state_dict(payload["state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler.load_state_dict(payload["scaler_state_dict"])
    restore_rng(payload["rng_state"], device)
    generator.set_state(payload["loader_generator_state"])


@torch.no_grad()
def score_controller_metrics(model, features, probabilities, availability, label, cfg, device):
    """Full-volume gate in depth slabs. Returns scalars only; the fused volume is not retained."""
    target = torch.as_tensor(label)[None].long()
    final = torch.zeros_like(probabilities[:, 0], dtype=torch.float32)
    if final.shape[2:] != tuple(target.shape[1:]):
        raise ValueError("Controller features and labels have different spatial shapes")
    for start in range(0, target.shape[1], SLAB):
        sl = (slice(start, start + SLAB), slice(None), slice(None))
        output = model(features[(..., *sl)].float().to(device),
                       probabilities[(..., *sl)].float().to(device),
                       availability[(..., *sl)].float().to(device))
        final[(..., *sl)] = output["probabilities"].cpu()
    loss = float(segmentation_loss(final, target))
    metrics = segmentation_metrics(final.argmax(1)[0].numpy(), target[0].numpy(), cfg["regions"],
                                   len(cfg["label_mapping"]), hd95=False)
    scored = {"loss": loss, "brats_mean_dice": float(metrics["brats_mean_dice"])}
    scored.update({name: float(metrics[name]["dice"]) for name in BRATS_REGIONS})
    del final
    return scored


@torch.no_grad()
def score_controller_volume(model, features, probabilities, availability, label, cfg, device):
    """Full-volume gate in depth slabs. The caller discards the volumes after this returns."""
    scored = score_controller_metrics(model, features, probabilities, availability, label, cfg, device)
    return scored["loss"], scored["brats_mean_dice"]


@torch.no_grad()
def validate_controller(model, models, records, cfg, splits, device):
    """Score each validation case on its whole volume, then drop that case's tensors."""
    assert_split_isolation(splits)
    selected = records_for(records, splits["validation"], splits["test"])
    losses, scores = [], []
    _freeze_experts(models)
    was_training = model.training
    model.eval()
    try:
        for record in selected:
            if record["patient_id"] in set(splits["test"]):
                raise RuntimeError("Refusing to validate a test case")
            patient = load_patient(record, cfg)
            outputs = _expert_outputs(patient, models, cfg)
            features, probabilities, availability, _disagreement = _controller_volumes(patient, outputs)
            loss, dice = score_controller_volume(model, features, probabilities, availability,
                                                 patient["label"], cfg, device)
            losses.append(loss)
            scores.append(dice)
            del patient, outputs, features, probabilities, availability
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
    finally:
        model.train(was_training)
    if not scores:
        raise ValueError("Validation split is empty")
    return sum(losses) / len(losses), sum(scores) / len(scores)


class _EpochBoundaryStop:
    """SIGINT finishes the current epoch and exits after that checkpoint is saved."""

    def __init__(self):
        self.requested = False
        self._previous = None

    def install(self):
        self._previous = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._on_sigint)
        return self

    def _on_sigint(self, _signum, _frame):
        self.requested = True
        print("controller training will stop after the current epoch checkpoint is saved", flush=True)

    def close(self):
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)
            self._previous = None


def train_controller(records, splits, cfg, stop_after_epoch=None):
    """Train every configured epoch and save epoch_XXX.pt. Full-volume validation is a separate command.

    stop_after_epoch is only for an interrupted-run test. The CLI does not pass it.
    SIGINT does not cut off an epoch: the current epoch finishes, its checkpoint is saved,
    and the process exits. A later run of the same command continues at the next epoch.
    """
    assert_split_isolation(splits)
    if set(splits["controller_train"]) & set(splits["test"]):
        raise RuntimeError("Refusing to train the controller on a test case")
    known = {record["patient_id"] for record in records}
    if any(patient_id not in known for patient_id in splits["controller_train"]):
        raise FileNotFoundError("Controller training records are missing")
    _refuse_existing_run(cfg)
    seed_everything(cfg["seed"])
    device = device_for(cfg["device"])
    models, hashes = load_experts(cfg, splits)
    _freeze_experts(models)
    del models
    saved = _saved_epochs(cfg, splits, hashes, device)
    for path, payload in saved:
        _append_training_history(cfg, payload["epoch"], payload["train_loss"], path)
    target = int(cfg["controller_epochs"])
    if saved and saved[-1][1]["epoch"] >= target:
        print(f"controller training already has epochs 1..{target}; no checkpoints were overwritten", flush=True)
        return controller_run_dir(cfg)
    dataset = ControllerDataset(cfg, splits, hashes)
    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])
    model = GatingNetwork(feature_channels(cfg), cfg["controller_width"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    amp = cfg["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start = 1
    if saved:
        _restore_training(model, optimizer, scaler, generator, saved[-1][1], device)
        start = saved[-1][1]["epoch"] + 1
        print(f"controller training resume: epochs 1..{start - 1} already saved and will not be overwritten; "
              f"continuing at epoch {start}", flush=True)
    destination = controller_run_dir(cfg)
    destination.mkdir(parents=True, exist_ok=True)
    config_path = destination / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != json.loads(json.dumps(cfg)):
        raise FileExistsError("Controller config.json does not match this run. Use a new output_dir.")
    if not config_path.exists():
        save_json(config_path, cfg)
    architecture = destination / "architecture.txt"
    if architecture.exists() and architecture.read_text() != str(model):
        raise FileExistsError("Controller architecture.txt does not match this gate. Use a new output_dir.")
    if not architecture.exists():
        architecture.write_text(str(model))
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
                        generator=generator)
    pause = _EpochBoundaryStop().install()
    try:
        for epoch in range(start, target + 1):
            model.train()
            total = 0.
            optimizer.zero_grad(set_to_none=True)
            accumulation = cfg["gradient_accumulation"]
            for step, batch in enumerate(loader):
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, enabled=amp):
                    output = model(batch["features"], batch["probabilities"], batch["availability"])
                    loss = segmentation_loss(output["probabilities"], batch["label"])
                divisor = min(accumulation, len(loader) - (step // accumulation) * accumulation)
                scaler.scale(loss / divisor).backward()
                if (step + 1) % accumulation == 0 or step + 1 == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                total += float(loss.detach())
            train_loss = total / len(loader)
            path = _save_controller_epoch(cfg, model, optimizer, scaler, generator, splits, hashes, epoch, train_loss, device)
            print(f"controller epoch {epoch} saved; full-volume validation is deferred: {path}", flush=True)
            if pause.requested or (stop_after_epoch is not None and epoch >= int(stop_after_epoch)):
                print(f"controller training stopped after epoch {epoch}; rerun the same command to continue", flush=True)
                return destination
        return destination
    finally:
        pause.close()


def _validation_case_path(cfg, patient_id):
    if not PATIENT_ID.fullmatch(patient_id):
        raise ValueError(f"Unsafe patient id: {patient_id}")
    return validation_root(cfg) / "cases" / f"{patient_id}.json"


def _validation_provenance(cfg, splits, expert_hashes, epoch_paths):
    return {
        "protocol_hash": fingerprint(protocol(cfg)),
        "split_hash": fingerprint(splits),
        "expert_hashes": dict(expert_hashes),
        "epochs": [parse_epoch_filename(path) for path in epoch_paths],
        "checkpoint_sha256": {str(parse_epoch_filename(path)): file_hash(path) for path in epoch_paths},
        "scorer": {"slab": SLAB, "hd95": False, "metric": "brats_mean_dice"},
        "highres_max_patches": cfg["inference"]["highres_max_patches"],
        "overlap": cfg["inference"]["overlap"],
        "uncertainty_threshold": cfg["inference"]["uncertainty_threshold"],
        "mc_samples": cfg["inference"]["mc_samples"],
    }


def _case_result_is_complete(path, patient_id, epochs, provenance):
    path = Path(path)
    if path.suffix != ".json" or not path.is_file():
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if data.get("format") != CASE_FORMAT or data.get("complete") is not True or data.get("split") != "validation":
        return False
    if data.get("patient_id") != patient_id or data.get("provenance") != provenance:
        return False
    scores = data.get("scores")
    required = {"epoch", "loss", "brats_mean_dice", "WT", "TC", "ET"}
    if not isinstance(scores, list) or [row.get("epoch") for row in scores] != list(epochs):
        return False
    return all(isinstance(row, dict) and required <= set(row) for row in scores)


def _load_scoring_gates(cfg, epoch_paths, signature, forbidden_ids):
    gates = []
    for path in epoch_paths:
        epoch = parse_epoch_filename(path)
        payload = _read_epoch_payload(path)
        if not isinstance(payload, dict) or payload.get("complete") is not True or payload.get("epoch") != epoch:
            raise FileNotFoundError(f"{path.name} is incomplete and was not treated as done.")
        if payload.get("role") != "controller" or not _signature_matches(payload, signature):
            raise ValueError(f"{path.name} does not match this controller run.")
        if set(payload["training_patients"]) & set(forbidden_ids):
            raise RuntimeError("Refusing a controller checkpoint trained with test patients")
        model = GatingNetwork(feature_channels(cfg), cfg["controller_width"])
        model.load_state_dict(payload["state_dict"])
        model.eval().requires_grad_(False)
        gates.append((epoch, model))
    return gates


def _mean_metric(rows, key):
    return float(sum(row[key] for row in rows) / len(rows))


def aggregate_validation_scores(case_payloads, epoch_paths):
    """Mean of per-case scores, in numeric epoch order. Ties keep the earliest epoch."""
    by_epoch = {parse_epoch_filename(path): path for path in epoch_paths}
    per_epoch = []
    for epoch in sorted(by_epoch):
        rows = []
        for case in case_payloads:
            row = next(item for item in case["scores"] if item["epoch"] == epoch)
            rows.append(row)
        per_epoch.append({
            "epoch": epoch,
            "loss": _mean_metric(rows, "loss"),
            "brats_mean_dice": _mean_metric(rows, "brats_mean_dice"),
            "WT": _mean_metric(rows, "WT"),
            "TC": _mean_metric(rows, "TC"),
            "ET": _mean_metric(rows, "ET"),
            "checkpoint": str(by_epoch[epoch]),
            "checkpoint_sha256": file_hash(by_epoch[epoch]),
        })
    chosen = max(per_epoch, key=lambda row: (row["brats_mean_dice"], -row["epoch"]))
    return per_epoch, chosen


def _write_validation_summary(cfg, splits, expert_hashes, provenance, selected, per_epoch, chosen):
    summary = {
        "format": SUMMARY_FORMAT,
        "complete": True,
        "selected_epoch": chosen["epoch"],
        "selection_score": chosen["brats_mean_dice"],
        "selection_rule": "highest mean brats_mean_dice; earliest epoch breaks ties",
        "selected_checkpoint": chosen["checkpoint"],
        "best_checkpoint": str(controller_run_dir(cfg) / "best.pt"),
        "cases": [record["patient_id"] for record in selected],
        "epochs": [row["epoch"] for row in per_epoch],
        "per_epoch": per_epoch,
        "provenance": provenance,
        "metadata": {
            "protocol_hash": fingerprint(protocol(cfg)),
            "split_hash": fingerprint(splits),
            "expert_hashes": dict(expert_hashes),
            "expert_checkpoints": cfg.get("expert_checkpoints"),
            "seed": cfg["seed"],
            "controller_epochs": int(cfg["controller_epochs"]),
            "controller_width": cfg["controller_width"],
            "learning_rate": cfg["learning_rate"],
            "crop_size": cfg["crop_size"],
            "highres_patch_size": cfg["highres_patch_size"],
            "hd95": False,
            "slab": SLAB,
            "feature_channels": feature_channels(cfg),
        },
    }
    root = validation_root(cfg)
    atomic_write_text(root / "summary.json", json.dumps(summary, indent=2, allow_nan=False))
    lines = ["epoch,brats_mean_dice,WT,TC,ET,loss,checkpoint,checkpoint_sha256,selected"]
    for row in per_epoch:
        selected_flag = int(row["epoch"] == chosen["epoch"])
        lines.append(",".join(str(row[key]) for key in (
            "epoch", "brats_mean_dice", "WT", "TC", "ET", "loss", "checkpoint", "checkpoint_sha256"))
            + f",{selected_flag}")
    atomic_write_text(root / "summary.csv", "\n".join(lines) + "\n")
    return summary


def _promote_best(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if file_hash(destination) == file_hash(source):
            return
        raise FileExistsError(
            f"{destination} already exists and does not match the selected epoch {source}. "
            "No files were overwritten."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".best.", suffix=".partial", dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if file_hash(temporary) != file_hash(source):
            raise RuntimeError("Copied controller checkpoint does not match the selected epoch")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


@torch.no_grad()
def validate_saved_controllers(records, splits, cfg):
    """Run frozen experts once per validation case and score every saved epoch on those outputs.

    Completed case files are kept. An incomplete or stale case file is recomputed and is not
    treated as done. best.pt is written only after every validation case has a complete result.
    """
    assert_split_isolation(splits)
    selected = records_for(records, splits["validation"], splits["test"])
    forbidden = set(splits["test"])
    if any(record["patient_id"] in forbidden for record in selected):
        raise RuntimeError("Refusing to validate a test case")
    expected = list(range(1, int(cfg["controller_epochs"]) + 1))
    if not expected:
        raise ValueError("controller_epochs must be positive")
    epoch_paths = sorted_epoch_checkpoints(controller_run_dir(cfg))
    found = [parse_epoch_filename(path) for path in epoch_paths]
    if found != expected:
        raise FileNotFoundError(
            f"Need contiguous controller epochs {expected[0]}..{expected[-1]}; found {found}. "
            "Incomplete checkpoints are not treated as done."
        )
    device = device_for(cfg["device"])
    models, hashes = load_experts(cfg, splits)
    _freeze_experts(models)
    signature = _run_signature(cfg, splits, hashes)
    gates = _load_scoring_gates(cfg, epoch_paths, signature, forbidden)
    if [epoch for epoch, _model in gates] != expected:
        raise RuntimeError("Controller checkpoints were not scored in numeric epoch order")
    provenance = json.loads(json.dumps(_validation_provenance(cfg, splits, hashes, epoch_paths)))
    epochs = [epoch for epoch, _model in gates]
    pending, complete = [], []
    for record in selected:
        path = _validation_case_path(cfg, record["patient_id"])
        if _case_result_is_complete(path, record["patient_id"], epochs, provenance):
            complete.append(record)
        else:
            pending.append(record)
            if path.exists():
                print(f"validation: {record['patient_id']} result is incomplete and will be recomputed", flush=True)
    print(f"validation resume: {len(complete)} complete cases kept, "
          f"{len(pending)} incomplete cases will be rerun", flush=True)
    for record in pending:
        if record["patient_id"] in forbidden:
            raise RuntimeError("Refusing to validate a test case")
        patient = load_patient(record, cfg)
        outputs = _expert_outputs(patient, models, cfg)
        features, probabilities, availability, _disagreement = _controller_volumes(patient, outputs)
        scores = []
        for epoch, model in gates:
            model.to(device)
            scored = score_controller_metrics(model, features, probabilities, availability,
                                              patient["label"], cfg, device)
            model.cpu()
            scores.append({"epoch": epoch, **scored})
        case_id = patient["patient_id"]
        del patient, outputs, features, probabilities, availability
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        payload = {"format": CASE_FORMAT, "complete": True, "split": "validation", "patient_id": case_id,
                   "provenance": provenance, "scores": scores}
        atomic_write_text(_validation_case_path(cfg, case_id), json.dumps(payload, indent=2, allow_nan=False))
    case_payloads = []
    for record in selected:
        path = _validation_case_path(cfg, record["patient_id"])
        if not _case_result_is_complete(path, record["patient_id"], epochs, provenance):
            raise RuntimeError(f"Validation case {record['patient_id']} is still incomplete; best.pt was not written")
        case_payloads.append(json.loads(path.read_text()))
    per_epoch, chosen = aggregate_validation_scores(case_payloads, epoch_paths)
    summary = _write_validation_summary(cfg, splits, hashes, provenance, selected, per_epoch, chosen)
    _promote_best(chosen["checkpoint"], controller_run_dir(cfg) / "best.pt")
    if file_hash(controller_run_dir(cfg) / "best.pt") != file_hash(chosen["checkpoint"]):
        raise RuntimeError("best.pt does not match the selected epoch checkpoint")
    print(f"controller validation selected epoch {chosen['epoch']} "
          f"brats_mean_dice={chosen['brats_mean_dice']:.6f}", flush=True)
    return summary
