from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ..config import seed_everything, device_for, save_json, fingerprint, protocol
from ..data.brats_dataset import load_patient
from ..inference.pipeline import load_experts, predict_experts
from ..debate.disagreement import analyze_disagreement
from ..debate.features import controller_features, feature_channels, stack_evidence
from ..controller.controller_dataset import ControllerDataset, cache_provenance, load_cache
from ..controller.gating_network import GatingNetwork
from ..evaluation.metrics import segmentation_metrics
from .train_expert import segmentation_loss, append_log


def generate_controller_data(records, splits, cfg):
    seed_everything(cfg["seed"])
    models, hashes = load_experts(cfg, splits)
    destination = Path(cfg["output_dir"]) / "controller_cache"
    destination.mkdir(parents=True, exist_ok=True)
    provenance = cache_provenance(cfg, splits, hashes)
    allowed = splits["controller_train"] + splits["validation"]
    for record in records:
        if record["patient_id"] not in allowed:
            continue
        patient = load_patient(record, cfg)
        outputs = predict_experts(patient, models, cfg, device_for(cfg["device"]))
        debate = analyze_disagreement(outputs)
        features = controller_features(torch.from_numpy(patient["image"])[None], outputs, debate)
        probabilities, availability = stack_evidence(outputs)
        data = {"patient_id": patient["patient_id"], "provenance": provenance,
                "features": features[0].half(), "probabilities": probabilities[0].half(),
                "availability": availability[0].half(), "label": torch.from_numpy(patient["label"]),
                "disagreement": debate["map"][0, 0].half()}
        path = destination / f"{patient['patient_id']}.pt"
        torch.save(data, path)
        print(f"Cached {patient['patient_id']}: {path}", flush=True)
    save_json(destination / "provenance.json", provenance)


@torch.no_grad()
def validate_controller(model, cfg, splits, provenance, device):
    losses, scores = [], []
    for pid in splits["validation"]:
        data = load_cache(Path(cfg["output_dir"]) / "controller_cache" / f"{pid}.pt", provenance, splits["validation"])
        target = data["label"][None]
        final = torch.zeros_like(data["probabilities"][0:1], dtype=torch.float32)
        for start in range(0, target.shape[1], 8):
            sl = (slice(start, start + 8), slice(None), slice(None))
            args = [data[k][(..., *sl)][None].float().to(device) for k in ("features", "probabilities", "availability")]
            output = model(*args)
            final[(..., *sl)] = output["probabilities"].cpu()
        losses.append(float(segmentation_loss(final, target)))
        metrics = segmentation_metrics(final.argmax(1)[0].numpy(), target[0].numpy(), cfg["regions"], len(cfg["label_mapping"]))
        scores.append(metrics["brats_mean_dice"])
    return sum(losses) / len(losses), sum(scores) / len(scores)


def train_controller(records, splits, cfg):
    seed_everything(cfg["seed"])
    _, hashes = load_experts(cfg, splits)
    dataset = ControllerDataset(cfg, splits, hashes)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["num_workers"],
                        generator=torch.Generator().manual_seed(cfg["seed"]))
    device = device_for(cfg["device"])
    model = GatingNetwork(feature_channels(cfg), cfg["controller_width"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    amp = cfg["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    destination = Path(cfg["output_dir"]) / "checkpoints" / "controller"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "best.pt").exists():
        raise FileExistsError("Controller checkpoint exists; use a new output_dir")
    save_json(destination / "config.json", cfg)
    (destination / "architecture.txt").write_text(str(model))
    best = -1.
    for epoch in range(1, cfg["controller_epochs"] + 1):
        model.train()
        total = 0.
        optimizer.zero_grad(set_to_none=True)
        accumulation = cfg["gradient_accumulation"]
        for step, batch in enumerate(loader):
            batch = {k: value.to(device) for k, value in batch.items()}
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
        model.eval()
        val_loss, dice = validate_controller(model, cfg, splits, dataset.provenance, device)
        checkpoint = {"state_dict": model.cpu().state_dict(), "role": "controller", "name": "controller",
                      "protocol_hash": fingerprint(protocol(cfg)), "split_hash": fingerprint(splits),
                      "training_patients": splits["controller_train"], "expert_hashes": hashes,
                      "seed": cfg["seed"], "epoch": epoch, "validation_dice": dice, "architecture": str(model)}
        torch.save(checkpoint, destination / "last.pt")
        if dice > best:
            best = dice
            torch.save(checkpoint, destination / "best.pt")
        model.to(device)
        row = {"epoch": epoch, "seed": cfg["seed"], "train_loss": total / len(loader),
               "validation_loss": val_loss, "validation_dice": dice, "checkpoint_path": str(destination / "last.pt")}
        append_log(destination / "history.csv", row)
        print(f"controller: {row}", flush=True)
