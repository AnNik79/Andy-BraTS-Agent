from pathlib import Path
import csv
import time
import math
import json
import shutil
import torch
import torch.nn.functional as F
from ..config import seed_everything, device_for, save_json, protocol, fingerprint, EXPERTS, SEGMENTATION_EXPERTS, REASONING_EXPERT
from ..data.brats_dataset import BraTSPatches, load_patient, validate_training_audit, make_patch_loader
from ..data.preprocessing import geometry_targets
from ..experts import build_expert
from ..inference.patches import sliding_probabilities
from ..evaluation.metrics import segmentation_metrics, BRATS_REGIONS


def segmentation_loss(probabilities, target):
    p = probabilities.float().clamp_min(1e-7)
    ce = F.nll_loss(p.log(), target)
    one_hot = F.one_hot(target, p.shape[1]).movedim(-1, 1).float()
    axes = (0, 2, 3, 4)
    dice = (2 * (p * one_hot).sum(axes) + 1e-5) / (p.sum(axes) + one_hot.sum(axes) + 1e-5)
    return ce + 1 - dice[1:].mean()


def expert_loss(output, target, geometry=None):
    if "boundary_logits" in output["extra"] and geometry is None and target.device.type != "cpu":
        raise ValueError("Boundary geometry must be precomputed on CPU; refusing to synchronize device labels")
    loss = segmentation_loss(output["probabilities"], target)
    if "boundary_logits" not in output["extra"]:
        return loss
    if geometry is None:
        import numpy as np
        targets = [geometry_targets(label) for label in target.detach().numpy()]
        boundary = torch.from_numpy(np.stack([v[0] for v in targets]))[:, None]
        distance = torch.from_numpy(np.stack([v[1] for v in targets]))[:, None]
    else:
        boundary, distance = geometry
    boundary = boundary.to(device=output["extra"]["boundary_logits"].device, dtype=torch.float32)
    distance = distance.to(device=output["extra"]["distance_to_boundary"].device, dtype=torch.float32)
    positives = boundary.sum()
    pos_weight = ((boundary.numel() - positives) / positives.clamp_min(1)).clamp(1, 20)
    loss = loss + .5 * F.binary_cross_entropy_with_logits(output["extra"]["boundary_logits"].float(), boundary,
                                                          pos_weight=pos_weight)
    loss = loss + .2 * F.l1_loss(output["extra"]["distance_to_boundary"] / 16., distance)
    return loss


def training_output(model, image, name, cfg):
    """CNN loss does not consume summaries, argmax, top-k or entropy.

    Keep the public expert forward/prediction interface unchanged for debate.
    This path executes the same trainable layers and dropout calls.
    """
    if name == "cnn" and cfg.get("cnn_probability_only_training", False):
        return {"probabilities": model.segmentation_logits(image).float().softmax(1), "extra": {}}
    return model(image)


def append_log(path, row):
    exists = Path(path).exists()
    with open(path, "a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


from .checkpoint import save_training_checkpoint, load_training_checkpoint


class RunEvents:
    def __init__(self, destination, name, device, epochs):
        self.destination, self.name, self.device, self.epochs = destination, name, str(device), epochs
        self.started = time.monotonic()
        self.context = {"state": "running"}

    def __call__(self, message, **values):
        self.context.update({k: v for k, v in values.items() if k in
                             ("phase", "epoch", "step", "global_step", "case_index", "patient_id")})
        row = {**self.context, "event": message, "expert": self.name, "device": self.device,
               "elapsed_seconds": time.monotonic() - self.started, "epochs_requested": self.epochs, **values}
        with open(self.destination / "events.jsonl", "a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        temp = self.destination / "status.json.tmp"
        save_json(temp, row)
        temp.replace(self.destination / "status.json")
        print(json.dumps(row, allow_nan=False), flush=True)


@torch.inference_mode()
def validate_expert(model, validation, cfg, device, event, epoch):
    event("validation initialization begin", phase="validation", epoch=epoch)
    model.eval()
    running_loss, running_dice, num_cases = 0.0, 0.0, 0
    region_dice = {name: 0.0 for name in BRATS_REGIONS}
    pred_fg = true_fg = all_background = 0
    class_voxels = {i: 0 for i in range(len(cfg["label_mapping"]))}
    event("validation initialization complete", cases=len(validation))
    for index, record in enumerate(validation, 1):
        event("first validation batch begin" if index == 1 else "validation case begin",
              case_index=index, patient_id=record["patient_id"])
        event("validation patient load begin")
        patient = load_patient(record, cfg)
        event("validation patient load complete")
        def patch_event(message, **values):
            # Detailed tracing is optional; retain first/last patch and each tenth patch's progress.
            if cfg.get("trace_validation_transfers", False) or values.get("patch") in (1, values.get("patches")) or message == "validation patch progress":
                event(message, **values)
        patch = cfg["highres_patch_size"] if event.name == "highres" else cfg["crop_size"]
        probabilities = sliding_probabilities(model, torch.from_numpy(patient["image"])[None],
                                               patch, device, cfg["inference"]["overlap"], patch_event,
                                               **cfg.get("validation_performance", {}))
        event("validation CPU metrics begin")
        target = torch.from_numpy(patient["label"])[None]
        case_loss = float(segmentation_loss(probabilities, target).detach().item())
        if cfg.get("numpy_validation_argmax", False):
            prediction = probabilities.detach().numpy()[0].argmax(axis=0)
        else:
            prediction = probabilities.detach().argmax(1)[0].numpy()  # already CPU
        metrics = segmentation_metrics(prediction, patient["label"], cfg["regions"], len(cfg["label_mapping"]))
        case_dice = float(metrics["brats_mean_dice"])
        if not math.isfinite(case_loss) or not math.isfinite(case_dice):
            raise FloatingPointError("Nonfinite CPU validation statistic")
        running_loss += case_loss
        running_dice += case_dice
        for region in BRATS_REGIONS:
            region_dice[region] += float(metrics[region]["dice"])
        pred_count = int((prediction > 0).sum())
        pred_fg += pred_count
        true_fg += int((patient["label"] > 0).sum())
        all_background += int(pred_count == 0)
        for class_id in class_voxels:
            class_voxels[class_id] += int((prediction == class_id).sum())
        num_cases += 1
        del probabilities, target, prediction, metrics, patient
        event("validation CPU metrics complete", case_loss=case_loss, case_dice=case_dice)
        event("first validation batch complete" if index == 1 else "validation case complete", case_index=index)
        if index % 10 == 0 or index == len(validation):
            event("validation progress", completed_cases=index, total_cases=len(validation))
    event("validation metric aggregation begin")
    result = running_loss / num_cases, running_dice / num_cases
    summary = {
        "epoch": epoch, "cases": num_cases, "validation_loss": result[0], "brats_mean_dice": result[1],
        **{f"{name}_dice": region_dice[name] / num_cases for name in BRATS_REGIONS},
        "all_background_cases": all_background,
        "all_background_fraction": all_background / num_cases,
        "mean_predicted_foreground_voxels": pred_fg / num_cases,
        "mean_true_foreground_voxels": true_fg / num_cases,
        "predicted_class_voxel_totals": class_voxels,
        "hd95": False,
    }
    save_json(event.destination / f"validation_epoch_{epoch}.json", summary)
    event("validation metric aggregation complete", validation_loss=result[0], validation_dice=result[1],
          **{f"{name}_dice": summary[f"{name}_dice"] for name in BRATS_REGIONS},
          all_background_cases=all_background)
    return result


def train_expert(name, records, splits, cfg, resume=None, init_weights=None):
    if name == REASONING_EXPERT:
        raise ValueError("The LLM reasoning expert is not a trainable voxel network")
    if name not in SEGMENTATION_EXPERTS:
        raise ValueError(f"Unknown segmentation expert: {name}")
    validate_training_audit(cfg)
    if resume and init_weights:
        raise ValueError("Choose --resume or --init-weights, not both")
    if cfg.get("torch_threads"):
        torch.set_num_threads(cfg["torch_threads"])
    seed = cfg["seed"] + EXPERTS.index(name)
    seed_everything(seed)
    device = device_for(cfg["device"])
    if cfg.get("device") == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable; this debugging run must not silently use CPU")
    train = [r for r in records if r["patient_id"] in splits["expert_train"]]
    validation = [r for r in records if r["patient_id"] in splits["validation"]]
    if cfg.get("validation_case_ids"):
        selected = cfg["validation_case_ids"]
        if not set(selected) <= set(splits["validation"]):
            raise ValueError("Debug validation cases must belong to the frozen validation split")
        by_id = {r["patient_id"]: r for r in validation}
        validation = [by_id[pid] for pid in selected]
    if cfg.get("validation_max_cases"):
        validation = validation[:cfg["validation_max_cases"]]
    if not train or not validation:
        raise ValueError("Training and validation must be nonempty")
    dataset = BraTSPatches(train, cfg, specialist=name == "highres", geometry=name == "boundary")
    steps_per_epoch = math.ceil(len(dataset) / cfg["batch_size"])
    if cfg.get("max_train_steps"):
        steps_per_epoch = min(steps_per_epoch, cfg["max_train_steps"])
    model = build_expert(name, cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"])
    scheduler = None  # No scheduler in this experiment; checkpoint records that explicitly.
    amp = cfg["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    destination = Path(cfg["output_dir"]) / "checkpoints" / name
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "last.pt").exists() and not resume:
        raise FileExistsError("Existing run: supply --resume or use a fresh output_dir")
    if (destination / "best.pt").exists() and not resume:
        raise FileExistsError("Existing best checkpoint: supply --resume or use a fresh output_dir")
    event = RunEvents(destination, name, device, cfg["epochs"])
    save_json(destination / ("resume_config.json" if resume else "config.json"), cfg)
    (destination / "architecture.txt").write_text(str(model))
    best, global_step, start_epoch = -1.0, 0, 1
    restored = None
    if resume:
        event("checkpoint resume begin", path=str(resume))
        restored = load_training_checkpoint(resume, model, optimizer, scaler, cfg, splits, name, device, scheduler)
        best, global_step = float(restored["best_validation_metric"]), int(restored["global_step"])
        # A resumed run may write to a fresh directory. Preserve its historical
        # best model without replacing the live progress model/optimizer.
        previous_best = Path(resume).parent / "best.pt"
        if not (destination / "best.pt").exists() and previous_best.exists():
            candidate = torch.load(previous_best, map_location="cpu", weights_only=True)
            compatible = all(candidate.get(k) == restored.get(k) for k in
                             ("name", "protocol_hash", "split_hash", "training_signature", "best_validation_metric"))
            if compatible and candidate.get("completed_epoch", 0) <= restored["completed_epoch"]:
                event("historical best checkpoint copy begin", path=str(previous_best))
                temporary = destination / "best.pt.tmp"
                shutil.copyfile(previous_best, temporary)
                temporary.replace(destination / "best.pt")
                event("historical best checkpoint copy complete")
            del candidate
        start_epoch = restored["epoch"] + int(restored["completed_epoch"] >= restored["epoch"])
        event("checkpoint resume complete", epoch=start_epoch, global_step=global_step,
              completed_epoch=restored["completed_epoch"], next_batch=restored["next_batch"])
    elif init_weights:
        weights = torch.load(init_weights, map_location="cpu", weights_only=True)
        if weights["name"] != name or weights["protocol_hash"] != fingerprint(protocol(cfg)) or weights["split_hash"] != fingerprint(splits):
            raise ValueError("Initial weights do not match expert/protocol/splits")
        model.load_state_dict(weights["state_dict"])
        event("weights-only initialization; optimizer and epoch start fresh", path=str(init_weights))
        del weights
    event("training initialized", phase="training", epoch=start_epoch, global_step=global_step,
          train_cases=len(train), validation_cases=len(validation), steps_per_epoch=steps_per_epoch,
          num_workers=cfg["num_workers"], persistent_workers=False)
    stale = 0
    last_completed = start_epoch - 1
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        epoch_started = time.monotonic()
        running_loss, num_batches, next_batch = 0.0, 0, 0
        # Per-epoch sampler/loader generators are independent of dropout/augmentation RNG.
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
        indices = indices[:steps_per_epoch * cfg["batch_size"]]
        if restored is not None and restored["epoch"] == epoch and restored["completed_epoch"] < epoch:
            running_loss = float(restored["running_loss"])
            num_batches, next_batch = int(restored["num_batches"]), int(restored["next_batch"])
            if indices != restored["epoch_indices"]:
                raise ValueError("Resumed patient/patch order differs from checkpoint")
        restored = None  # Do not retain a second model/optimizer snapshot through the epoch.
        loader = make_patch_loader(dataset, indices[next_batch * cfg["batch_size"]:], cfg,
                                   seed + epoch + 100000)
        optimizer.zero_grad(set_to_none=True)
        accumulation = cfg["gradient_accumulation"]
        def cursor(training_complete=False, completed_epoch=None):
            return {"epoch": epoch, "completed_epoch": epoch - 1 if completed_epoch is None else completed_epoch,
                    "global_step": global_step, "next_batch": num_batches,
                    "training_loop_complete": training_complete, "running_loss": float(running_loss),
                    "num_batches": num_batches, "steps_per_epoch": steps_per_epoch, "epoch_indices": indices}
        for step, batch in enumerate(loader, start=next_batch):
            if name == "boundary":
                image, target, boundary, distance = batch
            else:
                image, target = batch
                boundary = distance = None
            final_steps = step >= steps_per_epoch - 10
            if final_steps:
                event("final training step begin", phase="training", epoch=epoch, step=step + 1, steps_per_epoch=steps_per_epoch)
            group_start = (step // accumulation) * accumulation
            divisor = min(accumulation, steps_per_epoch - group_start)
            device_image, device_target = image.to(device), target.to(device)
            geometry = None if boundary is None else (boundary.to(device), distance.to(device))
            with torch.autocast(device_type=device.type, enabled=amp):
                output = training_output(model, device_image, name, cfg)
                loss = expert_loss(output, device_target, geometry)
            if final_steps:
                event("training scalar extraction begin", step=step + 1)
            step_loss = float(loss.detach().item())  # The ONLY training statistic retained is a Python float.
            if final_steps:
                event("training scalar extraction complete", step=step + 1, loss=step_loss)
            if not math.isfinite(step_loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, step {step + 1}")
            scaler.scale(loss / divisor).backward()
            updated = (step + 1) % accumulation == 0 or step + 1 == steps_per_epoch
            if updated:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            running_loss += step_loss
            num_batches += 1
            # Release the final batch's graph/output as well, before entering any boundary code.
            del loss, output, device_image, device_target, image, target, boundary, distance, geometry
            if final_steps or step == 0 or (step + 1) % cfg.get("log_every_steps", 10) == 0:
                event("training step complete", phase="training", epoch=epoch, step=step + 1,
                      steps_per_epoch=steps_per_epoch, global_step=global_step, loss=step_loss)
            if updated and (step + 1) % cfg.get("save_every_steps", 100) == 0 and step + 1 < steps_per_epoch:
                save_training_checkpoint(destination / "progress.pt", model, optimizer, scaler, cfg, splits,
                                         name, seed, cursor(), best, device, event, scheduler)
        event("epoch training loop complete", epoch=epoch, step=num_batches, global_step=global_step)
        del loader
        event("loss aggregation begin")
        epoch_loss = running_loss / num_batches
        assert isinstance(running_loss, float) and isinstance(epoch_loss, float)
        event("loss aggregation complete", epoch_loss=epoch_loss, num_batches=num_batches)
        # Save recoverable training state BEFORE expensive validation, without model.cpu()/model.to().
        save_training_checkpoint(destination / "epoch_train_complete.pt", model, optimizer, scaler, cfg, splits,
                                 name, seed, cursor(True), best, device, event, scheduler)
        every = cfg.get("validation_every_epochs", 1)
        run_validation = ((cfg.get("validate_first_epoch", True) and epoch == 1)
                          or epoch == cfg["epochs"] or epoch % every == 0)
        val_loss, dice, improved = None, None, False
        if run_validation:
            val_loss, dice = validate_expert(model, validation, cfg, device, event, epoch)
            improved = dice > best
            best = max(best, dice)
        if scheduler:
            scheduler.step()
        save_training_checkpoint(destination / "last.pt", model, optimizer, scaler, cfg, splits, name,
                                 seed, cursor(True, epoch), best, device, event, scheduler, validation_dice=dice)
        if improved:
            event("best checkpoint save begin")
            temporary = destination / "best.pt.tmp"
            shutil.copyfile(destination / "last.pt", temporary)
            temporary.replace(destination / "best.pt")
            event("best checkpoint save complete")
        row = {"epoch": epoch, "seed": seed, "train_loss": epoch_loss, "validation_loss": val_loss,
               "validation_dice": dice, "learning_rate": optimizer.param_groups[0]["lr"],
               "epoch_seconds": time.monotonic() - epoch_started,
               "checkpoint_path": str(destination / "last.pt")}
        append_log(destination / "history.csv", row)
        event("epoch boundary complete", phase="epoch_complete", epoch=epoch, global_step=global_step,
              train_loss=epoch_loss, validation_loss=val_loss, validation_dice=dice,
              learning_rate=row["learning_rate"], epoch_seconds=row["epoch_seconds"])
        last_completed = epoch
        patience = cfg.get("early_stopping_patience", 0)
        if run_validation and patience:
            stale = 0 if improved else stale + 1
            if stale >= patience:
                event("early stopping", epoch=epoch, best_validation_dice=best, patience=patience)
                break
    event("training completed", phase="completed", state="completed", global_step=global_step,
          completed_epoch=last_completed, best_validation_dice=best)
    return destination / "best.pt"
