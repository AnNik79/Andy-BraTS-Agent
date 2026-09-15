# CNN MPS epoch-boundary investigation

## Finding and limits

No all-epoch GPU loss/graph retention was found. The previous trainer used `total += float(loss.detach())` (old line 94); validation stored Python floats. It retained the final batch's `loss` object until reassignment, but not 2,728 graphs. The new loop explicitly releases that last batch and uses `step_loss = float(loss.detach().item())`, a Python `running_loss`, and an integer `num_batches`.

The supplied `/private/tmp/cnn_sample.txt` shows `THPVariable_cpu → mps_copy → copy_and_sync → MTLCommandBuffer waitUntilCompleted`. The preserved `status.json` places this run in **validation case 33, BraTS-GLI-00291-000**, not loss aggregation or checkpoint serialization. No live old training process remained during this investigation.

The reproduced synchronization hotspot is **old `src/brats_debate/inference/patches.py:50`**:

```python
accum[(..., *sl)] += out["probabilities"].cpu()
```

Using the preserved old `Expert.predict` implementation, the old epoch-1/step-2700 weights, and three native-volume patches from that case, the probability CPU copy waited **3.083, 2.837, and 2.832 seconds**. Subsequent feature/MI copies took 0.3–1.1 milliseconds. The first copy waits for the already-enqueued GPU prediction work, not merely its own 4 MB memory copy.

The old single-sample prediction path calculated uncertainty three times: once in `self(image)`, once for mutual information, and once in the final `pack_output`. A separate synthetic-patch stage probe measured about **0.946 s** for entropy/top-k/margin calculation and **0.929 s** for another `pack_output`. Singleton stack/mean operations measured only 0.0004–0.008 s in that probe. They were redundant, but were not evidence of an epoch-length tensor list.

The old full validation workload is 191 cases × 196 windows = 37,436 windows. A simple extrapolation of the three legacy copy timings alone is around 30 hours; this is not a measured full validation duration. The old stdout log had no validation-case/patch progress, explaining why the last printed training step could remain 2720/2728 while validation progressed.

**An indefinite historical Metal deadlock was not reproduced.** The native sample cannot uniquely identify the historical Python `.cpu()` call among old lines 50, 51, and 58, or establish a driver-level root cause. The probability copy is the matching hotspot reproduced here, and the saved case-33 status plus multi-second waits support a very slow, poorly instrumented validation explanation. The new short tests do not prove that all long-run failure modes are eliminated.

## Operations audited

Line references in this table refer to copies preserved under `outputs/mps_boundary_debug/evidence/`.

| Original location | Operation | Lifetime/device finding |
|---|---|---|
| train_expert.py:86–98 | Loss, `isfinite`, `float(loss.detach())` | Scalar host reads can synchronize; accumulator already a Python float. Final `loss` reference survived the loop. No loss tensor list. |
| train_expert.py:101–105 | Per-parameter `.detach().cpu()`, progress serialization | Every 100 steps; CPU snapshots, not graph accumulation. Legacy snapshot had no optimizer/RNG state. |
| train_expert.py:106–112 | `eval`, float lists, validation startup | No model transfer; patient images load on CPU. Status advanced to case 33. |
| patches.py:30 | `image.cpu()` | Input already CPU in this call path; no MRI-volume GPU transfer. |
| patches.py:46 | Window `.to(device)` and `predict` | One GPU patch at a time; original CPU window could be strided. |
| base.py:35–41 | MC output list, `stack().mean()` | Scoped to one prediction patch, with `mc_samples=1`; never the entire epoch. Repeated uncertainty computation was expensive on this MPS setup. |
| patches.py:50/51/58 | Probability, feature, extra `.cpu()` | Per-patch synchronization; probability copy reproduced the long wait. Unneeded feature/uncertainty transfers during training validation. |
| train_expert.py:117–120 | CPU loss `.mean`, `.numpy`, float metric lists | Predictions already CPU; no epoch GPU metric list or GPU aggregate. |
| train_expert.py:121–129 | `model.cpu().state_dict()`, save, `model.to(device)` | After all validation cases, not before case 33. Avoidable migration of the live model. |
| train_expert.py:28–35 | Boundary-target CPU/NumPy conversion | Per-batch auxiliary boundary loss; not used by the CNN. |

There were no explicit `torch.mps.synchronize()` calls, and none were added.

## Changes

- Training statistics are detached Python scalars. Batch outputs, losses, and device inputs are deleted before boundary operations. Confidence/uncertainty outputs are detached from autograd.
- Every final ten training steps is logged, including 2728/2728 in the full run, followed immediately by `epoch training loop complete`.
- Structured `events.jsonl`, stdout, and atomic `status.json` report begin/end markers for loss aggregation, checkpoint snapshot/serialization, validation initialization, first validation batch, CPU metrics, and boundary completion. Validation reports every ten cases and every ten windows. Detailed per-transfer tracing is enabled in the reproduction config.
- Validation uses `torch.inference_mode()` and a segmentation-only forward path. It copies one detached, contiguous probability patch to CPU, then discards its GPU tensor. No auxiliary features/entropy/MC outputs are requested or accumulated. Full-volume reconstruction and metrics remain on CPU and are discarded after each case.
- Single-sample general expert prediction no longer builds singleton MC stacks or repeats uncertainty computation. The multi-sample MC interface remains available.
- The live model stays on its training device. Detached copies of model and optimizer state are serialized atomically on CPU. A complete training-state checkpoint is saved **before validation**.
- MPS debugging and restarted full-run DataLoaders use zero workers and explicit `persistent_workers=False`.

## MPS tests

`configs/brats_mps_repro.yaml` uses the real CNN architecture, 64³ crops, the audited real archive and original patient split, 20 training batches per epoch, and two full native 240×240×155 validation volumes. It includes the case recorded in the old stalled status. Three epochs ran consecutively in one MPS process; validation was not replaced by smaller toy crops.

| Crossing | Training steps | Validation cases/windows | Boundary elapsed | Result |
|---|---:|---:|---:|---|
| 1 | 20 | 2 / 392 | 32.65 s | Passed |
| 2 | 20 | 2 / 392 | 32.67 s | Passed |
| 3 | 20 | 2 / 392 | 32.60 s | Passed |

Median probability-copy waits were 0.064 s, with maximum observed waits below 0.100 s. All final-ten-step, aggregation, checkpoint, first-case, and transfer completion markers were checked. The run saved completed epoch 3/global step 60. An additional MPS process resumed that checkpoint and completed epoch 4/global step 80, including two real validation cases.

**31 unit tests pass.** Resume tests compare CPU uninterrupted training with mid-epoch, pre-validation, and completed-epoch resumes; weights, optimizer state, RNG state, and best metric match bit for bit. Additional tests verify detached snapshots have no live tensor aliases and that probability-only validation matches the standard prediction probabilities.

Evidence: `outputs/mps_boundary_debug/results.json`, `reproduction/run.log`, `resume_check/run.log`, and `evidence/legacy_probe.log` / `legacy_stages.log`. Original source files, supplied sample, and old status were preserved in `evidence/`.

## Resume semantics and commands

Format-2 checkpoints contain model, optimizer, optional scheduler (explicitly `None` here), AMP scaler, completed epoch, current epoch, global optimizer step, best validation metric, patch-order/cursor, scalar loss totals, and Python/NumPy/PyTorch CPU/MPS or CUDA RNG states. Resume rejects changed splits, model/data protocol, batch/accumulation settings, or debugging limits. Exact mid-epoch augmentation replay requires zero workers. Cross-device/library bitwise reproducibility is not promised.

- `progress.pt`: last saved optimizer-update boundary within the epoch.
- `epoch_train_complete.pt`: training finished; resume skips those training batches and retries validation.
- `last.pt`: completed training and validation epoch.
- `best.pt`: copy of the best fully validated checkpoint.

The old `outputs/brats2023_gli_cnn/checkpoints/cnn/progress.pt` contains **weights only**, at epoch 1/step 2700. Missing optimizer/RNG state cannot be recovered. `--resume` rejects it; `--init-weights` explicitly initializes a new optimizer/run if desired. The restarted full run begins fresh with seed 42 and preserves the old directory.

Exact full-run launch used after the checks passed:

```bash
PYTHONPATH=src:.venv-deps MPLCONFIGDIR=/private/tmp/brats-mpl \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python scripts/start_cnn_training.py --config configs/brats.yaml
```

For an installed virtual environment, omit `PYTHONPATH=src:.venv-deps`. MPS must be available to the process; in this session that requires execution outside the restricted sandbox. The launcher enables PyTorch's CPU fallback for unsupported MPS operations.

After a stopped run, resume from the latest appropriate new-format checkpoint:

```bash
PYTHONPATH=src:.venv-deps MPLCONFIGDIR=/private/tmp/brats-mpl \
  python scripts/start_cnn_training.py --config configs/brats.yaml \
  --resume outputs/brats2023_gli_cnn_mps_fixed/checkpoints/cnn/epoch_train_complete.pt
```

Use `last.pt` after a completed validation epoch. Do not start a second process while the job is running. Live output is under `outputs/brats2023_gli_cnn_mps_fixed/checkpoints/cnn/`.

PyTorch documents why an optimizer state dictionary is necessary for training resume in its [checkpoint guide](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
