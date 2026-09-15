# BraTS 2023 GLI real-data workflow

The designated extracted source is `data/BraTS-GLI/train/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/<case_id>/`. Initial inspection found only 128 extracted case directories, 636 files, and an incomplete `BraTS-GLI-00336-000-t1c.nii.gz`. The complete local official training ZIP contains 1,251 cases and 6,255 NIfTI members. Finishing extraction would leave less than 1 GB of free space on the inspected filesystem.

`configs/brats.yaml` therefore uses `dataset_archive` to read the complete training ZIP directly, without another full copy. The incomplete extraction is preserved and bypassed. No files from the official validation archive are used for quantitative evaluation, training, or this audit. No source images were modified or deleted.

## Audit

```bash
python scripts/audit_real_brats.py \
  --archive data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip \
  --extracted-root data/BraTS-GLI/train \
  --output outputs/brats2023_gli_audit --workers 2
python scripts/visualize_real_brats.py \
  --archive data/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip \
  --output outputs/brats2023_gli_audit/sanity
```

The exhaustive audit reads every member completely, checks outer ZIP and inner gzip CRCs, parses every NIfTI, and inspects every voxel for finite values. It records shapes, full affine matrices, spacing, orientation, NIfTI spatial units, qform/sform consistency, source dtype, MRI nonzero statistics, mask labels/counts, and voxel-content SHA-256 digests. Missing, corrupt, empty/constant MRI, and misaligned cases are reported explicitly. It also checks label coverage outside the union of nonzero MRI support and detects identical multimodal image sets.

`cases.jsonl` contains per-file evidence; `report.json` contains aggregate results. `extracted_inventory.json` describes the incomplete disk copy separately from the archive. The source archive's size and modification timestamp must remain unchanged. Masks are inspected **before** applying any label conversion. The observed mapping is identity: `0→0, 1→1, 2→2, 3→3`. The modalities are exactly `t1n, t1c, t2w, t2f`, with `seg` as the target. Numeric mask values establish the label mapping; they do not by themselves establish tissue semantics.

Five figures in `sanity/` each show all four modalities, source segmentation, and a t2f overlay in three native planes selected to contain tumor. `preprocessing_checks.json` records source labels, selected slices, tumor bounding boxes, and normalized foreground mean/std with background-preservation checks. After reviewing all five figures, the audit report must contain `sanity_visualizations_passed: true`; rerunning the exhaustive audit resets that approval. The current training guard refuses to run before both the exhaustive audit and visual review pass.

## Splits

The split seed is **42**. IDs are sorted before seeded group assignment. Scans such as `BraTS-GLI-00045-000` and `BraTS-GLI-00045-001` share the conservative subject key `BraTS-GLI-00045`. The 1,251 cases produce 1,133 such groups; 118 groups contain two scans. The supplied mapping spreadsheet contains entries for all 1,251 cases. Shared prefixes are grouped conservatively; this is not a claim of independently verified biological identity from imaging alone.

| Role | Subject-prefix groups | Scan cases |
|---|---:|---:|
| Training, total | 849 | 930 |
| CNN/expert optimization subset | 622 | 682 |
| Reserved controller subset | 227 | 248 |
| Internal validation | 170 | 191 |
| Internal test | 114 | 130 |
| Total | 1,133 | 1,251 |

`configs/splits/brats2023_gli_subjects.json` contains the three primary lists of subject IDs. `brats2023_gli_roles.json` records the expert/controller partition. `brats2023_gli_cases.json` contains the four corresponding scan-ID lists used by the pipeline. Existing splits are validated against the complete inventory and rejected if a subject group crosses roles. No official validation cases are mixed in.

Previous real-BraTS CNN experiment directories were cleared for a fresh GPU campaign. Do not resume an old `best.pt` / `last.pt`. Use a new `output_dir`.

## CNN only

Once the audit and visual review pass:

```bash
python scripts/train_experts.py --config configs/brats.yaml --expert cnn
# Or start a durable background job with a log/PID:
python scripts/start_cnn_training.py --config configs/brats.yaml
```

Do not run both commands simultaneously. The background wrapper launches only the CNN and records terminal errors. `checkpoints/cnn/launch.json` records the PID and command; `training.log` and `status.json` report live progress. On this Mac, MPS is available outside the restricted execution sandbox. MPS uses float32, with PyTorch CPU fallback enabled for unsupported operations by the background launcher. CUDA AMP remains optional if the experiment is run on CUDA hardware.

The configured run requests 50 epochs, 64³ native-voxel crops, batch size 1, four sampled patches per training case, learning rate 0.0003, and seed 42. Following the MPS investigation, it uses zero workers, probability-only full-volume validation, and resumable format-2 expert checkpoints. `best.pt` is selected only after full validation; `epoch_train_complete.pt` allows validation retry without repeating training. `history.csv` is written when an epoch completes. See [the MPS investigation](mps_epoch_boundary_debug.md) for the three passing boundary tests, commands, and exact resume behavior.

The default run does not train the transformer, boundary expert, high-resolution expert, or controller, and does not evaluate the internal test set.
