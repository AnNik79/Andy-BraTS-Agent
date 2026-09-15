# BraTS expert disagreement research beta

The system has **five experts**: four independently trained 3D segmentation experts (CNN, Transformer, Boundary / Sum Medical, High Resolution) plus an LLM reasoning expert. A disagreement layer compares the four imaging experts. A learned controller gates those four voxel models. The LLM reasons over structured debate evidence and cannot change the segmentation mask. Average and majority vote are experimental baselines only.

This is a research prototype, not a clinically validated system or a cancer diagnosis tool. Expert disagreement is a model statistic, not established medical uncertainty. Real BraTS 2023 GLI data is now configured; see [the real-data workflow](docs/real_data_workflow.md) for the complete training archive, exhaustive audit, patient grouping, and CNN-only run. Synthetic checks verify software behavior only; they provide no evidence of segmentation quality or controller improvement.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
# Optional MONAI Swin UNETR backend:
python -m pip install -e '.[monai]'
pytest -q
```

Python 3.10+ and PyTorch 2.2+ are required. CPU, CUDA, and MPS are selected automatically. CUDA AMP is optional; CPU/MPS use float32. This workspace was verified with Python 3.12/PyTorch 2.10 on CPU; see `docs/verification.md` for actual verification details.

## Configure and inspect real data

The current **`configs/brats.yaml`** points to BraTS 2023 GLI and uses `dataset_archive` because the extracted copy is incomplete. For a different, fully extracted dataset, clear `dataset_archive`, `audit_report`, and `split_file`, set `require_dataset_audit: false`, and set `dataset_root` to its patient folders after reviewing its structure. Relative paths resolve from the YAML file, not the shell's working directory.

```yaml
dataset_root: /path/to/your/extracted/BraTS
output_dir: ../outputs/experiment_001
```

The discovery code inspects NIfTI filenames and matches the final `_`/`-` suffix using configured aliases. It recognizes common names such as `patient_t1.nii.gz`, `patient_t1ce.nii.gz`, and `patient-t1n.nii.gz`/`patient-t1c.nii.gz`. It does not infer a BraTS release. All modalities for an automatically discovered patient must be in one directory; that directory name is its unique patient ID. Ambiguous, missing, and duplicate files are rejected. Do not point the root at predictions or a mixture of releases.

For other layouts, set `manifest` to a CSV path (relative to the YAML) with:

```csv
patient_id,t1,t1ce,t2,flair,seg
patient001,patient001/t1.nii.gz,patient001/t1ce.nii.gz,patient001/t2.nii.gz,patient001/flair.nii.gz,patient001/seg.nii.gz
```

Modality column names follow `modalities`; data paths in the CSV resolve relative to `dataset_root`. For repeated scans, configure a reviewed `patient_group_regex` whose first capture group identifies the subject; all scans in a group remain in one split. The real GLI config groups the shared `BraTS-GLI-NNNNN` prefix. Automatic discovery cannot establish biological identity from arbitrary unknown filenames.

**Verify the label mapping for your release.** The real 2023 GLI configuration uses the directly observed source labels `0,1,2,3` with identity conversion. Older synthetic fixtures retain their own explicit `0,1,2,4 → 0,1,2,3` convention and do not control the real configuration. Region definitions use internal class IDs. Predictions are converted back to source labels on export. Unknown labels and noninteger masks are rejected, rather than guessed.

```bash
python scripts/audit_dataset.py --config configs/brats.yaml
```

This validates every image's 3D shape, affine alignment, finite intensities, and label values and writes `dataset_audit.json` and `splits.json`. Misaligned modalities must be corrected upstream; the loader will not silently register them. Native image axes, spacing, and orientation are retained, not assumed to be anatomical axial coordinates. NIfTI arrays and affine metadata follow [nibabel's image model](https://nipy.org/nibabel/nibabel_images.html).

At least four patients are required. Four disjoint patient sets are reserved: `expert_train`, `controller_train`, `validation`, `test`. Automatic allocation reserves one patient per set and apportions the remaining patients by the configured fractions. For a real study, use a reviewed `split_file` JSON with these four patient-ID lists. An existing split is never silently regenerated. All patients must be represented exactly once. Expert training and controller optimization use separate patient sets. Validation selects checkpoints for both; the test set is only evaluated after validation has selected the best individual expert.

## Run the staged experiment

```bash
python scripts/train_experts.py --config configs/brats.yaml --expert cnn
python scripts/train_experts.py --config configs/brats.yaml --expert transformer
python scripts/evaluate.py --config configs/brats.yaml --initial
```

Inspect `evaluation/initial_cnn_transformer/` before continuing: per-expert Dice, recall, precision, pairwise error overlap, disagreement/error AUROC, NIfTI disagreement maps, and visualizations. This is the first real-data research checkpoint. Different architectures alone do not prove meaningfully different errors; evaluate the exported error-diversity statistics.

```bash
python scripts/train_experts.py --config configs/brats.yaml --expert boundary
python scripts/train_experts.py --config configs/brats.yaml --expert highres
python scripts/generate_controller_data.py --config configs/brats.yaml
python scripts/train_controller.py --config configs/brats.yaml
python scripts/evaluate.py --config configs/brats.yaml --split validation
python scripts/evaluate.py --config configs/brats.yaml --split test
python scripts/run_inference.py --config configs/brats.yaml --patient /path/to/one/patient
```

Alternatively, `train_experts.py` without `--expert` runs CNN, transformer, the initial comparison, then boundary and highres in that order. Training refuses to overwrite an existing best checkpoint without `--resume`; use a fresh `output_dir` for a new experiment. Expert training supports `--resume <checkpoint>` for format-2 training state and `--init-weights <checkpoint>` for an explicitly new run from weights. See [MPS debugging and resume details](docs/mps_epoch_boundary_debug.md).

## Models and processing

| Component | Role | Implementation |
|---|---|---|
| CNN | Segmentation expert 1 | Small two-downsampling-stage 3D U-Net with independent weights |
| Transformer | Segmentation expert 2 | Small native PyTorch global-token transformer with learned spatial positions and convolutional decoder, or optional MONAI Swin UNETR |
| Boundary | Segmentation expert 3; professor's **Sum Medical** geometry role | Independently initialized U-Net; class-interface BCE and truncated unsigned distance auxiliary loss. Internal key remains `boundary`. |
| High resolution | Segmentation expert 4 | No-pooling dilated 3D CNN, trained on native-voxel boundary, boundary-band, tumor, and background patches |
| LLM | Reasoning expert 5 | `LLMReasoningExpert`: structured debate evidence in, structured reasoning out. Not a voxel network. |
| Disagreement | Non-expert system component | Class probability variance, predictive entropy, class votes, pairwise hard disagreement and total-variation distance, normalized Jensen–Shannon map |
| Controller | Non-expert system component | Two 1×1×1 convolutions and GELU; masked softmax over the **four** segmentation experts |

All experts return `logits`, `probabilities`, `segmentation`, `confidence`, `uncertainty`, `features`, and `extra`. Compact features are deterministic channel summaries of trained decoder activations. Uncertainty includes normalized predictive entropy, top-class probability, and top-two margin. `mc_samples > 1` adds dropout-based mutual information. Confidence is not calibrated, and entropy is not an estimate of clinical ambiguity. Logits returned by stochastic/sliding inference are log mean probabilities, not averaged raw network logits.

Each modality is normalized independently over its nonzero voxels. Training uses crop/pad metadata, shared spatial flips, and modest intensity scaling/shifts/noise that preserve background. Patch coordinates stay in native voxel space; there is no downsampling of the highres input. Its distinction is patch size, no pooling, and targeted training, not newly acquired image detail. Its training hard-region proxy is an interface band; this beta does not implement learned online hard-example mining.

Inference tiles the **entire native volume** for the first three experts, with overlap averaging of each expert's own patch probabilities. This prevents central crop truncation. The highres specialist scores candidate patches by tumor/entropy proposals from the first three experts, with configurable patch cap. Coverage is explicit: uncovered voxels abstain, have uniform probabilities and zero controller weight, and are excluded from disagreement pairs, voting, and probability averaging. The standalone highres evaluation separately covers the full volume without other experts' proposals. `highres_seg` uses background placeholders outside coverage; `highres_full_seg` is the fair standalone baseline.

The disagreement map is the mean of normalized Jensen–Shannon divergence and mean valid-pair total-variation distance, in [0,1]. Identical predictions give zero, even if all experts are uncertain. It is separate from predictive entropy. Thresholds must be fixed using validation, not tuned on test results.

Set `model.transformer_backend: swin` for MONAI Swin UNETR. It requires the optional dependency and crop dimensions of at least 64, divisible by 32 in this wrapper, to accommodate downsampling and instance normalization. The compact `tiny` backend is the default for limited hardware; it is a prototype global-context architecture, not a reproduction of published Swin performance. The Swin constraints and implementation derive from [MONAI's architecture documentation](https://monai.readthedocs.io/en/latest/networks.html).

## Leakage prevention and compute

Controller caches contain only `controller_train` and `validation` patients. Frozen expert outputs are generated without reading those labels into expert inference; labels are used afterward as controller targets. Cache and checkpoint metadata bind the configuration protocol, split IDs, expert identities, training IDs, and SHA-256 hashes of the expert checkpoint files. Changed experts invalidate controller caches/checkpoints. The final test comparison rejects stale validation reports and uses the best individual expert selected on validation, rather than selecting an expert on test scores.

Do not change source patient data in place after training; data content itself is not cryptographically versioned. Archive the audit, manifest, split, source dataset version, config, and outputs together. Dataset identity across repeated scans remains the responsibility of the reviewed manifest.

Tune `crop_size`, `highres_patch_size`, `batch_size`, `num_workers`, `model.width`, transformer size, `mixed_precision`, and `gradient_accumulation` for hardware. Models are moved to the accelerator one at a time for inference; full outputs accumulate on CPU and the pointwise gate runs in slabs. Native-volume output features still consume substantial host RAM. Controller caches store full-volume float16 features/probabilities and may require several GB per real patient; patch loading currently reads one patient cache at a time. This favors a straightforward beta over optimized disk streaming. Use zero workers initially to avoid multiplying host memory. Reducing expert feature count reduces memory.

Seeds cover Python, NumPy, PyTorch, independent expert seeds, and DataLoader generators. CUDA cuDNN benchmarking is disabled and deterministic mode requested. Exact reproducibility across GPU/MPS devices, library versions, and all GPU kernels is not guaranteed. New expert checkpoints include optimizer, scaler, epoch/step, and RNG state; legacy expert and controller checkpoints remain weights/selection snapshots.

## Outputs and interpretation

```text
output_dir/
  dataset_audit.json
  splits.json
  checkpoints/<cnn|transformer|boundary|highres|controller>/
    config.json, architecture.txt, history.csv, best.pt, last.pt
  controller_cache/<patient_id>.pt
  evaluation/<initial_cnn_transformer|validation|test>/
    metrics.csv, aggregate_metrics.csv, disagreement_errors.csv
    expert_error_diversity.csv, report.json, report.md
  predictions/<patient_id>/
    cnn_seg.nii.gz, transformer_seg.nii.gz, boundary_seg.nii.gz
    highres_seg.nii.gz, highres_coverage.nii.gz
    highres_full_seg.nii.gz                 # evaluation only
    average_seg.nii.gz, vote_seg.nii.gz, final_seg.nii.gz
    disagreement.nii.gz, weight_<expert>.nii.gz
    expert_summary.json, llm_reasoning.json, reasoning.txt, visualization.png
```

All NIfTI maps retain the original volume dimensions and affine. Visualization selects a tumor-containing slice from ground truth if available, otherwise the final prediction. It labels native voxel-axis slices without inventing anatomical directions. Weight maps can be disabled. JSON includes connected disagreement-region size/centroid, tumor/core/edema/boundary statistics, pairwise disagreement, expert uncertainty, coverage, weights, and predicted volumes in mL.

Metrics include every foreground class and configured region. `mean_dice` averages the configured regions (including edema in the example); it is explicitly not a background-inclusive accuracy score. Both-empty binary masks score one; one-sided empty masks score zero. Optional HD95 is in mm; it is zero for both-empty masks and `null`/blank for a one-sided empty mask. Aggregate CSV means omit undefined HD95 values: consult individual rows before comparison. Vote ties resolve to the lowest internal class ID.

Disagreement/error analysis excludes MRI background while retaining ground-truth tumor. It reports errors/correct voxels, one-voxel error neighborhoods, high/low disagreement error rates, and error-detection AUROC for every method. Patient-level Spearman correlation is omitted if there are too few patients or constant values. Voxel statistics are descriptive; spatially correlated voxels are not treated as independent significance-test samples. `expert_error_diversity.csv` reports overlapping and complementary errors. These checks test the research hypotheses without assuming the controller wins.

The LLM is the fifth expert and sits beside the controller. It receives a structured compression of expert confidence, uncertainty, pairwise disagreement, disagreement-region blobs (centroids, bounding boxes, sizes), boundary-interface disagreement, high-resolution coverage, and controller weights. The spatial disagreement map remains in the pipeline as `disagreement.nii.gz`; it is not dumped voxel-wise into the prompt. The LLM returns structured reasoning (`llm_reasoning.json`) plus readable text. It cannot modify voxels.

Set `reasoning.mode: llm` and provide `BRATS_LLM_API_KEY` (optional `BRATS_LLM_BASE_URL`, `BRATS_LLM_MODEL` or `reasoning.model`) for a real OpenAI-compatible chat completion. If real-LLM mode is requested without a key, the run fails clearly. `reasoning.mode: deterministic` is an explicit test fallback and is labeled as **not** an LLM. Unusual/difficult regions are flagged from disagreement and uncertainty; this is not an anatomical abnormality detector.

For the measured local M2 Pro CNN optimizations, benchmark evidence, and numerical
resume checks, see [the MPS performance report](docs/mps_performance.md). Those
engineering measurements remain; the previous real-BraTS CNN checkpoints were
deleted for a clean GPU retraining campaign. Use a fresh `output_dir`. Performance
dependencies are available with `pip install -e '.[performance]'`.

## Synthetic verification

```bash
python scripts/smoke_test.py --output outputs/my_synthetic_check
```

The output directory must be empty. This creates eight small artificial multimodal NIfTI patients, trains each expert and the gate for one epoch, runs the CNN/transformer comparison before the specialists, evaluates validation then test, and exports the requested patient artifacts. `SMOKE_RESULT.json` and `SYNTHETIC_ONLY.txt` label the result. Do not report those Dice values as BraTS results.
