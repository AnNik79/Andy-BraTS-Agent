# CancerAgents Beta 1

Beta 1 is complete as a frozen research baseline. The 130-case BraTS 2023 test was scored once with the validation-selected controller. No weights, thresholds, or epoch were changed after those scores were known.

This is a research prototype. The Dice and HD95 numbers below are overlap and surface-distance scores on one held-out split. They are not an accuracy percentage and they are not evidence of clinical effectiveness.

## Research goal

Learn a voxel-wise gate over four frozen segmentation experts so the fused tumor segmentation can improve on the best individual expert. The experts stay independent. An LLM reasoning expert can explain the debate and cannot edit the mask.

## Four segmentation experts

| Expert | Role | Frozen checkpoint |
|---|---|---|
| CNN | Local 3D U-Net | `outputs/brats2023_gli_cnn_curve4/checkpoints/cnn/best.pt` |
| Transformer | Hierarchical window attention | `outputs/brats2023_gli_transformer_curve3/checkpoints/transformer/best.pt` |
| Boundary | Sum Medical boundary geometry | `outputs/brats2023_gli_boundary_curve2/checkpoints/boundary/best.pt` |
| HighRes | Dilated native-resolution detector | `outputs/brats2023_gli_highres_curve5/checkpoints/highres/best.pt` |

Validation mean Dice on 191 cases, from each expert's own selected epoch: CNN 0.86218 (epoch 15), Transformer 0.85762 (epoch 6), Boundary 0.86270 (epoch 16), HighRes 0.81395 (epoch 20). Boundary was the best individual expert on validation.

## Learned controller

The controller is a pointwise network, 65 inputs to a width-8 hidden layer with GELU, then 4 expert logits and a masked softmax. It was trained with AdamW at learning rate 0.0003 and weight decay 0.01, batch size 1, four cached 64³ patches per controller-training case, for 20 epochs. Mixed precision is off. The device for this run was MPS in float32.

Training used only the 248-case `controller_train` split. Full-volume validation was deferred until all 20 epoch checkpoints existed. The selection rule is the highest mean of per-case BraTS mean Dice, with the earliest epoch winning a tie. Epoch 19 won. Its validation scores on 191 cases are WT 0.90161, TC 0.87386, ET 0.83039, mean 0.86862. `best.pt` is byte-identical to `epoch_019.pt`.

## 65-feature architecture

Each voxel's controller input is the 4 MRI channels plus, for each of the 4 experts, class probabilities, uncertainty, confidence, margin, 4 compact features, and availability. Shared debate channels add the disagreement map, Jensen-Shannon divergence, probability variance, prediction entropy, and the expert vote. The boundary expert also contributes its boundary probability and a scaled distance. That total is 65 channels. The LLM is not one of them.

## Data splits and leakage prevention

The canonical split file is `configs/splits/brats2023_gli_cases.json`.

| Split | Cases |
|---|---:|
| Expert training | 682 |
| Controller training | 248 |
| Validation | 191 |
| Test | 130 |

The four lists do not overlap. Experts never train on controller, validation, or test cases. The controller never trains on expert-training, validation, or test cases. Checkpoint protocol hash `721b8dcb3bb3a6f84ad82c7ee78a443bef8754743aebc88368e0123159477d98` and split hash `705cf73c10ab6d047b095c461fe8c71329b3e0a3ee01f925ced205747102cfe2` are stored with the weights. The test comparison refuses to run unless the validation selection record names the same expert and controller files.

## Final test

All 130 test cases were scored with the frozen experts and epoch-19 controller. Fusion uses the HighRes proposal cap of 128 patches. The HighRes row below is a separate full-volume pass, so that expert is not penalized for voxels the fusion cap left uncovered. HD95 uses voxel spacing when the image provides it. Cases where a region is empty on only one of the prediction or the label leave HD95 undefined, and those cases are omitted from that region's mean.

| Model | WT Dice | TC Dice | ET Dice | Mean Dice | WT HD95 (mm) | TC HD95 (mm) | ET HD95 (mm) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN | 0.88975 | 0.87074 | 0.84484 | 0.86844 | 18.950 | 13.520 | 11.923 |
| Transformer | 0.88666 | 0.86631 | 0.84510 | 0.86602 | 17.239 | 13.148 | 10.177 |
| Boundary | 0.89014 | 0.86263 | 0.83757 | 0.86344 | 18.393 | 17.223 | 13.833 |
| HighRes | 0.80420 | 0.83077 | 0.80536 | 0.81344 | 47.914 | 34.394 | 26.407 |
| Controller, epoch 19 | 0.89479 | 0.87679 | 0.84872 | 0.87344 | 17.279 | 14.421 | 12.527 |

On the test set the CNN is the best single expert. The controller's mean Dice is 0.00499 higher, and WT, TC, and ET Dice are all higher than the CNN. Whole-tumor HD95 is lower than the CNN. Tumor-core and enhancing-tumor HD95 are higher than the CNN and the Transformer, so the surface-distance result is not an improvement on every region. Majority vote reaches 0.87312 mean Dice, 0.00032 below the controller. HighRes remains well behind the other three experts.

Validation mean Dice was 0.86862 and test mean Dice is 0.87344. Whole-tumor Dice is lower on the test set than on validation (0.90161 to 0.89479). Enhancing-tumor Dice is higher (0.83039 to 0.84872).

Machine-readable copies: [test/report.json](test/report.json), [test/report.md](test/report.md), and [test/aggregate_metrics.csv](test/aggregate_metrics.csv).

## Expert disagreement

Across the 130 test cases, average disagreement and segmentation error rate move together. Spearman correlations are about 0.66 to 0.70 for the CNN, Transformer, Boundary, vote, average, and controller, and about 0.90 for HighRes. That is a cohort-level description. Neighboring voxels are not independent samples, and disagreement is not a calibrated clinical uncertainty.

## Inference procedure

The frozen settings are overlap 0.5, one Monte Carlo sample, uncertainty threshold 0.55, and a HighRes cap of 128 patches. Experts run one at a time. The first three experts tile the native volume. HighRes scores proposal patches from their tumor votes and uncertainty, up to the cap. The controller then mixes the four probability maps with the 65-channel features. Labels in exported segmentations follow the configured source mapping, which for this study is internal classes 0, 1, 2, and 3.

```bash
python scripts/evaluate.py --config configs/brats_controller.yaml --split test
python scripts/run_inference.py --config configs/brats_controller.yaml --patient /path/to/one/patient
```

`evaluate.py` resumes from `evaluation/test/cases/<patient_id>.json`. A case is skipped only when that file is complete and its checkpoint, protocol, and inference settings match. The test does not train and does not select a new checkpoint.

A smoke run of `run_inference.py` on a synthetic 64×64×48 volume wrote a `final_seg.nii.gz` of that same shape with labels in `{0, 1, 2, 3}`. The five checkpoint hashes were unchanged. The synthetic volume is not a BraTS case and its labels are not a test score. See [smoke_test.json](smoke_test.json).

## Checkpoint hashes

| Checkpoint | SHA256 |
|---|---|
| CNN | `e8fbde5084353dc5aa499626d208fad7c7c4595e23376daa7d98a4890259697c` |
| Transformer | `dd77255376fb0831e7b91795dff3dbf1bba647b3e305df9ef5dfa21d6d55b64b` |
| Boundary | `dd360d8d6430acd4eff116b181ed3b83b2144fffd16c0bf8c0baaa913f00b79c` |
| HighRes | `15bd9b270ff3ca160e7a4f07a93809e1e5ad9c7e25b352966df5940d639b83c0` |
| Controller epoch 19 / `best.pt` | `86b604082a934a1d4f4a68467a88d16d268ae7653c552dcfc6c3087ba1d79fe2` |

The config used for the test is `configs/brats_controller.yaml`. Its protocol hash is `721b8dcb3bb3a6f84ad82c7ee78a443bef8754743aebc88368e0123159477d98`. Full paths and package versions are in [manifest.json](manifest.json). Checkpoint binaries are not stored in git.

## Known limitations

The gain over the best test expert is small, and majority vote is nearly as strong as the learned gate. HD95 does not improve for every tumor region. HighRes is much weaker than the other experts on this split. Controller validation did not compute HD95, so validation and test surface distances cannot be compared. Exact numeric replay can differ across devices and library builds. The deterministic reasoning mode used for this test is a labeled fallback, not a live language model.

## Later research, not part of Beta 1

These ideas are reserved for a later comparison against this baseline. They are not implemented here.

- Formal error prediction and calibrated uncertainty
- Expert-weight correctness and conditional information analysis
- Probabilistic tumor volume
- Anatomical, volume, and topology objectives
- Active perception and action selection
