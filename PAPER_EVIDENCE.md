# Paper evidence: controller architecture and saved comparisons

This file records what the code and saved reports support. A later low-memory pilot did run the spatial comparison; its result is in section 15. It did not improve boundary accuracy.

Cohorts are kept separate. A number from the historical test is not a number from a later fit-subject experiment.

## 1. Original controller

The frozen Beta 1 controller is `GatingNetwork` in `src/brats_debate/controller/gating_network.py`, configured by `configs/brats_controller.yaml`.

It is two 3D convolutions, both with kernel size 1, and a GELU between them. The hidden width is 8. With the production input width of 65 and 4 experts, the parameter count is 564.

```
features [65, D, H, W]
        |
        v
 Conv3d, kernel 1x1x1, 65 -> 8
        |
       GELU
        |
 Conv3d, kernel 1x1x1, 8 -> 4 expert logits
        |
 availability <= 0  ->  logit -10000
        |
 softmax over the expert axis
        |
 fused probability = sum_k  weight_k * probability_k
```

A kernel of size 1 reads only the feature vector at the same voxel. The controller does not mix features from a neighboring voxel. Neighboring anatomy can still appear inside that voxel’s own features, because those features were computed by the experts and by the MRI. The gate itself never looks one voxel to the left or right.

### Input channels

`feature_channels` in `src/brats_debate/debate/features.py` is

```
modalities + experts * (classes + compact features + 4) + 5 + 2 * classes
```

Production values are 4 MRI channels, 4 experts, 4 classes, and 4 compact expert-feature channels:

```
4 + 4 * (4 + 4 + 4) + 5 + 8 = 65
```

For each expert the 12 channels are: 4 class probabilities, uncertainty, confidence, margin, availability, and 4 compact features. The shared maps are the disagreement map, Jensen–Shannon disagreement, class-wise probability variance (4), normalized predictive entropy, class votes divided by 4, boundary probability, and distance-to-boundary divided by 16.

Label identity is `{0:0, 1:1, 2:2, 3:3}`. Regions are WT = {1,2,3}, TC = {1,3}, ET = {3}.

### Fusion and masking

Let `z_k(v)` be the logit for expert `k` at voxel `v`, and let `a_k(v)` be 1 when that expert’s prediction is available. Beta 1 forms

```
w_k(v) = exp(z_k(v)) / sum_j exp(z_j(v))     if a_k(v) > 0
```

with unavailable logits replaced by `-10000` before the softmax. The fused class distribution is

```
p(v) = sum_k w_k(v) p_k(v)
```

Weights are normalized because they are a softmax. They are not forced to exact zero for an unavailable expert; a logit of `-10000` makes that weight numerically negligible. The forward pass rejects a voxel where every expert is unavailable.

HighRes inside this controller is proposal-limited: at most 128 windows of `32³` at overlap 0.5 (`highres_max_patches: 128`). Uncovered voxels are stored as the uniform distribution `1/4` with availability 0. That placeholder is not a prediction and is not counted as an error.

### What already carries spatial context

The gate input at a voxel can include MRI intensities, the boundary expert’s distance-to-boundary, and whatever spatial context each expert folded into its own probability and compact features. Those are properties of one voxel’s feature vector. They are not a neighborhood gathered by the controller.

### Training objective

The controller is trained with `segmentation_loss` in `src/brats_debate/training/train_expert.py`:

```
L = NLL(p, y) + 1 - mean_{c in {1,2,3}} softDice(p_c, y_c)
```

Background is inside the negative log-likelihood and outside the soft-Dice average. Checkpoint selection uses BraTS mean Dice (unweighted WT, TC, and ET) with `hd95: false`. There is no surface-distance or HD95 term in the controller objective. The boundary expert has its own boundary and distance losses; the controller does not.

This is an overlap-plus-classification objective. It does not, by itself, prove that the historical HD95 gap was caused by the loss.

## 2. Beta 2 residual gate is a different model

`ResidualAdaptiveGate` in `src/brats_debate/controller/adaptive_gate.py` is not a pair of pointwise convolutions. It is an MLP, 36 → 32 → 4, with GELU and a zero-initialized last layer. Its logits are

```
log(clamp(frozen Beta 1 weight, 1e-8)) + residual
```

Unavailable experts are then set to exact weight 0 and the remaining weights are renormalized. The Stage 2 revision multiplies the residual by a hard-disagreement mask, so agreement voxels stay at the frozen gate. The 36 inputs are 4×4 expert probabilities, 4 entropies, 4 availability flags, 4 frozen weights, Jensen–Shannon disagreement, predictive entropy, boundary probability, boundary distance, and 4 MRI channels. Labels are not inputs.

The saved checkpoint `outputs/beta2/stage2_adaptive_20261020/revised_gate.pt` was not modified or retrained for this session.

## 3. Spatial alternative that was implemented and not trained

`src/brats_debate/controller/spatial_gate.py` adds two new modules. Neither replaces `GatingNetwork`.

| Model | Layers | Parameters at 65 inputs and 4 experts |
| --- | --- | --- |
| Frozen Beta 1 | two kernel-1 convolutions, width 8 | 564 |
| Matched pointwise control | two kernel-1 convolutions, width 11 | 774 |
| Spatial-context gate | kernel-1 projection to width 8, depthwise 3×3×3, kernel-1 logits | 788 |

The spatial gate’s 3×3×3 layer is depthwise (`groups=8`) with padding 1, so each hidden channel mixes only its own 26 neighbors plus the center. Both new gates use the same masked softmax as Beta 1. The width-11 pointwise net is the capacity-matched control: 774 versus 788 parameters, about 1.8% apart. Frozen Beta 1 remains a separate reference with fewer parameters and its original training run.

A spatial smoothing penalty was not added. The intended comparison changes spatial support only.

Tests in `tests/beta2/test_spatial_gate.py` check three facts: a neighboring voxel changes the spatial gate’s center weight and does not change the pointwise gate; an unavailable expert is masked and the fused probabilities sum to 1; the two new parameter counts stay within 5%. Those tests passed. They do not measure Dice or HD95.

### Why it was not trained

Saved rich rows are independent voxels. `X` has shape `(N, 8)` plus a flat index. They do not contain a `3×3×3` neighborhood, and neighbors were not invented around those rows.

A spatial gate needs new contiguous volumes from frozen experts. At the start of this session, `shutil.disk_usage` reported 24.09 GiB free. The pre-registered rule requires at least 25 GiB before that inference. A later check in the same session reported 24.07 GiB free, about 3.6 GiB of available memory, and 14.14 GiB of 15 GiB swap in use. Inference was not started.

The unused split was written to `outputs/beta2/spatial_controller_pilot/protocol.json` before any new outcome existed. There is no new outcome.

Training scans, previously observed development subjects, first scan only:

`BraTS-GLI-00006-000`, `00025-000`, `00049-000`, `00056-000`, `00101-000`, `00104-000`, `00112-000`, `00133-000`.

`BraTS-GLI-00056-001` stays with subject `00056` and is not an evaluation scan.

Evaluation scans, the next four fit subjects in immutable manifest order after every subject already used in Stage 2, Stage 3, or Stage 4:

`BraTS-GLI-01016-000`, `01020-000`, `01022-000`, `01064-000`.

These four were not chosen by Dice or HD95. They have not been scored by a spatial controller. They must not be retuned after a future result and then described as fresh.

## 4. Historical test cohort

Source: `docs/beta1/test/aggregate_metrics.csv` and `docs/beta1/test/report.json`. One hundred thirty subjects. Cohort means only. There is no saved per-subject file. Test MRI was not opened.

HighRes in this table is the published full-volume HighRes run. It is not the proposal-limited HighRes that the controller fuses. Mean Dice below is the unweighted mean of WT, TC, and ET Dice (`brats_mean_dice`).

| Model | WT Dice | TC Dice | ET Dice | Mean Dice | WT HD95 | TC HD95 | ET HD95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.8898 | 0.8707 | 0.8448 | 0.8684 | 18.950 | 13.520 | 11.923 |
| Transformer | 0.8867 | 0.8663 | 0.8451 | 0.8660 | 17.239 | 13.148 | 10.177 |
| Boundary | 0.8901 | 0.8626 | 0.8376 | 0.8634 | 18.393 | 17.223 | 13.833 |
| HighRes, full volume | 0.8042 | 0.8308 | 0.8054 | 0.8134 | 47.914 | 34.394 | 26.407 |
| Majority vote | 0.8940 | 0.8756 | 0.8498 | 0.8731 | 17.221 | 12.452 | 10.919 |
| CancerAgents | 0.8948 | 0.8768 | 0.8487 | 0.8734 | 17.279 | 14.421 | 12.527 |

CancerAgents minus Transformer, on these cohort means:

| Region | Dice difference | HD95 difference (mm) |
| --- | ---: | ---: |
| WT | +0.00813 | +0.039 |
| TC | +0.01048 | +1.272 |
| ET | +0.00363 | +2.350 |

Positive HD95 means CancerAgents is worse on that cohort mean. A subject-level paired bootstrap cannot be computed. The case-level historical results are not in the saved report, and the test images stay locked.

### Discrete Dice–HD95 points

Figure: `docs/beta2/figures/historical_test_dice_hd95.png`.

Each point is one frozen model on this 130-subject cohort. The plot is not a fitted curve and not a Pareto frontier. No operating point was tuned on this cohort for this figure.

Among these six points, a model is non-dominated on a region when no other listed model has both higher Dice and lower HD95:

- WT: majority vote and CancerAgents.
- TC: majority vote and CancerAgents.
- ET: Transformer and majority vote.

On ET, majority vote has both higher Dice (0.8498 versus 0.8487) and lower HD95 (10.919 mm versus 12.527 mm) than CancerAgents, so CancerAgents is dominated on ET by the vote inside this set. Transformer remains non-dominated on ET because its HD95, 10.177 mm, is the lowest of the six.

## 5. Later fit cohorts, not mixed with the test table

These subjects come from the fit pool. Beta 1 checkpoint selection used the validation pool, so none of these tables is an external test.

### Stage 2 revised gate, five subjects

Source: `outputs/beta2/stage2_adaptive_20261020/revised_evaluation_report.json`.

Cases: `BraTS-GLI-00304-000`, `00313-000`, `00338-000`, `00373-000`, `00382-000`.

Transformer was not scored as its own method in this run. HighRes is proposal-limited.

| Model | WT Dice | TC Dice | ET Dice | Mean Dice | WT HD95 | TC HD95 | ET HD95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.9310 | 0.9181 | 0.8585 | 0.9025 | 4.342 | 5.930 | 6.646 |
| Majority vote | 0.9328 | 0.9209 | 0.8631 | 0.9056 | 4.396 | 10.374 | 10.994 |
| Frozen Beta 1 | 0.9362 | 0.9228 | 0.8588 | 0.9059 | 4.357 | 7.002 | 7.867 |
| Stage 2 residual gate | 0.9356 | 0.9262 | 0.8659 | 0.9092 | 4.268 | 6.640 | 7.549 |

Mean Dice for the residual gate is 0.0033 above frozen Beta 1 on these five subjects. That difference was not confirmed on the next cohort.

### Stage 4, twenty later subjects

Source: `outputs/beta2/stage4_frozen_eval_20261020/report.json`. One first-fit scan per subject. Transformer was again not a separate scored method.

| Model | WT Dice | TC Dice | ET Dice | Mean Dice | WT HD95 | TC HD95 | ET HD95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.9321 | 0.9395 | 0.9149 | 0.9289 | 13.626 | 2.704 | 1.564 |
| Majority vote | 0.9354 | 0.9507 | 0.9232 | 0.9364 | 9.305 | 2.926 | 1.473 |
| Frozen Beta 1 | 0.9344 | 0.9546 | 0.9261 | 0.9383 | 10.078 | 2.065 | 1.449 |
| Stage 2 residual gate | 0.9357 | 0.9536 | 0.9254 | 0.9382 | 10.034 | 2.426 | 1.430 |

Stage 2 minus Beta 1, mean Dice, subject as the unit, 10,000 bootstrap resamples, seed 20261020:

- Mean paired difference: −0.00014.
- Median: −0.00049.
- Better on 8 subjects, worse on 12.
- 95% interval: [−0.00086, +0.00077]. The interval includes zero.

Stage 2 minus majority vote, same bootstrap:

- Mean +0.00179, better on 16 of 20.
- 95% interval: [+0.00086, +0.00282].

HD95, Stage 2 versus Beta 1, counted by subject (lower is better):

| Region | Better | Worse | Unchanged |
| --- | ---: | ---: | ---: |
| WT | 3 | 5 | 12 |
| TC | 3 | 3 | 14 |
| ET | 2 | 0 | 18 |

Voxel labels versus Beta 1 on these twenty volumes: 16,177 errors repaired, 14,799 new errors introduced. All of those changed voxels lie in the hard-disagreement mask, because the Stage 2 residual is multiplied by that mask.

The supported Stage 4 conclusion is that the revised gate does not replace Beta 1. It is slightly ahead of majority vote on mean Dice in this fit cohort. It is not evidence about a spatial controller.

## 6. Hard disagreement and Jensen–Shannon disagreement

Both definitions use available experts only. An unavailable HighRes voxel is uniform `1/4` with availability 0 and is excluded.

Hard disagreement at voxel `v` is 1 when the minimum and maximum hard labels among available experts differ. With one available expert the two are equal, so the flag is 0. Class 0, background, counts as a label. The implementation is `hard_disagreement` in `src/brats_debate/uncertainty/stage2_gate.py`.

Jensen–Shannon disagreement, from `analyze_disagreement` in `src/brats_debate/debate/disagreement.py`:

```
m(v) = sum_{k available} p_k(v) / K(v)

D_JS(v) = [ H(m(v)) - mean_{k available} H(p_k(v)) ] / ln(C)
```

`H` is Shannon entropy in nats (`p * ln p`, with `p` clamped at `1e-8`). `C = 4`. Division by `ln(4)` puts `D_JS` on `[0, 1]`. Predictive entropy stored for the controller and for the rich samples is the same kind of normalized entropy: `H(m) / ln(4)`, also checked to lie in `[0, 1]`. The threshold 0.5 used in earlier Beta 2 audits is on this normalized entropy, not on raw nats.

`K(v)` is the number of experts with availability greater than 0. It is not always 4.

## 7. HighRes availability on saved development samples

Source: 21 rich case files, 20 subjects, under `outputs/beta2/oracle_gap_20261019/cases` and `outputs/beta2/stage2_adaptive_20261020/rich/cases`. Rates use the stored inverse-inclusion weights. This is the proposal-limited HighRes seen by the controller, not the full-volume HighRes in the historical table.

| Quantity | Result |
| --- | --- |
| Volume coverage, subject-mean (median) | 0.459 (0.450); range 0.312–0.603 |
| Sampled WT voxels with HighRes available | 691,947 of 691,947 |
| Sampled near-boundary voxels with HighRes available | 579,922 of 580,207 |
| Conditional hard accuracy where HighRes is available, subject-mean | 0.972 |

Near-boundary means the stored distance feature is at most `1/16`. That feature is raw distance divided by 16, and the raw target is 0 on boundary voxels. The 285 missing near-boundary rows are 0.05% of the sampled near-boundary rows.

Coverage inside the whole tumor is complete in the saved sample. It is not a census of every WT voxel in every volume: unsampled voxels are not in these files. The inverse-inclusion estimate of WT coverage is also 1.0 for every one of the 20 subjects, because no sampled WT row was uncovered.

Among hard-disagreement rows, the inverse-inclusion share on which HighRes is already available has median 0.996 (minimum 0.960, maximum 1). The share that is both HighRes-unavailable and either hard-disagreement or normalized entropy at least 0.5 has median `3.3e-5` and maximum 0.0013. That is why an uncertainty trigger for extra HighRes windows had almost no eligible mass.

When HighRes is unavailable, fusion ignores it through the availability mask. The uniform placeholder is not scored as a wrong prediction.

## 8. Disagreement and context on the same development sample

Same 20 subjects and same inverse-inclusion weights. The quantity is the estimated controller-error rate inside each subset. Tumor size in voxels is not stored as a volume count in these files, so tumor size was not analyzed. Background versus tumor uses the saved label. This is association under one sampling design. It is not an adjusted claim that disagreement predicts error after the other factors are held fixed.

Subject-mean error rate:

| Subset | Subject-mean error rate | Subjects with the higher rate |
| --- | ---: | --- |
| Hard disagreement | 0.290 | 20 of 20, versus agreement |
| Expert agreement | 0.004 |  |
| Normalized entropy ≥ 0.5 | 0.408 | 20 of 20, versus entropy < 0.5 |
| Normalized entropy < 0.5 | 0.005 |  |
| Distance feature ≤ 1/16 | 0.107 | 20 of 20, versus farther voxels |
| Distance feature > 1/16 | 0.002 |  |
| Whole-tumor label | 0.093 | 20 of 20, versus background |
| Background label | 0.003 |  |

Background accounts for a subject-mean share of about 0.943 of the weighted volume. Hard disagreement accounts for about 0.012. An unstratified correlation is therefore mostly a background effect. The stratified rates still move together: disagreement, high entropy, the boundary neighborhood, and tumor voxels all have higher controller-error rates in every one of these 20 subjects.

## 9. Calibration

No calibration map was fit. The diagnostic is expected calibration error of the maximum class probability against hard correctness, 10 equal-width bins, inverse-inclusion weights, available experts only, development subjects only.

| Expert | Whole volume, mean ECE | WT, mean ECE | TC, mean ECE | ET, mean ECE | Background, mean ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN | 0.0023 | 0.047 | 0.086 | 0.065 | 0.0018 |
| Transformer | 0.0025 | 0.055 | 0.062 | 0.044 | 0.0016 |
| Boundary | 0.0025 | 0.046 | 0.067 | 0.044 | 0.0016 |
| HighRes, where available | 0.0096 | 0.040 | 0.070 | 0.052 | 0.0072 |

The volume number is small because most weighted voxels are background. Inside tumor regions the subject-mean ECE is several percent. Medians are lower than means (WT CNN median 0.019; TC CNN median 0.041), so a few subjects pull the mean up. An unweighted ECE on the raw stratified rows is about 0.06 and is not a volume estimate, because those rows oversample errors.

This does not say the probabilities are calibrated for clinical use. It says that, on this development sample, argmax confidence tracks hard accuracy closely in background and only roughly inside ET, TC, and WT.

## 10. What this session measured about spatial context

Nothing about Dice or HD95. The spatial module runs, uses a real `3×3×3` neighborhood, and is close in parameter count to the matched pointwise control. Whether that neighborhood improves HD95, and whether it costs Dice, is unmeasured.

No contour figure was made. A contour would require volumes from the held-out fit cases, and those volumes were not inferred.

## 11. Supported statements

- The Beta 1 controller is two pointwise convolutions. It does not combine neighboring controller inputs.
- Its training loss has soft Dice and negative log-likelihood, and no HD95 term. Checkpoint selection also ignored HD95.
- On the 130-subject historical test, CancerAgents has higher mean Dice than the Transformer and worse cohort-mean HD95 in WT, TC, and ET. Paired subject intervals for that gap are not available from saved files.
- On the 20-subject Stage 4 fit cohort, the Beta 2 residual gate does not improve mean Dice over frozen Beta 1. The paired interval includes zero.
- In the saved development samples, proposal-limited HighRes is missing from about half of the volume and from none of the 691,947 sampled whole-tumor voxels.
- Controller errors in those samples concentrate in hard disagreement, high normalized entropy, the boundary neighborhood, and tumor labels. Those factors are not separated.

## 12. Missing evidence

- Any Dice or HD95 for the matched pointwise control or the spatial gate.
- A subject-level bootstrap of CancerAgents versus Transformer on the historical test.
- Full-volume HighRes coverage as a census, as opposed to the sampled inverse-inclusion estimate.
- Calibration fit on training subjects and evaluated on untouched subjects. Nothing was fit.
- A joint model of disagreement, tumor size, boundary distance, and entropy. Tumor size was not in the saved rows.
- Contour plates and 3D component or surface diagnostics for the spatial controller.
- A spatial regularizer. It was deliberately not included in the unimplemented main comparison.

## 13. Figure caption

**Historical test, Dice against HD95.** Each marker is one frozen model (CNN, Transformer, Boundary, full-volume HighRes, majority vote, CancerAgents) on the same 130-subject test cohort. Columns are WT, TC, and ET. Dice is higher upward. HD95 is in millimeters and is better toward the left. Points are the published cohort means. They are not a continuous frontier, and no threshold was chosen on this cohort to place them.

## 14. Reproduction of what was actually run

From `/Users/andersonliew/CancerAgents` with `/opt/anaconda3/bin/python`:

```
python -m pytest tests/beta2/test_spatial_gate.py -q
```

The three tests passed. The later low-memory run is separate and is summarized in section 15. Frozen checkpoints, the partition manifest, and the Stage 2 checkpoint were not modified.

## 15. Low-memory spatial pilot

Run directory: `outputs/beta2/spatial_controller_pilot/lowmem_20261025/`. Report: `report.json`. Status: complete. The original `scripts/beta2_spatial_pilot.py` swap gate was not changed and was not used for this path.

Execution was one frozen expert at a time on CPU. A warmed 64³ patch on MPS reserved about 1.4 GiB of driver memory and was not faster than CPU. On the saved BraTS-GLI-00006-000 sample, CPU expert probabilities and the frozen Beta 1 controller matched the earlier MPS reference with maximum absolute difference about 6e-7 and argmax agreement 1.0.

Training was the predeclared budget: 40 epochs, seed 20261024, Adam learning rate 1e-3, the Beta 1 segmentation loss, one window from each of eight development subjects at each step. No spatial regularizer. Final training loss was 0.8252 for the width-11 pointwise gate and 0.8159 for the 3×3×3 spatial gate.

Evaluation is four predeclared fit subjects. Mean Dice and mean HD95 in millimeters:

| Model | Mean Dice | WT HD95 | TC HD95 | ET HD95 |
| --- | ---: | ---: | ---: | ---: |
| Transformer | 0.8380 | 5.24 | 9.80 | 7.03 |
| Frozen Beta 1 | 0.8538 | 3.32 | 6.88 | 6.80 |
| Pointwise control | 0.8459 | 17.55 | 7.88 | 7.70 |
| Spatial controller | 0.8427 | 18.09 | 11.78 | 11.49 |

Spatial minus pointwise mean Dice was negative on three subjects and +0.0031 on BraTS-GLI-01064-000. Spatial HD95 was not lower in any region on the four-subject mean. Where spatial Dice rose and HD95 worsened, the saved component flags attribute more than 1 mm of that HD95 to false-positive islands. Contours are in `figures/`. This cohort is exploratory and is not an external test. The new gates also had a much smaller training budget than frozen Beta 1, so the gap versus Beta 1 is not a pure architecture result. The spatial-versus-pointwise comparison is the architecture result, and it does not support a boundary improvement.
