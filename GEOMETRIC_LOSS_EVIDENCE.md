# Geometric loss evidence

Status: training and the four predeclared evaluations are complete. Fit-only comparison on four development scans.

## Equation

For one region, with predicted probability \(p(v)\) and reference indicator \(g(v)\),

\[
L_{\mathrm{dist}}=\frac{\sum_v p(v)\,d(v,G)^2+\sum_v g(v)\,(1-p(v))\,d(v,\partial G)^2}{\sum_v(p(v)+g(v))+\varepsilon}
\]

\(d(v,G)\) is the Euclidean distance in millimeters to the reference region and is 0 inside it. \(d(v,\partial G)\) is the distance in millimeters to the reference surface. Both are capped at 32 mm. \(\varepsilon=10^{-6}\).

\(L_{\mathrm{total}}=L_{\mathrm{baseline}}+\lambda L_{\mathrm{dist}}\). \(L_{\mathrm{baseline}}\) is the existing controller loss: voxel negative log-likelihood plus one minus the mean soft Dice of classes 1, 2, and 3. The geometric term is the mean of the WT, TC, and ET terms. Region probabilities are sums of class probabilities: WT = 1+2+3, TC = 1+3, ET = 3.

Distances are computed on the full labeled volume and then cropped to the 16³ training interior. They are not recomputed inside an 18³ window. They are not network inputs at inference.

## Why this is not the professor's absolute-surface formula

The displayed \(P|\phi_G|+G|\phi_P|\) expression is positive inside a perfect tumor, because absolute distance to the surface is positive in the interior. The surrogate above is zero for a perfect binary prediction of a non-empty region: \(d(v,G)=0\) wherever \(p=1\), and \(1-p=0\) wherever \(g=1\).

## Empty masks and the surface

If the reference region is empty, both distances are the 32 mm cap. A zero prediction then has zero loss. Any positive prediction is penalized.

If the reference is a single voxel, that voxel is the surface, so \(d(v,\partial G)=0\) there. A complete miss of that one voxel is not charged by \(L_{\mathrm{dist}}\). A miss of the interior of a thicker region is charged. The baseline Dice and cross-entropy terms still penalize a missed surface voxel. This is a property of the written surrogate, not a second formula.

## Tests

`tests/beta2/test_geometric_loss.py`: 9 passed. They cover a perfect mask, a shifted boundary, a distant false-positive island versus a near one, a missing extension, an interior false negative, an empty prediction of a thick tumor, an empty reference, a one-voxel reference, non-unit spacing, the 32 mm cap, finite loss and gradients without an argmax, WT/TC/ET probability nesting, normalized masked fusion, and the requirement that a patch-edge distance transform is not used in place of the full-volume map.

## Lambda

Chosen before evaluation, on the training windows only, at the shared initialization:

\(\lambda=0.25\times 0.852584 / 357.0391=0.000596982\).

The 0.25 factor makes the geometric term one quarter of the baseline loss value at initialization. It does not match gradient size. During training the baseline gradient norm was about 0.04 to 0.07. The geometric gradient norm was about 0.001 to 0.006.

## Training

Eight subjects, nine scans: `BraTS-GLI-00006-000`, `00025-000`, `00049-000`, `00056-000`, `00056-001`, `00101-000`, `00104-000`, `00112-000`, `00133-000`.

Both models are `PointwiseMatchedGate` width 11, 774 parameters, seed 20261024, Adam learning rate 0.001, weight decay 0.0001, gradient clip 1, 40 epochs. They started from the same `init.pt`. Expert patches are the frozen spatial-pilot windows. Spacing on these scans was 1 mm.

| Model | Epoch 1 total | Epoch 40 total | Epoch 40 baseline term | Epoch 40 geometric term | Seconds | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.85115 | 0.82522 | 0.82522 | 0 | 30.0 | 1298 MiB |
| Geometric | 1.06436 | 1.03873 | 0.82543 | 357.30 | 42.3 | 832 MiB |

The baseline model reproduced the earlier pointwise run's epoch-40 loss, 0.8252169224951003. The geometric model's baseline term ended at 0.82543. The geometric term moved from 357.09 to 357.30. It did not fall.

## Evaluation

All four predeclared scans are scored from saved expert volumes. The controllers were not retrained. Lambda stayed 0.000596982. Mean Dice is the unweighted mean of WT, TC, and ET, and it matches the stored `mean_dice` field. Four-scan means are unweighted. HD95 is in millimeters. These scans were held out of this training run and belong to the development partition.

### Four-scan means

| Model | Mean Dice | WT Dice | TC Dice | ET Dice | WT HD95 | TC HD95 | ET HD95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Transformer | 0.8227 | 0.9024 | 0.7992 | 0.7666 | 23.690 | 25.575 | 25.370 |
| Frozen Beta 1 | 0.8166 | 0.9115 | 0.7843 | 0.7539 | 23.407 | 25.317 | 16.142 |
| Baseline loss | 0.7855 | 0.8429 | 0.7714 | 0.7421 | 42.541 | 25.936 | 25.485 |
| Geometric loss | 0.7928 | 0.8510 | 0.7783 | 0.7492 | 42.162 | 25.872 | 25.382 |

Geometric minus baseline on these means: mean Dice +0.0073, WT/TC/ET Dice +0.0081 / +0.0069 / +0.0070, WT/TC/ET HD95 −0.379 / −0.064 / −0.103 mm. Geometric minus frozen Beta 1: mean Dice −0.0238. Geometric minus Transformer: mean Dice −0.0299. The WT HD95 mean is pulled up by 01077 and 01081. The ET HD95 gap versus frozen Beta 1 comes entirely from 01077.

### Per scan

| Case | Method | Mean Dice | WT Dice | WT HD95 | TC Dice | TC HD95 | ET Dice | ET HD95 | Voxel errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01065-000 | Transformer | 0.9433 | 0.9639 | 1.414 | 0.9614 | 1.414 | 0.9047 | 1.414 | 16256 |
| 01065-000 | Frozen Beta 1 | 0.9466 | 0.9660 | 1.414 | 0.9644 | 1.414 | 0.9093 | 1.414 | 15340 |
| 01065-000 | Baseline | 0.9476 | 0.9658 | 1.414 | 0.9650 | 1.414 | 0.9122 | 1.414 | 15248 |
| 01065-000 | Geometric | 0.9486 | 0.9662 | 1.414 | 0.9660 | 1.414 | 0.9136 | 1.414 | 15123 |
| 01071-000 | Transformer | 0.8863 | 0.9036 | 5.196 | 0.9064 | 2.236 | 0.8488 | 1.414 | 23570 |
| 01071-000 | Frozen Beta 1 | 0.8916 | 0.9117 | 4.899 | 0.9099 | 2.828 | 0.8531 | 1.414 | 21857 |
| 01071-000 | Baseline | 0.8848 | 0.9059 | 6.403 | 0.9006 | 3.162 | 0.8479 | 1.732 | 23536 |
| 01071-000 | Geometric | 0.8872 | 0.9062 | 6.164 | 0.9038 | 3.000 | 0.8516 | 1.414 | 23370 |
| 01077-000 | Transformer | 0.4915 | 0.7725 | 87.151 | 0.3510 | 97.650 | 0.3510 | 97.650 | 3189 |
| 01077-000 | Frozen Beta 1 | 0.4530 | 0.7940 | 86.313 | 0.2806 | 96.023 | 0.2845 | 60.741 | 3351 |
| 01077-000 | Baseline | 0.3448 | 0.5543 | 95.488 | 0.2387 | 98.167 | 0.2415 | 97.794 | 9076 |
| 01077-000 | Geometric | 0.3689 | 0.5801 | 95.486 | 0.2621 | 98.072 | 0.2645 | 97.701 | 8213 |
| 01081-000 | Transformer | 0.9698 | 0.9696 | 1.000 | 0.9779 | 1.000 | 0.9618 | 1.000 | 5356 |
| 01081-000 | Frozen Beta 1 | 0.9751 | 0.9741 | 1.000 | 0.9824 | 1.000 | 0.9687 | 1.000 | 4457 |
| 01081-000 | Baseline | 0.9646 | 0.9456 | 66.859 | 0.9812 | 1.000 | 0.9671 | 1.000 | 8028 |
| 01081-000 | Geometric | 0.9665 | 0.9514 | 65.582 | 0.9812 | 1.000 | 0.9669 | 1.000 | 7309 |

### Paired geometric minus baseline

| Scan | Mean Dice | WT Dice | WT HD95 | TC Dice | TC HD95 | ET Dice | ET HD95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01065-000 | +0.0010 | +0.0004 | 0.000 | +0.0011 | 0.000 | +0.0014 | 0.000 |
| 01071-000 | +0.0024 | +0.0003 | −0.239 | +0.0032 | −0.162 | +0.0037 | −0.318 |
| 01077-000 | +0.0241 | +0.0257 | −0.002 | +0.0234 | −0.095 | +0.0230 | −0.094 |
| 01081-000 | +0.0019 | +0.0058 | −1.277 | −0.000006 | 0.000 | −0.000120 | 0.000 |

Mean Dice rose on every scan. HD95 fell or stayed the same on every region of every scan. On 01081, TC and ET Dice fell by 0.000006 and 0.000120, and WT Dice rose by 0.0058, so the scan-level mean still rose.

Geometric minus frozen Beta 1, mean Dice: +0.0021, −0.0044, −0.0841, −0.0086. Geometric minus Transformer, mean Dice: +0.0053, +0.0009, −0.1226, −0.0032. The same order is 01065, 01071, 01077, 01081. Geometric mean Dice is ahead of both references on 01065 only. It is also ahead of the Transformer on 01071 by 0.0009 and behind frozen Beta 1 there.

### False-positive islands

Whole-tumor false-positive islands are recorded as count / voxels. HD95 after those islands are removed is in the last four columns.

| Scan | Transformer | Frozen Beta 1 | Baseline | Geometric | Transformer HD95 | Beta 1 HD95 | Baseline HD95 | Geometric HD95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 01065-000 | 3 / 5 | 1 / 1 | 10 / 19 | 8 / 15 | 1.414 | 1.414 | 1.414 | 1.414 |
| 01071-000 | 10 / 297 | 7 / 37 | 34 / 570 | 26 / 503 | 4.583 | 4.690 | 5.196 | 5.196 |
| 01077-000 | 54 / 2160 | 30 / 2168 | 215 / 7955 | 218 / 7122 | 1.000 | 1.000 | 1.414 | 1.414 |
| 01081-000 | 3 / 136 | 3 / 80 | 101 / 3282 | 91 / 2558 | 1.000 | 1.000 | 1.000 | 1.000 |

On 01077, removing disconnected predicted components drops WT HD95 from about 95 mm to 1.414 mm for both new controllers, and to 1.000 mm for the Transformer and frozen Beta 1. On 01081 the same removal drops both new controllers from about 66 mm to 1.000 mm. The distance term reduced island voxels, from 7,955 to 7,122 on 01077 WT and from 3,282 to 2,558 on 01081 WT. It left the islands that set HD95. The largest 01081 WT island is 2,037 voxels for the baseline and 1,592 for the geometric controller, against 62 for frozen Beta 1.

Island counts did not move in one direction. They fell for every region on 01065 and for WT on 01071 and 01081. They rose for TC and ET on 01071, from 11 to 25 islands in each region. They rose for WT on 01077, from 215 to 218, while TC fell from 42 to 35 and ET fell from 41 to 34.

On 01071, WT HD95 after island removal is still 5.196 mm for both new controllers. All four methods miss the same three WT components there, 1,883 voxels. The shared miss accounts for the HD95 that remains after the islands are removed. On 01077, TC and ET HD95 for both new controllers fall from about 98 mm to about 7.6 mm after island removal, so a several-millimeter surface error remains. Missed reference components elsewhere are one voxel on 01065 WT for both new controllers, one voxel on 01077 WT for every method, and two ET voxels on 01081 for every method.

Voxel mistakes fell on every scan: 15,248 to 15,123, 23,536 to 23,370, 9,076 to 8,213, and 8,028 to 7,309. On 01077 both new controllers remain well above the Transformer (3,189) and frozen Beta 1 (3,351).

Figures for all four scans are in `outputs/beta2/geometric_loss_20261103/figures/`.

## Conclusion

Against the matched baseline, geometric training raised mean Dice on all four scans and lowered mean WT, TC, and ET HD95 by a fraction of a millimeter. It did so with a geometric gradient about ten times smaller than the baseline gradient, and with a geometric term that stayed near 357. Against frozen Beta 1 and the Transformer, mean Dice is lower and WT HD95 is worse. The WT HD95 failures on 01077 and 01081 are distant false-positive islands. Four development scans cannot support a significance claim.

## Files

`outputs/beta2/geometric_loss_20261103/protocol.json`
`outputs/beta2/geometric_loss_20261103/training_baseline.json`
`outputs/beta2/geometric_loss_20261103/training_geometric.json`
`outputs/beta2/geometric_loss_20261103/baseline.pt`
`outputs/beta2/geometric_loss_20261103/geometric.pt`
`outputs/beta2/geometric_loss_20261103/init.pt`
`outputs/beta2/geometric_loss_20261103/evaluation/BraTS-GLI-01065-000.json`
`outputs/beta2/geometric_loss_20261103/evaluation/BraTS-GLI-01071-000.json`
`outputs/beta2/geometric_loss_20261103/evaluation/BraTS-GLI-01077-000.json`
`outputs/beta2/geometric_loss_20261103/evaluation/BraTS-GLI-01081-000.json`
