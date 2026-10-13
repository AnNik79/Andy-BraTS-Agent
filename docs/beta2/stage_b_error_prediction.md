# Stage B error-prediction foundation

Status: **Implemented and synthetically tested. One real validation case passed
an engineering extraction pilot; no model was fitted, calibrated, or
experimentally evaluated.** See
[`stage_b_one_case_pilot.md`](stage_b_one_case_pilot.md) and the subsequent
[`stage_b_memory_audit.md`](stage_b_memory_audit.md).

## Scope

Stage B estimates whether the frozen Beta 1 controller's hard class prediction
is wrong:

`E(v)=1[argmax_c p_CA,c(v) != y(v)]`.

Three regularized logistic baselines are implemented:

- disagreement-only: `q(v)=sigmoid(beta_0+beta_d D_JS(v))`;
- entropy-only: `q(v)=sigmoid(beta_0+beta_h H(p_bar(v)))`; and
- combined: `q(v)=sigmoid(beta_0+beta_d D+beta_h H+beta_b B+beta_x X_tilde)`.

The combined feature schema is:

1. normalized Jensen-Shannon disagreement;
2. normalized entropy of the mean expert distribution;
3. Boundary-expert boundary probability;
4. Boundary-expert distance divided by 16; and
5. the four normalized MRI modality intensities in configured order.

The implementation is in
`src/brats_debate/uncertainty/error_prediction.py`. Models are small
scikit-learn logistic regressions. The fitted scaler, coefficients, intercept,
feature schema, subjects, split role, and weighted prevalence serialize to
plain versioned JSON. Platt calibration consumes the frozen model logit on a
separate subject partition.

## Subject-grouped protocol

`uncertainty/partitions.py` assigns whole subjects, including repeated scans,
to fit/calibration/evaluation roles. The default exploratory allocation is
60%/20%/20% of the 170 validation subjects using seed `20261012`. Beta 1 test
cases are explicitly forbidden. The controller was selected using the complete
validation set, so these partitions do not constitute independent validation
of the underlying segmenter.

The fixed assignment is recorded in
`configs/beta2/subject_partitions.json`: 111/41/39 cases and 101/35/34
subjects for fit/calibration/evaluation. Its manifest hash, protocol hash, and
frozen split hash are verified before collection. Cases are ordered
lexicographically within each role; collection selection does not inspect
labels or performance.

The intended sequence is:

1. fit each logistic baseline on fit subjects;
2. fit Platt calibration on calibration subjects only;
3. evaluate once on evaluation subjects; and
4. use a new external cohort for confirmatory claims.

## Sampled-voxel artifact

`uncertainty/extraction.py` produces one atomic compressed NPZ per case. It does
not archive full-volume probabilities or controller features.

Arrays:

- `X`: sampled float32 feature rows;
- `y`: uint8 controller error target;
- `flat_index`: source voxel index;
- `inclusion_probability`: exact stratum inclusion probability;
- `sample_weight`: raw inverse-probability weight `1/pi`; and
- `stratum`: error, correct boundary, correct foreground, or other correct.

JSON metadata inside the NPZ records patient/subject identity, validation role,
spatial shape, class order, feature names, sample/population counts, raw and
weighted prevalence, policy, checkpoint hashes, protocol/split hashes,
inference settings, and code commit.

The default maximum is 80,000 voxels per case:

- 20,000 error voxels;
- 10,000 correct boundary voxels;
- 20,000 other correct foreground voxels; and
- 30,000 other correct voxels.

Because this intentionally changes class/stratum prevalence, fitting and
calibration must use the stored raw inverse-inclusion weights. Weights are not
normalized per case, because that would bias pooled cases of different
population sizes; an estimator may apply one common normalization after
pooling. The artifact records both sampled and full-domain prevalence as a
verification check.

## Resumability and safety

- Source cases must belong to frozen validation and not test.
- Artifacts are written to a temporary file, flushed, and atomically renamed.
- Resume requires exact provenance and patient identity.
- The one-case runner refuses to overwrite stale or mismatched artifacts.
- Full expert/controller tensors remain in memory only for one case and are
  released before the next case.
- The one-case runner is `scripts/beta2_error_pilot.py`; it refuses a time
  budget above 1,800 seconds and never fits a model.

## Metrics and outputs

Implemented metrics are AUROC, AUPRC, Brier score, log loss, weighted error
prevalence, binned reliability, ECE, and MCE. AUROC is `null` for a one-class
target; AUPRC is `null` when no positive errors exist. These cases are not
silently converted to perfect performance.

`predict_risk_map` returns an error-risk map and `1-q` reliability map.
Uncalibrated output is labeled `uncalibrated_logistic_error_score`; only output
passed through an independently fitted Platt calibrator is labeled calibrated
error probability.
