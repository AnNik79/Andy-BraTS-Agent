# Beta 2 equation-to-code audit

Status as of 2026-10-12:

- **Implemented** means executable code exists.
- **Tested** means synthetic mathematical tests pass.
- **Experimentally evaluated** requires held-out real-data evidence. None of the
  new Beta 2 equations has this status yet.
- **Proposed** means the definition and experiment are specified but code or
  evidence is incomplete.
- **Blocked** means a scientific prerequisite is unavailable.

The frozen Beta 1 forward path remains unchanged. New observational code is in
`src/brats_debate/uncertainty/`.

## Symbols and tensor conventions

- `B,E,C,D,H,W`: batch, expert, class, and voxel-grid dimensions.
- `E=4`, with expert order CNN, Transformer, Boundary, HighRes.
- `C=4`, with the configured BraTS internal class order `0,1,2,3`.
- `p_k`: `[B,C,D,H,W]`; stacked `P`: `[B,E,C,D,H,W]`.
- Availability `A`: `[B,E,1,D,H,W]`; unavailable experts contribute neither
  probability mass nor disagreement.
- Gate weights `w`: `[B,E,D,H,W]`.
- Labels `y`: `[B,D,H,W]`.
- Natural logarithms are used. Entropy/JS maps exposed to the controller are
  divided by `log(C)`; raw terms are retained in Beta 2 analysis.

## 1. Expert probability distributions

**Equation**

`p_k(v) in Delta^(C-1), k=1,...,4`, and
`P(v)={p_1(v),p_2(v),p_3(v),p_4(v)}`.

1. **Definition:** every available expert emits a nonnegative, finite
   four-class vector summing to one at every aligned voxel.
2. **Existing implementation:** `Expert.pack_output`, sliding-window overlap
   averaging, and `stack_evidence`.
3. **Source files:** `experts/base.py`, `inference/patches.py`,
   `debate/features.py`, and `uncertainty/probability.py`.
4. **Missing functionality:** empirical probability calibration remains
   unimplemented.
5. **Inputs:** four expert probability volumes, class order, voxel grid, and
   availability maps.
6. **Supervision:** none for representation checks; labels are needed only for
   calibration/evaluation.
7. **New code:** `validate_probability_tensor`,
   `stack_expert_probabilities`, and `ExpertProbabilityBatch`.
8. **Limitations:** softmax normalization does not imply calibrated posterior
   probability; spatial registration is checked by shape/device, while affine
   identity remains an inference-pipeline responsibility.
9. **Tests:** simplex, zero probabilities, malformed values, missing experts,
   class order, grid mismatch, and all-unavailable voxels.
10. **Experiment:** measure class/region calibration on subject-held-out
    predictions. Status: **Implemented, Tested; not experimentally evaluated**.

## 2. Individual expert entropy

**Equation**

`H(p_k(v)) = -sum_c p_(k,c)(v) log p_(k,c)(v)`.

1. **Definition:** Shannon entropy of one expert's class distribution.
2. **Existing implementation:** normalized entropy in `experts/base.py`.
3. **Source files:** `experts/base.py` and
   `uncertainty/disagreement.py`.
4. **Missing functionality:** real-data calibration and region-conditioned
   interpretation.
5. **Inputs:** one expert probability vector.
6. **Supervision:** none to compute; labels for predictive-value experiments.
7. **New code:** `entropy_map` exposes raw or `log(C)`-normalized entropy and
   `analyze_uncertainty` preserves each expert separately.
8. **Limitations:** predictive entropy mixes multiple uncertainty sources and
   is not clinical ambiguity.
9. **Tests:** uniform distribution has normalized entropy one; one-hot vectors
   have entropy zero; zero entries are numerically stable.
10. **Experiment:** compare entropy-only error detection by subject. Status:
    **Implemented, Tested; not experimentally evaluated**.

## 3. Mean expert prediction

**Equation**

`p_bar(v) = (1/4) sum_(k=1)^4 p_k(v)`.

1. **Definition:** equal-weight arithmetic mean when all four experts exist.
2. **Existing implementation:** masked mean in production disagreement.
3. **Source files:** `debate/disagreement.py` and
   `uncertainty/probability.py`.
4. **Missing functionality:** none for the mathematical operation.
5. **Inputs:** aligned probabilities and availability.
6. **Supervision:** none.
7. **New code:** `masked_mean_probability`; it is exactly the displayed
   equation when all four availability maps equal one.
8. **Limitations:** with missing experts the denominator is the available count,
   so that generalized quantity is not the literal four-expert formula.
9. **Tests:** equal-four mean and missing-expert mean.
10. **Experiment:** compare mean-probability baseline with majority vote and
    controller using frozen cases. Status: **Implemented, Tested; Beta 1
    baseline evaluated, Beta 2 analysis pending**.

## 4. Jensen-Shannon disagreement by KL

**Equation**

`D_JS(v) = (1/4) sum_(k=1)^4 D_KL(p_k(v) || p_bar(v))`.

1. **Definition:** average KL divergence from each expert to the mixture.
2. **Existing implementation:** production computes the equivalent entropy
   identity, normalized by `log(C)`.
3. **Source files:** `debate/disagreement.py` and
   `uncertainty/disagreement.py`.
4. **Missing functionality:** empirical error-prediction validation.
5. **Inputs:** `P`, `p_bar`, availability.
6. **Supervision:** none to compute; voxel error labels to evaluate.
7. **New code:** explicit zero-safe weighted-KL form
   `js_from_weighted_kl`.
8. **Limitations:** JS measures model disagreement, not correctness; identical
   wrong experts yield zero.
9. **Tests:** identical, uniform, opposed one-hot, unavailable-expert, and
   production-parity cases.
10. **Experiment:** Q1/Q2 subject-grouped JS versus error and entropy. Status:
    **Implemented, Tested; not experimentally evaluated**.

## 5. Jensen-Shannon entropy identity

**Equation**

`D_JS(v) = H(p_bar(v)) - (1/4)sum_k H(p_k(v))`.

1. **Definition:** entropy of the expert mixture minus mean individual entropy.
2. **Existing implementation:** this is the production Beta 1 JS computation.
3. **Source files:** `debate/disagreement.py` and
   `uncertainty/disagreement.py`.
4. **Missing functionality:** none mathematically.
5. **Inputs:** expert and mean probabilities.
6. **Supervision:** none.
7. **New code:** both sides and `js_identity_residual` are exposed.
8. **Limitations:** clamping may produce floating-point residuals near zero;
   availability changes the equal-four average to a masked average.
9. **Tests:** KL and entropy forms agree within numerical tolerance.
10. **Experiment:** no separate scientific experiment; numerical verification
    accompanies every generated analysis batch. Status: **Implemented, Tested**.

## 6. Probability variance, voting, and pairwise disagreement

**Definitions**

Per-class expert variance, hard-vote disagreement
`1-max_c votes_c / E_available`, and all six expert pairs.

1. **Definition:** complementary views of probability dispersion and hard-label
   conflict.
2. **Existing implementation:** per-class variance, class votes, pairwise hard
   disagreement, and total variation.
3. **Source files:** `debate/disagreement.py`, `analysis/pairwise.py`, and
   `uncertainty/disagreement.py`.
4. **Missing functionality:** cohort-scale boundary and region reports.
5. **Inputs:** probabilities, hard predictions, availability, optional masks.
6. **Supervision:** none unless masks derive from labels.
7. **New code:** explicit vote-disagreement map plus generic region/boundary
   summaries; existing pairwise tensors are reused.
8. **Limitations:** vote ties resolve to the lowest class in Beta 1; probability
   and hard-vote disagreement answer different questions.
9. **Tests:** pair enumeration, missing experts, vote fractions, and summaries.
10. **Experiment:** evaluate each definition as an error predictor and report
    multiplicity. Status: **Implemented, Tested; not experimentally evaluated**.

## 7. Voxel error

**Equation**

`E(v) = 1[y_hat(v) != y(v)]`, where `y_hat` is the frozen controller label.

1. **Definition:** multiclass hard-label error at a voxel.
2. **Existing implementation:** generic segmentation error maps.
3. **Source files:** `analysis/error_maps.py`.
4. **Missing functionality:** a streamed Beta 2 dataset joining frozen
   controller outputs with predictors.
5. **Inputs:** controller segmentation and ground truth.
6. **Supervision:** ground-truth segmentation is mandatory.
7. **New code:** no duplicate map implementation is needed.
8. **Limitations:** voxels are spatially and subject-correlated; class error is
   distinct from WT/TC/ET binary-region error.
9. **Tests:** existing false-positive/false-negative/region-map tests.
10. **Experiment:** construct targets only on permitted development subjects.
    Status: **Implemented, Tested; dataset construction Proposed**.

## 8. Conditional error risk

**Equations**

`P(E=1 | D)` and `P(E=1 | D,H,B,X)`.

1. **Definition:** conditional probability of frozen-controller error from
   disagreement alone or combined evidence.
2. **Existing implementation:** descriptive AUROC/enrichment exists, but no
   calibrated conditional model.
3. **Source files:** `evaluation/disagreement_analysis.py`,
   `evaluation/evaluate.py`, and future `uncertainty/error_prediction.py`.
4. **Missing functionality:** grouped sampling, model fitting, calibration, and
   held-out evaluation.
5. **Inputs:** JS/other disagreement, entropy, boundary features, restricted MRI
   evidence, and error target.
6. **Supervision:** controller error labels on fit/calibration partitions.
7. **New code:** Stage B error-prediction dataset/model/evaluation modules.
8. **Limitations:** these conditionals depend on data distribution and feature
   definition; they are not identified by descriptive correlation.
9. **Tests:** group isolation, feature ordering, bounded outputs, and no test
   subjects.
10. **Experiment:** disagreement-only, entropy-only, and combined held-out
    comparisons. Status: **Proposed**.

## 9. Logistic error model and reliability

**Equations**

`q(v)=sigmoid(beta_0 + beta_d D(v) + beta_h H(v) + beta_b B(v) + beta_x X(v))`
and `Q(v)=1-q(v)`.

1. **Definition:** calibrated estimate of hard segmentation error, followed by
   correctness reliability.
2. **Existing implementation:** none; schema placeholders are not producers.
3. **Source files:** future `uncertainty/error_prediction.py` and
   `uncertainty/calibration.py`.
4. **Missing functionality:** all fitting, calibration, persistence, and map
   export.
5. **Inputs:** vector-valued feature blocks, frozen error targets, subject IDs.
6. **Supervision:** disjoint subject-grouped fit and calibration labels.
7. **New code:** regularized logistic baselines, calibration fit/transform, and
   metadata labeling calibrated versus uncalibrated outputs.
8. **Limitations:** voxel subsampling and prevalence shifts affect intercept and
   calibration; standard errors cannot treat voxels as independent.
9. **Tests:** sigmoid range, known synthetic coefficients, calibration
   monotonicity, persistence, and group leakage.
10. **Experiment:** AUROC, AUPRC, Brier, log loss, reliability curves, and
    high-risk enrichment with patient bootstrap intervals. Status: **Proposed**.

## 10. Controller simplex and fusion

**Equations**

`w(v)=f_theta(X(v),P(v),U(v),D(v),B(v))`, `w(v) in Delta^3`, and
`p_CA(v)=sum_k w_k(v)p_k(v)`.

1. **Definition:** nonnegative available-expert weights summing to one and their
   probability mixture.
2. **Existing implementation:** 65-to-8-to-4 pointwise gate, masked softmax,
   and weighted fusion.
3. **Source files:** `controller/gating_network.py` and `debate/features.py`.
4. **Missing functionality:** no training change is needed; behavioral evidence
   is missing.
5. **Inputs:** frozen controller features, expert probabilities, availability.
6. **Supervision:** controller was trained on `controller_train`; trust analysis
   needs separate labeled cases.
7. **New code:** validation in `uncertainty/expert_trust.py`; no Beta 1 forward
   modification.
8. **Limitations:** weights are allocation coefficients, not calibrated
   probabilities or causal explanations.
9. **Tests:** masked simplex, zero unavailable weight, fusion equality, gradient
   tests already present.
10. **Experiment:** Q4-Q6 controller trust and error-stratified analysis.
    Status: **Implemented, Tested; Beta 1 fusion evaluated, trust pending**.

## 11. Theoretical optimal weights

**Equation**

`w*(v)=argmin_(w in Delta^3) E[L(sum_k w_k p_k(v),y(v)) | Z(v)]`,
with `Z=[X,P,U,D,B]`.

1. **Definition:** conditional risk minimizer over the weight simplex.
2. **Existing implementation:** the trained gate is an approximation learned
   with finite data and Dice/cross-entropy objectives, not a solved `w*`.
3. **Source files:** `training/train_controller.py` and
   `controller/gating_network.py`.
4. **Missing functionality:** oracle/risk-gap analysis under specified losses.
5. **Inputs:** conditional features, expert probabilities, labels, chosen loss.
6. **Supervision:** labeled controller-training data; unbiased assessment needs
   held-out subjects.
7. **New code:** Stage A trust metrics implement hard-expert oracle coverage and
   controller gap; continuous simplex-oracle optimization remains proposed.
8. **Limitations:** global nonlocal losses make voxelwise optimization
   inseparable; combining probabilities can beat every hard expert top class.
9. **Tests:** synthetic fix/harm/oracle cases.
10. **Experiment:** compare controller, hard oracle, and constrained
    probability-mixture oracle without tuning on test. Status: **Partially
    Implemented, Tested; continuous oracle Proposed**.

## 12. Correct-expert trust metrics

**Definitions**

Correct-expert weight mass `sum_k w_k 1[argmax p_k=y]`, maximum-weight expert
correctness, hard-expert oracle coverage, fusion fixes, harms, and all-wrong
events.

1. **Definition:** observational answers to whether the controller allocates
   weight to experts whose hard labels are correct.
2. **Existing implementation:** none in Beta 1.
3. **Source files:** `uncertainty/expert_trust.py`.
4. **Missing functionality:** streamed real-case aggregation and patient
   confidence intervals.
5. **Inputs:** `P`, `w`, fused probabilities, availability, labels, optional
   region/boundary/disagreement masks.
6. **Supervision:** ground truth.
7. **New code:** per-voxel maps, global and stratified summaries, and
   weight/correctness correlations.
8. **Limitations:** hard correctness discards probability quality and does not
   establish why the controller selected a weight.
9. **Tests:** oracle, all-wrong, soft-fusion rescue, harm, regional, boundary,
   and disagreement synthetic cases.
10. **Experiment:** Q4-Q6/Q13 on grouped development subjects. Status:
    **Implemented, Tested; not experimentally evaluated**.

## 13. Composite anatomical loss

**Equation**

`L=lambda_1 L_Dice + lambda_2 L_HD95 + lambda_3 L_volume + lambda_4 L_topology`.

1. **Definition:** configurable weighted training objective.
2. **Existing implementation:** expert/controller training uses segmentation
   losses; the full four-term objective does not exist.
3. **Source files:** `training/train_expert.py`,
   `training/train_controller.py`, and future Beta 2 loss module.
4. **Missing functionality:** differentiable surrogates, spacing propagation,
   configuration, and ablations.
5. **Inputs:** soft predictions, labels, spacing, region definitions.
6. **Supervision:** segmentation labels.
7. **New code:** Stage C loss components in an isolated Beta 2 path.
8. **Limitations:** weights have different units/scales; improved surrogate loss
   need not improve true HD95/topology.
9. **Tests:** finite gradients, empty regions, anisotropic spacing, and each
   lambda ablation.
10. **Experiment:** separate Beta 2 checkpoints and controlled patient-level
    comparisons. Status: **Proposed; retraining requires approval**.

## 14. Dice term

**Equation**

For region `T`, `L_Dice=1-(2 sum_v s_T(v)y_T(v)+eps) /
(sum_v s_T(v)+sum_v y_T(v)+eps)`, averaged over specified tumor regions.

1. **Definition:** differentiable soft overlap loss.
2. **Existing implementation:** multiclass Dice plus cross-entropy in training.
3. **Source files:** `training/train_expert.py` and
   `training/train_controller.py`.
4. **Missing functionality:** explicit Beta 2 region-specific configuration.
5. **Inputs:** region probabilities and binary labels.
6. **Supervision:** labels.
7. **New code:** reuse/refactor only if isolated loss composition requires it.
8. **Limitations:** overlap does not encode physical boundary distance.
9. **Tests:** empty/perfect/partial overlap and gradient flow.
10. **Experiment:** existing loss baseline versus composite ablations. Status:
    **Implemented in Beta 1; Beta 2 composition Proposed**.

## 15. HD95 term

**Equation**

`HD95(A,B)=percentile_95({d(a,B):a in boundary(A)} union
{d(b,A):b in boundary(B)})`.

1. **Definition:** symmetric 95th-percentile physical surface distance.
2. **Existing implementation:** nondifferentiable evaluation in millimetres.
3. **Source files:** `evaluation/metrics.py`.
4. **Missing functionality:** differentiable training surrogate.
5. **Inputs:** binary region masks and voxel spacing.
6. **Supervision:** labels.
7. **New code:** specify a distance-transform/surface surrogate, while retaining
   true HD95 only for evaluation.
8. **Limitations:** true percentile, thresholding, and surface extraction are
   not generally differentiable; one-sided empty HD95 is undefined.
9. **Tests:** anisotropic spacing, empty cases, surrogate gradients.
10. **Experiment:** train with surrogate and evaluate true HD95 plus Dice.
    Status: **Evaluation Implemented; training surrogate Proposed**.

## 16. Volume-consistency term

**Equation**

For region `T`, one operational form is
`L_volume=|sum_v s_T(v) Delta V_v - sum_v y_T(v) Delta V_v| /
(sum_v y_T(v) Delta V_v + eps)`.

1. **Definition:** relative soft expected-volume error in physical units.
2. **Existing implementation:** hard volumes only in reasoning summaries.
3. **Source files:** `reasoning/reasoning_engine.py` and future
   `uncertainty/volume.py`.
4. **Missing functionality:** soft volume, uncertainty, interval evaluation,
   and loss implementation.
5. **Inputs:** region probability, labels, spacing/affine-derived voxel volume.
6. **Supervision:** labels for loss/evaluation.
7. **New code:** Stage C physical-volume utility and optional loss.
8. **Limitations:** matching total volume does not ensure correct location or
   anatomy.
9. **Tests:** units, anisotropic voxels, empty region, gradient.
10. **Experiment:** volume bias/absolute error and Dice/HD95 trade-offs. Status:
    **Proposed**.

## 17. Topology term

**Operational definition**

Topology must be declared before use, for example a differentiable clDice-style
connectivity surrogate per region, with component counts reported separately.

1. **Definition:** penalize specified connectivity/skeleton inconsistency, not
   an unspecified “biological plausibility.”
2. **Existing implementation:** none.
3. **Source files:** future isolated Beta 2 loss module.
4. **Missing functionality:** exact operational choice and validation.
5. **Inputs:** region probabilities and labels.
6. **Supervision:** labels.
7. **New code:** only after the topology target is approved.
8. **Limitations:** BraTS tumors can be multifocal; forcing one component would
   be unsupported and potentially wrong.
9. **Tests:** synthetic components/holes, empty regions, gradients.
10. **Experiment:** topology metrics plus conventional segmentation metrics.
    Status: **Proposed; biological constraint selection Blocked pending
    approval**.

## 18. Local optimal weights versus global loss

1. **Definition:** `w*(v)` is voxel-local only when the conditional loss
   decomposes over voxels.
2. **Existing implementation:** gate is pointwise, while Dice and proposed
   surface/volume/topology losses couple many voxels.
3. **Source files:** controller and training modules.
4. **Missing functionality:** explicit global-objective experiments.
5. **Inputs:** full volumes for nonlocal terms.
6. **Supervision:** labels.
7. **New code:** document and optimize the actual global objective rather than
   claiming it solves the displayed local argmin independently.
8. **Limitations:** HD95, volume, and topology make one voxel's optimal weight
   depend on other voxels.
9. **Tests:** gradient influence beyond a single voxel for nonlocal losses.
10. **Experiment:** local-only versus global-objective ablation. Status:
    **Mathematically documented; implementation Proposed**.

## 19. Information pathway

**Conceptual graph**

`X -> Z -> Y`, with expert predictions and disagreement as observables.

1. **Definition:** a conceptual latent-information model, not an identified
   causal graph from current data.
2. **Existing implementation:** exploratory latent representations, SVD, and
   Mahalanobis novelty.
3. **Source files:** `analysis/extract.py`, `analysis/svd.py`, and
   `analysis/mahalanobis.py`.
4. **Missing functionality:** preregistered restricted representation
   `X_tilde` and conditional comparison.
5. **Inputs:** MRI evidence, frozen representations, disagreement, labels.
6. **Supervision:** labels for predictive evaluation.
7. **New code:** Stage D controlled feature ablations and optional justified
   estimators.
8. **Limitations:** `Z` is latent and not uniquely recoverable; arrows do not
   prove causality.
9. **Tests:** deterministic feature construction and split guards.
10. **Experiment:** compare matched predictors with/without `D`. Status:
    **Exploratory tools Implemented; framework Proposed**.

## 20. Conditional information in disagreement

**Equations**

`I(Y;D|X)` and `I(E;D|H,X)`.

1. **Definition:** conditional mutual information under explicitly defined
   random variables and distributions.
2. **Existing implementation:** none.
3. **Source files:** future Stage D analysis.
4. **Missing functionality:** restricted conditioning representation,
   estimator choice, uncertainty quantification.
5. **Inputs:** `Y` or `E`, disagreement, entropy, and `X_tilde`.
6. **Supervision:** labels.
7. **New code:** prefer held-out conditional predictive comparisons before
   high-dimensional CMI estimation.
8. **Limitations:** if deterministic experts make `D=g(X)` and `X` is the
   complete input, then both conditional-information quantities involving `D`
   are zero. Positive incremental value is meaningful only relative to a
   restricted representation/model class or stochastic mechanism.
9. **Tests:** feature ablation identity and synthetic zero-CMI case.
10. **Experiment:** likelihood/Brier/log-loss improvement from adding `D` to
    fixed `H,X_tilde`; do not label this exact CMI without a justified
    estimator. Status: **Proposed**.

## 21. Segmentation probability and reliability output

**Equations**

`S_T(v)=Pr(Y_v in T)=sum_(c in T) p_(CA,c)(v)` and `Q(v)=1-q(v)`.

1. **Definition:** region-membership probability and calibrated correctness
   probability.
2. **Existing implementation:** fused class probabilities and hard NIfTI masks;
   no `q/Q`.
3. **Source files:** `controller/gating_network.py`,
   `inference/pipeline.py`, and future calibration/output modules.
4. **Missing functionality:** region probability exports, error-risk/reliability
   maps, and structured provenance.
5. **Inputs:** fused probabilities and calibrated error model.
6. **Supervision:** none at inference after calibration; labels for fitting.
7. **New code:** Stage C aligned NIfTI/metadata exports.
8. **Limitations:** class confidence, entropy, disagreement, and calibrated
   correctness are distinct; if uncalibrated, `Q` must be labeled a score.
9. **Tests:** region sums, shapes, affine preservation, metadata status.
10. **Experiment:** Q7/Q8 calibration and enrichment. Status: **S component
    available internally; explicit outputs and Q Proposed**.

## 22. Expected probabilistic tumor volume

**Equation**

`V_hat_T = sum_v S_T(v) Delta V_v`.

1. **Definition:** expected region volume under voxel marginal probabilities.
2. **Existing implementation:** only hard-mask volume.
3. **Source files:** `reasoning/reasoning_engine.py` and future
   `uncertainty/volume.py`.
4. **Missing functionality:** expected WT/TC/ET volumes and evaluation.
5. **Inputs:** region probabilities and physical voxel volumes.
6. **Supervision:** none to compute; labels for bias/error analysis.
7. **New code:** streamed scalar accumulation supporting spacing/affine.
8. **Limitations:** calibrated voxel marginals are required for probabilistic
   interpretation.
9. **Tests:** constant probabilities, anisotropic spacing, millilitre conversion.
10. **Experiment:** patient-level bias, MAE, and agreement versus reference
    volume. Status: **Proposed**.

## 23. Approximate tumor-volume variance

**Equation**

`Var(V_hat_T) approx sum_v p_v(1-p_v)(Delta V_v)^2`.

1. **Definition:** independent-Bernoulli marginal variance approximation.
2. **Existing implementation:** none.
3. **Source files:** future `uncertainty/volume.py`.
4. **Missing functionality:** implementation and empirical interval coverage.
5. **Inputs:** `p_v=S_T(v)` and voxel volumes.
6. **Supervision:** none to compute; labels for coverage evaluation.
7. **New code:** stable scalar accumulation and explicit approximation metadata.
8. **Limitations:** spatial covariance terms are omitted; segmentation voxels
   are strongly dependent, so this is not exact uncertainty.
9. **Tests:** `p=0/1` gives zero, `p=0.5` maximum, units are squared volume.
10. **Experiment:** coverage of nominal intervals; report failure if
    under/over-dispersed. Status: **Proposed**.

## 24. Active-perception cost efficiency

**Equation**

`a*=argmax_a E[uncertainty reduction|a]/C(a)`.

1. **Definition:** rank feasible actions by estimated reduction per positive
   cost.
2. **Existing implementation:** HighRes proposal selection and an active
   learning priority score, not this action policy.
3. **Source files:** `inference/pipeline.py`, `analysis/active_learning.py`, and
   future `active_perception/`.
4. **Missing functionality:** registry, feasibility, cost models, outcome
   models, policy, and offline evaluation.
5. **Inputs:** current evidence, feasible actions, expected outcomes, costs.
6. **Supervision:** retrospective paired/action-outcome data or explicit
   simulation assumptions.
7. **New code:** Stage E action/utility/policy/evaluation modules.
8. **Limitations:** identical deterministic reruns provide no new information;
   unavailable modalities and nonexistent resolution cannot be selected.
9. **Tests:** unavailable actions excluded, positive cost, deterministic no-gain,
   ranking, and stop behavior.
10. **Experiment:** selected versus random/no-action uncertainty and error
    reduction per cost. Status: **Proposed**.

## 25. Active-perception entropy utility

**Equation**

`a*=argmax_a [H(Y|X)-E_(X_a)H(Y|X,X_a)-lambda C(a)]`.

1. **Definition:** expected information gain minus cost penalty.
2. **Existing implementation:** none.
3. **Source files:** future `active_perception/utility.py`.
4. **Missing functionality:** posterior/outcome estimator and validated costs.
5. **Inputs:** current predictive distribution, action-outcome distribution,
   updated predictive distribution, `lambda`, cost.
6. **Supervision:** offline action outcomes or a declared simulation model.
7. **New code:** entropy utility with abstain/stop when no feasible positive
   net utility exists.
8. **Limitations:** entropy reduction can be confidently wrong and is not the
   same as clinical benefit.
9. **Tests:** zero information action, cost monotonicity, ranking, abstention.
10. **Experiment:** entropy and actual error changes, both reported. Status:
    **Proposed**.

## 26. Structured reasoning constraint

**Invariant**

The reasoning expert consumes structured evidence and
`can_modify_segmentation=False`.

1. **Definition:** language output is downstream observational explanation only.
2. **Existing implementation:** structured summary, validated provider response,
   deterministic fallback, and immutable-mask contract.
3. **Source files:** `reasoning/reasoning_engine.py` and
   `reasoning/providers.py`.
4. **Missing functionality:** Beta 2 evidence schema fields for calibrated
   risk, volume uncertainty, and actions.
5. **Inputs:** only measured/derived structured evidence.
6. **Supervision:** curated factuality test cases; no voxel training labels
   required for schema validation.
7. **New code:** Stage F versioned schema and unsupported-claim checks.
8. **Limitations:** schema constraints reduce but do not eliminate unsupported
   generation from a live model.
9. **Tests:** missing evidence, contradictory claims, no diagnosis/treatment,
   no mask modification.
10. **Experiment:** blinded structured-summary factuality against source JSON.
    Status: **Beta 1 invariant Implemented, Tested; Beta 2 schema Proposed**.
