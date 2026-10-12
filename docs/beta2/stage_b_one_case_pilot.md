# Stage B one-case extraction pilot

Status: **Engineering smoke test completed; no error predictor was fitted,
calibrated, or scientifically evaluated.**

Date: 2026-10-12

## Case and provenance

- Case: `BraTS-GLI-00006-000`
- Subject: `BraTS-GLI-00006`
- Frozen role: validation
- Test cases used: 0
- Code commit executed: `3e7e89f5cc367eb44fca9ba4bdbfb5c9571afc98`
- Protocol hash: `721b8dcb3bb3a6f84ad82c7ee78a443bef8754743aebc88368e0123159477d98`
- Split hash: `705cf73c10ab6d047b095c461fe8c71329b3e0a3ee01f925ced205747102cfe2`
- Controller SHA-256:
  `86b604082a934a1d4f4a68467a88d16d268ae7653c552dcfc6c3087ba1d79fe2`

The four expert hashes matched the frozen Beta 1 manifest. The case was checked
against the immutable validation and test lists before loading.

## Runtime

- Checkpoints and patient load: 0.72 seconds
- Four-expert inference: 693.08 seconds
- Controller and debate: 25.43 seconds
- Feature extraction and sampling: 28.12 seconds
- Atomic artifact save: 0.15 seconds
- Total measured pipeline: 747.78 seconds (12.46 minutes)
- Command wall clock: 747.95 seconds

The run stayed below the approved 30-minute limit and no other MPS job was
active.

## Memory, swap, and storage

- Peak process RSS: 3610.91 MiB
- System swap used before: 8737.62 MiB
- System swap used after: 16178.25 MiB
- Observed swap growth: 7440.63 MiB
- Compact NPZ artifact: 2,228,634 bytes (2.13 MiB)
- Internal free bytes immediately after extraction: 25,130,852,352
- Later filesystem report: approximately 25 GiB available
- Kingston SSD: not mounted and not required

The artifact is small, but the system-level swap increase is substantial.
Scaling based only on artifact size would therefore be unsafe.

## Alignment and numerical checks

- Native grid: `240 x 240 x 155`
- Class order: `0,1,2,3`
- MRI feature channels: t1n, t1c, t2w, t2f
- Controller and target both contained all four configured classes
- Sampled rows: 68,727 with eight finite features
- Full evaluated domain: 1,519,601 voxels
- Full controller error voxels: 8,727
- Raw sampled error fraction: 0.1269807
- Full-domain error fraction: 0.005742955
- Inverse-probability weighted sampled error fraction: 0.005742955
- Maximum JS KL/entropy-identity residual: `1.40e-7`

The close prevalence match verifies the sampling correction for this case. The
error fraction is a smoke-test observation, not an estimate of Beta 2
performance.

## Resumability

The per-case NPZ contains only sampled rows, indices, inclusion probabilities,
raw inverse-probability weights, strata, and structured provenance. It contains
no full-volume tensor archive. Completion requires matching patient identity,
feature schema, protocol/split/checkpoint hashes, sampling policy, and code
commit. Temporary files are atomically renamed only after a complete write.

## Scaling implication

Naively multiplying the measured 747.78 seconds by 191 cases gives about
39.7 hours of serial runtime. Artifact storage would be approximately 0.4 GiB
if cases were similar, but swap behavior—not output size—is the main blocker.
The one-case measurement does not justify a full run. Before scaling:

1. investigate/reduce retained full-volume tensors and swap growth;
2. benchmark a second case only with separate approval;
3. preserve at least the current internal free-space margin for macOS swap; and
4. do not assume Kingston would solve unified-memory or internal-swap pressure.
