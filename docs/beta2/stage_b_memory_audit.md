# Stage B memory audit and bounded benchmark plan

Status: **Optimized and synthetically verified; no additional patient inference
has been run.**

## Evidence from the first pilot

The `240 x 240 x 155` pilot grid contains 8,928,000 voxels. At this size:

- one float32 scalar map is 34.06 MiB;
- one four-class float32 map is 136.23 MiB;
- one int64 segmentation is 68.12 MiB;
- four expert probability maps are 544.92 MiB; and
- a hypothetical materialized 65-channel controller volume would be 2.16 GiB.

The frozen gate already operated on depth-eight slabs, so it did not create the
last allocation. The original Stage B path did, however, retain all four rich
expert dictionaries, a full production debate, a full controller-weight
volume, and then a second full uncertainty/debate analysis during extraction.
The production debate includes six expert pairs, each with hard disagreement,
probability distance, and validity maps. These overlapping lifetimes are the
main evidence-backed source of memory pressure.

The first pilot reported 3,610.91 MiB peak process RSS while system swap grew
by 7,440.63 MiB. On Apple silicon, CPU and MPS use unified physical memory;
PyTorch MPS driver allocations and macOS pressure can therefore exceed what
Python RSS alone suggests. RSS, MPS allocator bytes, MPS driver bytes, and swap
are overlapping views and must not be summed. The 2.13 MiB NPZ is persistent
storage, not working memory. No HD95 or other evaluation metric is called by
the extraction path.

PyTorch exposes current allocator and driver bytes but no MPS peak API in the
installed version. The runner now samples both every 0.5 seconds and separately
records process peak RSS, system swap, and free filesystem space.

## Accepted optimizations

Only the Stage B runner changes; frozen Beta 1 production inference remains
unchanged.

1. Each expert output is compacted immediately to probabilities, uncertainty,
   confidence, compact features, margin, and availability. Boundary evidence
   is retained only for the Boundary expert.
2. The first three hard segmentations remain live only until the unchanged
   HighRes proposal is formed. Logits, mutual-information maps, and unused
   extras are released after each expert.
3. Debate, Jensen-Shannon disagreement, predictive entropy, 65-channel gate
   features, and gating are computed on the same depth-eight slabs.
4. Only final probabilities, JS disagreement, and predictive entropy survive
   the slab loop. Full controller weights and the second full disagreement
   computation are eliminated.
5. CPU-to-MPS slab buffers are materialized once and explicitly released.
6. MPS cached blocks are released once after all expert models return to CPU;
   cache clearing is not performed inside inference or slab loops.
7. Existing artifacts with different provenance are never overwritten.

Synthetic multi-slab tests prove exact equality for final probabilities,
controller weights, JS, predictive entropy, sampled coordinates and order,
features, targets, inclusion probabilities, weights, strata, and metadata.
The compact and legacy HighRes proposals are also exactly equal.

## Deliberately unchanged

The four probability maps and compact expert feature maps are required by the
frozen 65-channel gate and cannot be discarded before gating. The first three
hard segmentations are required to form the HighRes proposal. Final
probabilities, JS, and entropy remain full-volume until deterministic sampling
because error strata and inclusion probabilities depend on whole-case counts.
Precision, patch sizes, HighRes cap, proposal semantics, feature definitions,
sampling, checkpoints, and model architecture are unchanged.

## Proposed one-case benchmark

Subject to explicit approval, run only validation case
`BraTS-GLI-00019-000` with the existing 1,800-second hard timeout. Do not run
the prior pilot case or any test case.

The run will record:

- total, checkpoint/load, expert, controller, disagreement, extraction, and
  save wall times;
- process peak RSS;
- sampled peak MPS current allocation and driver allocation;
- swap before and after;
- free internal disk before and after;
- artifact bytes;
- controller probability reconstruction, simplex, and Stage A/production JS
  residuals; and
- immutable split, checkpoint, protocol, subject, and code provenance.

The expected runtime remains approximately 12-15 minutes, bounded at 30
minutes. Expected artifact storage is below 3 MiB. Working-memory and swap
improvement cannot be claimed until this controlled measurement is complete.
The Kingston SSD is not needed for the benchmark.
