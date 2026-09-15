"""Build the measured optimization report after all local gates pass."""
import json
from pathlib import Path
import statistics
from brats_debate.config import save_json


def main():
    root=Path('outputs/mps_performance')
    read=lambda name: json.loads((root/name).read_text())
    before,after=read('before/results.json'),read('after/results.json')
    equivalence,resumes=read('equivalence.json'),read('mps_resume.json')
    assert equivalence['passed'] and len(resumes)==4 and all(r['passed'] for r in resumes)
    events=[json.loads(line) for line in (root/'boundaries/checkpoints/cnn/events.jsonl').read_text().splitlines()]
    boundaries=[r for r in events if r['event']=='epoch boundary complete']
    assert len(boundaries)==3
    stages_before=read('attribution_validation_before/results.json')
    stages_after=read('attribution_after/results.json')
    cache=read('patient_cache/build_report.json')
    cached={p.name.rsplit('-',1)[0] for p in (root/'patient_cache').glob('*.lz4')}
    hits=sum(pid in cached for pid in after['patient_ids'])
    estimates=[]
    for r in [before,after]:
        train=r['seconds_per_step']*2728
        validation=r['seconds_per_case']*191
        checkpoint=29*r['checkpoint_seconds']
        estimates.append(dict(training_seconds=train,validation_seconds=validation,
                         checkpoint_seconds=checkpoint,epoch_seconds=train+validation+checkpoint,
                         training_fraction=train/(train+validation),validation_fraction=validation/(train+validation)))
    remaining=46*estimates[1]['epoch_seconds']+328*after['seconds_per_step']+191*after['seconds_per_case']+5*after['checkpoint_seconds']
    rows=[('Training step',before['seconds_per_step'],after['seconds_per_step'],'s'),
          ('100 training steps',before['training_seconds'],after['training_seconds'],'s'),
          ('Validation case',before['seconds_per_case'],after['seconds_per_case'],'s'),
          ('5 validation cases',before['validation_seconds'],after['validation_seconds'],'s'),
          ('Estimated full epoch',estimates[0]['epoch_seconds']/3600,estimates[1]['epoch_seconds']/3600,'h'),
          ('Estimated 50 epochs',50*estimates[0]['epoch_seconds']/3600,50*estimates[1]['epoch_seconds']/3600,'h')]
    table='| Stage | Before | After | Speedup |\n|---|---:|---:|---:|\n'+'\n'.join(f'| {label} | {a:.3f} {unit} | {b:.3f} {unit} | {a/b:.2f}× |' for label,a,b,unit in rows)
    profile='| Training stage (s/step) | Before | After |\n|---|---:|---:|\n'
    for key in ['data_batch','load_patient','nifti_decode','normalize','sample_center','augmentation','cpu_to_mps','forward','loss','scalar_read','backward','optimizer']:
        profile+=f'| {key} | {stages_before["training_stages"][key]/20:.5f} | {stages_after["training_stages"][key]/20:.5f} |\n'
    validation='| Validation stage (s/case) | Before | After |\n|---|---:|---:|\n'
    for key in ['validation_load','sliding_inference','validation_loss','validation_metrics']:
        validation+=f'| {key} | {before["validation_stages"][key]/5:.4f} | {after["validation_stages"][key]/5:.4f} |\n'
    optional={str(frequency):dict(hours_saved=(50-50//frequency)*estimates[1]['validation_seconds']/3600,
                                total_hours=(50*estimates[1]['epoch_seconds']-(50-50//frequency)*estimates[1]['validation_seconds'])/3600)
              for frequency in [2,5]}
    summary=dict(before=before,after=after,estimates=estimates,remaining_active_hours=remaining/3600,
                 weighted_epoch_speedup=estimates[0]['epoch_seconds']/estimates[1]['epoch_seconds'],
                 combined_benchmark_speedup=(before['training_seconds']+before['validation_seconds'])/(after['training_seconds']+after['validation_seconds']),
                 cache_cases=cache['cases'],benchmark_cache_hits=hits,equivalence=equivalence,
                 resumes=resumes,boundaries=boundaries,optional_validation_frequency=optional,
                 full_job_restarted=False)
    save_json(root/'summary.json',summary)
    command='''cd /Users/andersonliew/CancerAgents
PYTHONPATH=src:.venv-deps MPLCONFIGDIR=/private/tmp/brats-mpl \\
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_ENABLE_MPS_FALLBACK=1 \\
/opt/anaconda3/bin/python scripts/start_cnn_training.py \\
  --config configs/brats_mps_optimized.yaml \\
  --resume outputs/brats2023_gli_cnn_mps_fixed/checkpoints/cnn/progress.pt'''
    document=f'''# Local M2 Pro CNN performance report

The paired real-data benchmark projects **{summary['weighted_epoch_speedup']:.2f}× faster full epochs**, with the research protocol unchanged. The full job remains stopped. This is a projection from 100 training steps and five full validation cases, not a measured optimized 2728-step/191-case epoch. The combined short benchmark improved {summary['combined_benchmark_speedup']:.2f}×.

{table}

Estimates include 27 progress saves plus two epoch-boundary saves per epoch. Best-file copying is negligible. From the saved epoch-4/batch-2400 cursor, approximately **{remaining/3600:.1f} hours of additional active runtime** remain at these measured rates. Sleep, contention, thermal throttling and longer-run behavior can change this estimate. The original three completed epochs averaged about 3.52 hours each and spent approximately 47.3% training / 52.7% validating. The current short benchmark weights instead imply **{100*estimates[0]['training_fraction']:.1f}% / {100*estimates[0]['validation_fraction']:.1f}% before**, and **{100*estimates[1]['training_fraction']:.1f}% / {100*estimates[1]['validation_fraction']:.1f}% after**, excluding checkpoint overhead. The historical validation rate was slower than the five-case benchmark; a long-run 2× improvement has not yet been demonstrated.

## Measured bottlenecks and attribution

Training called the full public expert output builder. `pack_output()` computed unused feature summaries, segmentation, top-k and entropy. `loss.detach().item()` waited for that queued MPS work. It was not retaining an epoch of loss tensors or graphs. The CNN loss now calls the same trainable layers/dropout through `segmentation_logits()` and the same float32 softmax and loss. Public expert predictions remain unchanged.

The following attribution run uses 20 real steps with explicit synchronization between stages, with prefetch disabled to separate costs. These diagnostic synchronizations are **not** present in production. Nested data rows overlap: `data_batch` contains loading, sampling, augmentation and collation; `load_patient` contains decoding and normalization. Do not sum every row.

{profile}

The 100-step throughput run has no added per-stage synchronization. Its old/new data-batch waits were {before['training_stages']['data_batch']:.2f}/{after['training_stages']['data_batch']:.2f} seconds; background loading consumed {after['training_stages']['load_patient']:.2f} seconds overlapping MPS. Asynchronous `.to()` and `.item()` wall times include pending device work, so they are not isolated copy/kernel costs.

{validation}

CPU argmax was measured separately: PyTorch at 1/2/4/8 threads took 0.627/0.900/1.171/1.422 seconds; NumPy took 0.092 seconds with identical tie-breaking/output. Per-case totals also include this conversion, logging and benchmark probability-file writes, identically in before/after runs.

The ordinary probability-copy timers included pending inference: {before['transfer_including_pending_gpu_seconds']/5:.3f} → {after['transfer_including_pending_gpu_seconds']/5:.3f} s/case. A separate synchronized one-case experiment attributed {stages_before['transfer_including_pending_gpu_seconds']:.3f} → {stages_after['transfer_including_pending_gpu_seconds']:.3f} seconds to the CPU-transfer event itself, and {stages_before['validation_stages']['validation_window_compute_and_h2d']:.3f} → {stages_after['validation_stages']['validation_window_compute_and_h2d']:.3f} seconds to window preparation, device transfer and forward/softmax. Synchronization perturbs scheduling; those attribution numbers should not replace end-to-end totals. Checkpoint save measured {before['checkpoint_seconds']:.3f} → {after['checkpoint_seconds']:.3f} seconds and was left intact.

## Attempts and decisions

* **Accept minimal CNN training outputs:** fixed-patch forward/loss/backward/AdamW fell from 1.259 to 0.331 s; no trainable layer, probability/loss formula or dropout call changed. One-step probability difference was zero; maximum parameter difference was 7.45e-9.
* **Accept ZIP handle reuse and faster validated label mapping:** five-case loading averaged 0.611 s originally, 0.497 with ZIP reuse, 0.534 with label mapping, and 0.423 together. All images/masks matched exactly. CRC reads, finite/integer label checks and geometry checks remain.
* **Accept flat candidate enumeration:** 0.0736 → 0.0103 s/crop center; preserves C-order enumeration and the exact RNG draw, including specialist modes. Avoids allocating three coordinates for every background voxel.
* **Accept bounded deterministic prefetch:** only loading occurs in one background thread; crop sampling, augmentation and dropout stay on the main thread. Twenty real steps averaged 0.852 s without prefetch and 0.488 s with prefetch at four threads. `num_workers=0`, no persistent workers, and existing cursors/RNG remain valid.
* **Retain four CPU threads:** prefetch runs at 2/4/8 threads measured 0.508/0.488/0.518 s/step. Augmentation alone was similar across thread counts. No thread count or augmentation change is hidden in the experiment.
* **Accept a bounded, checksummed lossless LZ4 SSD cache:** {cache['cases']} training cases use {cache['bytes']/1024**3:.3f} GiB, or {100*cache['cases']/682:.1f}% of training cases. The after benchmark hit it on {hits}/100 steps; it was built in fixed sorted split order, independently of benchmark sample selection. Full normalized volumes cannot fit on the available SSD (roughly 20 GiB for training+validation at this sample compression ratio; uncompressed would be much larger). Stable admission prevents sequential scans from evicting every cached case. Misses read the original ZIP. No random augmentation is cached.
* **Cache read alternatives measured:** archive loading 0.56–0.58 s; lossless image-only LZ4 decoding 0.056–0.083 s; mmap full-image reads 0.022–0.039 s. `.npy` was rejected for disk footprint. Actual complete cache records include image, compact labels, affine, NIfTI header and spacing; every built case was compared bitwise to source. `patient_cache/build_report.json` records per-case cold build and warm read times. “Cold” means an absent preprocessing cache, not a purged OS page cache. Raw source files are never modified.
* **Accept CPU NumPy argmax and bounded cached window geometry:** same probability-only validation, uniform overlap averaging, all 196 windows, all 191 cases each full epoch, same loss and metrics. Public final/debate prediction code remains unchanged.
* **Reject larger validation window batches:** CPU accumulation at batches 1/2/4/8/16 took 13.97/15.29/14.54/14.93/15.02 s on one real case. Batch 16 was the largest tested within a conservative 6 GiB driver-memory budget (5.53 GB observed); deliberately did not probe an OOM threshold on a 16 GB Mac. Batch 1 is the throughput choice.
* **Reject full-volume MPS accumulation and chunked transfers as defaults:** MPS accumulation at batches 1/2/4/8/16 took 15.01/14.99/14.11/14.19/16.22 s. This tested one full-case detached transfer. Batched CPU accumulation tested the hybrid/chunk alternative. Neither improved on batch-1 CPU accumulation. Largest probability difference was 2.98e-7 at batch 16; no segmentation voxel changed. Candidate paths remain explicitly selectable for reproducibility, not enabled in the optimized config.
* **Reject mixed precision:** float16/bfloat16 measured 0.416/0.399 s versus float32 0.331 s, with max probability differences 0.00181/0.01422. No convergence claim is made from these short probes; neither is accepted. MPS remains float32.
* **Do not adopt channels-last-3D or foreach AdamW:** 0.321 and 0.330 s versus minimal 0.331 s in the small GPU probe; insufficient end-to-end benefit under the measured data-loading bottleneck. The original contiguous model and optimizer settings remain. Increasing training batch size would change the batch-reduced Dice loss/optimizer semantics, so validation microbatches were tested without changing training batches.
* **Reject `torch.compile` for this run:** actual Inductor forward/backward did not complete within the explicit 90-second probe limit and was terminated. This does not establish that MPS compilation is universally unsupported; it establishes no demonstrated useful speedup here. Removing unused operators was the proven graph-level simplification; no custom fusion or changed loss was adopted.

## Correctness, memory and resume

The suite passed **50 tests**. Regression coverage includes public/minimal fixed forwards, one optimizer step, sampler RNG, prefetch augmentation/RNG, cache values/geometry/corruption recovery, validation padding/partial window batches and tie breaking, and exact CPU resumes from mid-epoch, pre-validation and completed-epoch checkpoints. Relocating a resumed run now preserves a compatible historical best checkpoint without replacing the live progress model/optimizer.

Five real validation volumes had **zero probability differences, zero changed segmentation voxels and identical validation loss/Dice**. Identical predictions and targets also preserve every configured class/region metric. After the paired 100 training steps: maximum weight difference {equivalence['after_100_steps_max_parameter_abs']:.3g}, optimizer-state difference {equivalence['after_100_steps_max_optimizer_abs']:.3g}, loss difference {equivalence['max_training_loss_abs']:.3g}; Python/NumPy/CPU Torch/MPS RNG states matched exactly. GPU roundoff is therefore not described as bitwise training determinism.

Three real MPS epochs of 20 steps + two complete validation cases crossed the exact production epoch-boundary path in one three-epoch process; global steps were 20/40/60. Separate MPS resumes from progress and pre-validation checkpoints reached the same completed epoch/global step with equivalence checks; completed-epoch resume restored the next-epoch cursor and preserved the historical best. Detailed differences are in `outputs/mps_performance/mps_resume.json`.

Process high-water RSS was {before['max_rss_bytes']/1024**3:.3f} → {after['max_rss_bytes']/1024**3:.3f} GiB. Sampled MPS driver high-water memory (including allocator cache) was {before['peaks']['mps_driver_bytes']/1024**3:.3f} → {after['peaks']['mps_driver_bytes']/1024**3:.3f} GiB. These are distinct unified-memory accounting measures; do not add them. Samples were taken after training steps and validation inference, so transient allocations not retained by the allocator may be missed. No OOM occurred.

The untouched source checkpoint and `outputs/mps_performance/backup/progress.pt` share SHA-256 `{resumes[-1]['sha256']}`. Loading under the optimized full config restored model weights, AdamW state and RNG **exactly**, epoch 4, completed epoch 3, global step 10584, next step **2401/2728**, best validation metric 0.768654118447957. Last logged step was 2420; those 20 unsaved steps must be repeated. Scheduler state is explicitly `None`; scaler state is retained. Protocol and training signatures match. This is not weights-only initialization.

## Optional validation-frequency changes — not implemented

Keeping the training budget fixed, validating every two epochs would save approximately {optional['2']['hours_saved']:.1f} h over 50 epochs (estimated total {optional['2']['total_hours']:.1f} h); every five would save {optional['5']['hours_saved']:.1f} h (total {optional['5']['total_hours']:.1f} h). These would change the selection/monitoring protocol and require a separate decision. The optimized config still validates every epoch. Full sliding-window validation is now approximately 70% of the projected epoch, the main remaining limit.

## Safe resume command — prepared, not executed

```sh
{command}
```

The new output directory is `outputs/brats2023_gli_cnn_mps_optimized`; the previous run's checkpoints remain preserved. After the new run saves progress, subsequent resumes must point to its new `checkpoints/cnn/progress.pt` to retain that later progress. The config still specifies all 50 epochs, original splits, 682 training cases × four patches, batch size one, 64³ crops, 191 validation cases, unchanged architecture/loss/AdamW/augmentation and probability/metric semantics.

Raw evidence: `outputs/mps_performance/{{before,after}}/results.json`, `experiment_*.json`, `equivalence.json`, `mps_resume.json`, `summary.json`, and preserved source in `baseline_source/`. Reproduce throughput using `scripts/benchmark_mps.py` (baseline package first on PYTHONPATH for before), candidate probes with `scripts/experiment_mps.py`, short production-loop checks with `configs/brats_mps_performance_repro.yaml`, and cache building with `scripts/build_patient_cache.py`. Benchmark caps are confined to those scripts/debug configs, not the full-run config.
'''
    Path('docs/mps_performance.md').write_text(document)
    print(table)
    print(f'Remaining active hours: {remaining/3600:.3f}')


if __name__=='__main__':main()
