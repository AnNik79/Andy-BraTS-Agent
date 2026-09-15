"""Paired real-data benchmark. Never writes to the source run/checkpoint.

Use the preserved baseline package on PYTHONPATH for the before measurement.
Default timings do not insert MPS synchronizations; --synchronized is a separate
attribution experiment, not the throughput measurement.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path
import resource
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from brats_debate.config import load_config, save_json
import brats_debate.data.brats_dataset as data
import brats_debate.training.train_expert as training
from brats_debate.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from brats_debate.experts import build_expert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/brats.yaml')
    parser.add_argument('--checkpoint', default='outputs/mps_performance/backup/progress.pt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--cases', type=int, default=5)
    parser.add_argument('--synchronized', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--optimized', action='store_true')
    parser.add_argument('--prefetch', action='store_true')
    parser.add_argument('--window-batch', type=int, default=1)
    parser.add_argument('--accumulation', choices=['cpu', 'model'], default='cpu')
    parser.add_argument('--cache-dir')
    args = parser.parse_args()
    assert torch.backends.mps.is_available(), 'Real MPS required'
    cfg = load_config(args.config)
    cfg['torch_threads'] = args.threads
    if args.optimized:
        cfg['cnn_probability_only_training'] = True
        cfg['prefetch_patients'] = args.prefetch
        cfg.update(flat_candidate_sampling=True, fast_label_mapping=True, reuse_archive_handle=True,
                   numpy_validation_argmax=True)
        if args.cache_dir:
            cfg.update(patient_cache_dir=str(Path(args.cache_dir).resolve()), patient_cache_max_gib=1)
        cfg['validation_performance'] = dict(batch_size=args.window_batch,
                                            accumulation_device=args.accumulation, cached_geometry=True)
    out = Path(args.output).resolve()
    assert out != Path(cfg['output_dir']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    device = torch.device('mps')
    data.validate_training_audit(cfg)
    records = data.discover_patients(cfg)
    splits = json.loads(Path(cfg['split_file']).read_text())
    train = [r for r in records if r['patient_id'] in splits['expert_train']]
    validation = [r for r in records if r['patient_id'] in splits['validation']][:args.cases]
    model = build_expert('cnn', cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    state = load_training_checkpoint(args.checkpoint, model, optimizer, scaler, cfg, splits, 'cnn', device)
    epoch, offset = state['epoch'], state['next_batch']
    indices = state['epoch_indices'][offset:offset + args.steps]
    assert len(indices) == args.steps
    timer = defaultdict(float)
    peaks = {'mps_allocated_bytes': 0, 'mps_driver_bytes': 0}

    def memory():
        peaks['mps_allocated_bytes'] = max(peaks['mps_allocated_bytes'], torch.mps.current_allocated_memory())
        peaks['mps_driver_bytes'] = max(peaks['mps_driver_bytes'], torch.mps.driver_allocated_memory())

    @contextmanager
    def timed(name):
        if args.synchronized:
            torch.mps.synchronize()
        start = time.perf_counter()
        yield
        if args.synchronized:
            torch.mps.synchronize()
        timer[name] += time.perf_counter() - start

    def wrap(module, name, label):
        original = getattr(module, name)
        def wrapped(*a, **kw):
            with timed(label):
                return original(*a, **kw)
        setattr(module, name, wrapped)
    wrap(data, 'load_patient', 'load_patient')
    wrap(data, 'load_nifti', 'nifti_decode')
    wrap(data, 'normalize_nonzero', 'normalize')
    wrap(data, 'sample_center', 'sample_center')
    wrap(data, 'augment', 'augmentation')
    dataset = data.BraTSPatches(train, cfg)
    loader = data.make_patch_loader(dataset, indices, cfg, cfg['seed'] + epoch + 100000) if args.optimized else DataLoader(dataset, batch_size=cfg['batch_size'], sampler=indices,
                        num_workers=0, persistent_workers=False,
                        generator=torch.Generator().manual_seed(cfg['seed'] + epoch + 100000))
    event = training.RunEvents(out, 'cnn', device, cfg['epochs'])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []  # Python floats only, bounded to the benchmark, not training state.
    torch.mps.synchronize()
    started = time.perf_counter()
    iterator = iter(loader)
    for i in range(args.steps):
        with timed('data_batch'):
            image, target = next(iterator)
        with timed('cpu_to_mps'):
            x, y = image.to(device), target.to(device)
        with timed('forward'):
            output = training.training_output(model, x, 'cnn', cfg) if args.optimized else model(x)
        with timed('loss'):
            loss = training.expert_loss(output, y)
        with timed('scalar_read'):
            scalar = float(loss.detach().item())
        assert np.isfinite(scalar)
        with timed('backward'):
            loss.backward()
        with timed('optimizer'):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(scalar)
        memory()
        del output, loss, x, y, image, target
        if (i + 1) % 10 == 0:
            event('benchmark training progress', step=i+1, loss=scalar)
    torch.mps.synchronize()
    train_seconds = time.perf_counter() - started
    train_timers = dict(timer)
    cursor = {k: state[k] for k in ('epoch', 'completed_epoch', 'global_step', 'next_batch',
              'training_loop_complete', 'running_loss', 'num_batches', 'steps_per_epoch', 'epoch_indices')}
    cursor.update(global_step=state['global_step']+args.steps, next_batch=offset+args.steps,
                  num_batches=state['num_batches']+args.steps, running_loss=state['running_loss']+sum(losses))
    started = time.perf_counter()
    save_training_checkpoint(out / 'benchmark.pt', model, optimizer, scaler, cfg, splits,
                             'cnn', cfg['seed'], cursor, state['best_validation_metric'], device, event)
    checkpoint_seconds = time.perf_counter() - started
    # Validation uses IDENTICAL saved model state before/after, independently of training roundoff.
    model.load_state_dict(state['state_dict'])
    model.eval()
    del state
    timer.clear()
    original_sliding = training.sliding_probabilities
    cases = []
    def sliding(*a, **kw):
        with timed('sliding_inference'):
            p = original_sliding(*a, **kw)
        memory()
        np.save(out / f'probabilities_{len(cases)}.npy', p.numpy())
        return p
    training.sliding_probabilities = sliding
    wrap(training, 'load_patient', 'validation_load')
    wrap(training, 'segmentation_loss', 'validation_loss')
    wrap(training, 'segmentation_metrics', 'validation_metrics')
    transfer_start = None
    patch_start = None
    transfer_seconds = 0.0
    class Events:
        name = 'cnn'
        def __call__(self, message, **values):
            nonlocal transfer_start, transfer_seconds, patch_start
            if args.synchronized and message == 'validation patch forward begin':
                torch.mps.synchronize()
                patch_start = time.perf_counter()
            if args.synchronized and message == 'validation patch forward complete':
                torch.mps.synchronize()
                timer['validation_window_compute_and_h2d'] += time.perf_counter() - patch_start
            if message == 'validation probability CPU transfer begin':
                transfer_start = time.perf_counter()
            if message == 'validation probability CPU transfer complete' and transfer_start:
                transfer_seconds += time.perf_counter() - transfer_start
            if message in ('validation CPU metrics complete',):
                cases.append(values)
            if 'patch' not in message:
                event(message, **values)
    cfg['trace_validation_transfers'] = True
    started = time.perf_counter()
    if validation:
        training.validate_expert(model, validation, cfg, device, Events(), epoch)
    validation_seconds = time.perf_counter() - started
    result = dict(steps=args.steps, cases=args.cases, epoch=epoch, next_batch=offset,
                  training_seconds=train_seconds, seconds_per_step=train_seconds/args.steps,
                  validation_seconds=validation_seconds,
                  seconds_per_case=validation_seconds/args.cases if args.cases else None,
                  checkpoint_seconds=checkpoint_seconds, training_stages=train_timers,
                  validation_stages=dict(timer), transfer_including_pending_gpu_seconds=transfer_seconds,
                  validation_cases=cases, losses=losses, peaks=peaks,
                  max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  synchronized=args.synchronized, threads=args.threads,
                  patient_ids=[train[i//cfg['patches_per_patient']]['patient_id'] for i in indices],
                  validation_ids=[r['patient_id'] for r in validation])
    save_json(out / 'results.json', result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('losses','patient_ids')}, indent=2))


if __name__ == '__main__':
    main()
