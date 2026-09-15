"""Bounded MPS candidate experiments, separate from full-run throughput gates."""
import argparse
import json
from pathlib import Path
import time
import resource
import numpy as np
import torch
from brats_debate.config import load_config, save_json
from brats_debate.data.brats_dataset import discover_patients, load_patient, BraTSPatches
from brats_debate.experts import build_expert
from brats_debate.inference.patches import sliding_probabilities
from brats_debate.training.train_expert import segmentation_loss
from brats_debate.training.checkpoint import restore_rng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['validation', 'training', 'compile', 'data', 'data_pipeline'])
    args = ap.parse_args()
    cfg = load_config('configs/brats.yaml')
    out = Path('outputs/mps_performance')
    torch.set_num_threads(4)
    state = torch.load(out / 'backup/progress.pt', weights_only=True)
    records = discover_patients(cfg)
    splits = json.loads(Path(cfg['split_file']).read_text())
    val = [r for r in records if r['patient_id'] in splits['validation']]
    results = []
    def record(**row):
        results.append(row)
        save_json(out / f'experiment_{args.mode}.json', results)
        print(json.dumps(row), flush=True)
    if args.mode == 'data_pipeline':
        for candidate,options in [('original',{}),('reuse_zip',dict(reuse_archive_handle=True)),
                                  ('fast_labels',dict(fast_label_mapping=True)),
                                  ('both',dict(reuse_archive_handle=True,fast_label_mapping=True))]:
            durations=[]
            for case in val[:5]:
                start=time.perf_counter(); patient=load_patient(case,dict(cfg,**options))
                durations.append(time.perf_counter()-start)
                expected=load_patient(case,cfg)
                assert np.array_equal(patient['image'],expected['image'])
                assert np.array_equal(patient['label'],expected['label'])
                del patient,expected
            record(candidate=candidate,seconds_per_case=sum(durations)/len(durations),exact=True)
        return
    if args.mode == 'data':
        import lz4.frame
        patient = None
        for i in range(3):
            start = time.perf_counter()
            patient = load_patient(val[0], cfg)
            record(candidate='archive', repeat=i, seconds=time.perf_counter()-start)
        from brats_debate.data.preprocessing import sample_center
        for flat in [False, True]:
            start=time.perf_counter()
            centers=[sample_center(patient['label'],np.random.default_rng(seed),flat=flat) for seed in range(10)]
            record(candidate='flat_sampling' if flat else 'original_sampling',seconds_per_call=(time.perf_counter()-start)/10)
        # Exact normalized values, losslessly compressed; no random transforms cached.
        raw = patient['image'].tobytes()
        start = time.perf_counter()
        packed = lz4.frame.compress(raw)
        record(candidate='lz4_normalized_write', seconds=time.perf_counter()-start, bytes=len(packed))
        (out / 'cache_probe.lz4').write_bytes(packed)
        del raw, packed
        for i in range(3):
            start = time.perf_counter()
            restored = np.frombuffer(lz4.frame.decompress((out / 'cache_probe.lz4').read_bytes()), dtype=np.float32).reshape(patient['image'].shape)
            record(candidate='lz4_normalized_read', repeat=i, seconds=time.perf_counter()-start,
                   exact=bool(np.array_equal(restored, patient['image'])))
        np.save(out / 'cache_probe.npy', patient['image'])
        for i in range(3):
            start = time.perf_counter()
            restored = np.load(out / 'cache_probe.npy', mmap_mode='r')
            checksum = float(restored.sum())  # Touch every page, not just the header.
            record(candidate='npy_mmap_read', repeat=i, seconds=time.perf_counter()-start, checksum=checksum)
        # CPU argmax is independent per voxel; compare equivalent kernels.
        p = np.load(out / 'before/probabilities_0.npy')
        reference = None
        for threads in [1,2,4,8]:
            torch.set_num_threads(threads)
            start=time.perf_counter(); pred=torch.from_numpy(p).argmax(1)[0].numpy()
            record(candidate='torch_argmax', threads=threads, seconds=time.perf_counter()-start)
            if reference is None: reference=pred
        start=time.perf_counter(); pred=np.argmax(p[0],axis=0)
        record(candidate='numpy_argmax',seconds=time.perf_counter()-start,exact=bool(np.array_equal(pred,reference)))
        for threads in [1,2,4,8]:
            torch.set_num_threads(threads)
            from brats_debate.data.preprocessing import augment
            x=torch.from_numpy(patient['image'][:,:64,:64,:64].copy()); y=torch.from_numpy(patient['label'][:64,:64,:64].copy())
            start=time.perf_counter()
            for _ in range(20): augment(x,y)
            record(candidate='augmentation',threads=threads,seconds_per_call=(time.perf_counter()-start)/20)
        return
    assert torch.backends.mps.is_available()
    device = torch.device('mps')
    model = build_expert('cnn', cfg).to(device)
    model.load_state_dict(state['state_dict'])
    patient = load_patient(val[0],cfg)
    image=torch.from_numpy(patient['image'])[None]
    if args.mode == 'validation':
        model.eval()
        reference=np.load(out/'before/probabilities_0.npy',mmap_mode='r')
        for accumulation in ['cpu','model']:
            for batch in [1,2,4,8,16]:
                torch.mps.empty_cache(); torch.mps.synchronize()
                start=time.perf_counter()
                try:
                    actual=sliding_probabilities(model,image,cfg['crop_size'],device,.5,
                            batch_size=batch,accumulation_device=accumulation,cached_geometry=True)
                    torch.mps.synchronize()
                    seconds=time.perf_counter()-start
                    diff=np.abs(actual.numpy()-reference)
                    record(candidate=accumulation,batch=batch,seconds=seconds,max_abs=float(diff.max()),
                           mean_abs=float(diff.mean()),prediction_changes=int(np.count_nonzero(actual.numpy().argmax(1)!=reference.argmax(1))),
                           driver_bytes=torch.mps.driver_allocated_memory(),allocated_bytes=torch.mps.current_allocated_memory(),
                           rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    del actual,diff
                    if torch.mps.driver_allocated_memory()>6*1024**3: break
                except (RuntimeError,NotImplementedError) as error:
                    record(candidate=accumulation,batch=batch,error=str(error));break
        return
    train=[r for r in records if r['patient_id'] in splits['expert_train']]
    restore_rng(state['rng_state'],device)
    x,y=BraTSPatches(train,cfg)[state['epoch_indices'][state['next_batch']]]
    x,y=x[None].to(device),y[None].to(device)
    if args.mode=='compile':
        try:
            compiled=torch.compile(model.segmentation_logits)
            start=time.perf_counter()
            compiled(x).sum().backward();torch.mps.synchronize()
            record(candidate='torch.compile_inductor',first_call_seconds=time.perf_counter()-start)
        except Exception as error:
            record(candidate='torch.compile_inductor',error=str(error))
        return
    reference=None
    for candidate in ['standard','minimal','channels_last_3d','autocast_float16','autocast_bfloat16','adamw_foreach']:
        try:
            torch.mps.empty_cache()
            model=build_expert('cnn',cfg).to(device)
            model.load_state_dict(state['state_dict'])
            if candidate=='channels_last_3d': model=model.to(memory_format=torch.channels_last_3d)
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],foreach=True if candidate=='adamw_foreach' else None)
            optimizer.load_state_dict(state['optimizer_state_dict'])
            if candidate=='adamw_foreach':
                for group in optimizer.param_groups: group['foreach']=True
            timings=[]
            first=None
            for iteration in range(6):
                model.load_state_dict(state['state_dict']);optimizer.load_state_dict(state['optimizer_state_dict'])
                if candidate=='adamw_foreach':
                    for group in optimizer.param_groups:group['foreach']=True
                restore_rng(state['rng_state'],device)
                model.train();optimizer.zero_grad(set_to_none=True)
                v=x.contiguous(memory_format=torch.channels_last_3d) if candidate=='channels_last_3d' else x
                torch.mps.synchronize(); start=time.perf_counter()
                with torch.autocast('mps',dtype=torch.bfloat16 if candidate=='autocast_bfloat16' else torch.float16,enabled=candidate.startswith('autocast')):
                    p=model(v)['probabilities'] if candidate=='standard' else model.segmentation_logits(v).float().softmax(1)
                    loss=segmentation_loss(p,y)
                scalar=float(loss.detach().item());loss.backward();optimizer.step()
                torch.mps.synchronize();timings.append(time.perf_counter()-start)
                if iteration==0:
                    first=(p.detach().cpu(),torch.cat([v.detach().flatten().cpu() for v in model.parameters()]))
                del p,loss
            if reference is None: reference=first
            record(candidate=candidate,seconds=sum(timings[1:])/5,loss=scalar,
                   probability_max_abs=float((first[0]-reference[0]).abs().max()),
                   parameter_max_abs=float((first[1]-reference[1]).abs().max()),
                   finite=bool(torch.isfinite(first[1]).all()),driver_bytes=torch.mps.driver_allocated_memory())
        except (RuntimeError,NotImplementedError) as error:
            record(candidate=candidate,error=str(error))


if __name__=='__main__':main()
