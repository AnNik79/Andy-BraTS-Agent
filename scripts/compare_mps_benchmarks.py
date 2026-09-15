"""Compare paired real-data artifacts without loading all full volumes at once."""
import json
from pathlib import Path
import numpy as np
import torch
from brats_debate.config import save_json


def compare_tree(a,b):
    if isinstance(a,torch.Tensor):
        assert a.shape==b.shape and a.dtype==b.dtype
        return float((a-b).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        return max((compare_tree(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        return max((compare_tree(x,y) for x,y in zip(a,b)),default=0.)
    assert a==b,(a,b)
    return 0.


def main():
    root=Path('outputs/mps_performance')
    before=json.loads((root/'before/results.json').read_text())
    after=json.loads((root/'after/results.json').read_text())
    assert before['patient_ids']==after['patient_ids']
    assert before['validation_ids']==after['validation_ids']
    cases=[]
    for i in range(5):
        a=np.load(root/f'before/probabilities_{i}.npy',mmap_mode='r')
        b=np.load(root/f'after/probabilities_{i}.npy',mmap_mode='r')
        maximum=0.
        for x in range(a.shape[2]):
            maximum=max(maximum,float(np.abs(a[:,:,x]-b[:,:,x]).max()))
        changes=int(np.count_nonzero(a.argmax(1)!=b.argmax(1)))
        assert maximum<=1e-6 and changes==0
        assert before['validation_cases'][i]==after['validation_cases'][i]
        cases.append(dict(patient_id=before['validation_ids'][i],max_probability_abs=maximum,
                          changed_segmentation_voxels=changes,validation_statistics_exact=True))
        del a,b
    a=torch.load(root/'before/benchmark.pt',weights_only=True)
    b=torch.load(root/'after/benchmark.pt',weights_only=True)
    weights=compare_tree(a['state_dict'],b['state_dict'])
    optimizer=compare_tree(a['optimizer_state_dict'],b['optimizer_state_dict'])
    rng=compare_tree(a['rng_state'],b['rng_state'])
    assert weights<1e-5 and optimizer<1e-5 and rng==0
    report=dict(cases=cases,after_100_steps_max_parameter_abs=weights,
                after_100_steps_max_optimizer_abs=optimizer,rng_state_exact=rng==0,
                max_training_loss_abs=float(np.max(np.abs(np.asarray(before['losses'])-after['losses']))),
                source_epoch=a['epoch'],source_next_batch=before['next_batch'],
                final_global_step=a['global_step'],passed=True)
    save_json(root/'equivalence.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
