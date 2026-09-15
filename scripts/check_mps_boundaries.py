"""Verify the requested three MPS epoch crossings from structured events and checkpoints."""
import argparse
from pathlib import Path
import json
import statistics
import torch
from brats_debate.config import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='outputs/mps_boundary_debug/reproduction')
    parser.add_argument('--output',default='outputs/mps_boundary_debug/results.json')
    args=parser.parse_args()
    path=Path(args.run)/'checkpoints/cnn'
    rows=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    finished=[r for r in rows if r['event']=='epoch boundary complete']
    assert len(finished)>=3, 'Need at least three completed boundaries'
    assert rows[-1]['event']=='training completed' and rows[-1]['device']=='mps'
    results=[]
    for end in finished[:3]:
        epoch=end['epoch']; subset=[r for r in rows if r.get('epoch')==epoch]
        complete=next(r for r in subset if r['event']=='epoch training loop complete')
        assert complete['step']==20
        assert {r['step'] for r in subset if r['event']=='training step complete'} >= set(range(11,21))
        for message in ('loss aggregation begin','loss aggregation complete','checkpoint save begin','checkpoint save complete',
                        'validation initialization begin','validation initialization complete',
                        'first validation batch begin','first validation batch complete'):
            assert any(r['event']==message for r in subset), message
        cases=[r for r in subset if r['event'] in ('first validation batch complete','validation case complete')]
        assert len(cases)==2
        before=[r for r in subset if r['event']=='validation probability CPU transfer begin']
        after=[r for r in subset if r['event']=='validation probability CPU transfer complete']
        assert len(before)==len(after)==392
        durations=[b['elapsed_seconds']-a['elapsed_seconds'] for a,b in zip(before,after)]
        results.append({'epoch':epoch,'train_steps':complete['step'],'validation_cases':len(cases),
                        'validation_patches':len(after),'passed':True,
                        'boundary_seconds':end['elapsed_seconds']-complete['elapsed_seconds'],
                        'probability_copy_median_seconds':statistics.median(durations),
                        'probability_copy_max_seconds':max(durations)})
    data=torch.load(path/'last.pt',map_location='cpu',weights_only=True)
    assert data['completed_epoch']==3 and data['global_step']==60 and data['format_version']==2
    assert data['optimizer_state_dict']['state'] and data['rng_state']['mps'].device.type=='cpu'
    result={'passed':True,'device':'mps','tests':results,'completed_epoch':data['completed_epoch'],
            'global_step':data['global_step'],'full_volume_shape':[240,240,155],
            'crop_size':[64,64,64],'num_workers':0,'persistent_workers':False,
            'limits':'Short tests do not establish behavior after 2728 training steps or all 191 validation cases.'}
    save_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
