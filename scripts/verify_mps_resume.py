"""Real-data MPS resume gates; all destinations are isolated from the full run."""
import json
from pathlib import Path
import torch
from brats_debate.config import load_config, save_json, file_hash
from brats_debate.data.brats_dataset import discover_patients
from brats_debate.experts import build_expert
from brats_debate.training.train_expert import train_expert
from brats_debate.training.checkpoint import load_training_checkpoint, cpu_snapshot, rng_state, training_signature
from compare_mps_benchmarks import compare_tree


def main():
    assert torch.backends.mps.is_available()
    torch.set_num_threads(4)
    root=Path('outputs/mps_performance')
    cfg=load_config('configs/brats_mps_performance_repro.yaml')
    splits=json.loads(Path(cfg['split_file']).read_text())
    records=discover_patients(cfg)
    original=root/'boundaries/checkpoints/cnn'
    reference=torch.load(original/'last.pt',weights_only=True)
    rows=[]
    for source,destination in [('progress.pt','resume_mid'),('epoch_train_complete.pt','resume_prevalidation'),('last.pt','resume_completed')]:
        resumed_cfg=dict(cfg,output_dir=str((root/destination).resolve()))
        train_expert('cnn',records,splits,resumed_cfg,resume=original/source)
        if source=='last.pt':
            assert file_hash(original/'best.pt')==file_hash(root/destination/'checkpoints/cnn/best.pt')
            rows.append(dict(source=source,passed=True,next_epoch=4,global_step=60))
            continue
        actual=torch.load(root/destination/'checkpoints/cnn/last.pt',weights_only=True)
        weights=compare_tree(reference['state_dict'],actual['state_dict'])
        optimizer=compare_tree(reference['optimizer_state_dict'],actual['optimizer_state_dict'])
        rng=compare_tree(reference['rng_state'],actual['rng_state'])
        dice=abs(reference['validation_dice']-actual['validation_dice'])
        assert weights<1e-5 and optimizer<1e-5 and rng==0 and dice<1e-5
        assert actual['completed_epoch']==3 and actual['global_step']==60
        rows.append(dict(source=source,passed=True,max_parameter_abs=weights,max_optimizer_abs=optimizer,
                         rng_exact=rng==0,validation_dice_abs=dice,global_step=actual['global_step']))
        save_json(root/'mps_resume.json',rows)
    # Inspect the REAL full-run resume with the final config, without taking a training step.
    cfg=load_config('configs/brats_mps_optimized.yaml')
    baseline=load_config('configs/brats.yaml')
    assert training_signature(cfg)==training_signature(baseline)
    source=Path(baseline['output_dir'])/'checkpoints/cnn/progress.pt'
    assert file_hash(source)==file_hash(root/'backup/progress.pt')
    model=build_expert('cnn',cfg).to('mps')
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'])
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    state=load_training_checkpoint(source,model,optimizer,scaler,cfg,splits,'cnn',torch.device('mps'))
    assert compare_tree(state['state_dict'],cpu_snapshot(model.state_dict()))==0
    assert compare_tree(state['optimizer_state_dict'],cpu_snapshot(optimizer.state_dict()))==0
    assert compare_tree(state['rng_state'],rng_state(torch.device('mps')))==0
    rows.append(dict(source=str(source),passed=True,model_and_optimizer_exact=True,rng_exact=True,
                     resume_epoch=state['epoch'],next_training_step=state['next_batch']+1,
                     global_step=state['global_step'],completed_epoch=state['completed_epoch'],
                     best_validation_metric=state['best_validation_metric'],sha256=file_hash(source)))
    save_json(root/'mps_resume.json',rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
