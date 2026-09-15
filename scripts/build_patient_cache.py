"""Prepare as much deterministic preprocessing as the explicit SSD budget fits."""
import argparse
import json
from pathlib import Path
import time
from brats_debate.config import load_config, save_json
from brats_debate.data.brats_dataset import discover_patients, load_patient, validate_training_audit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/brats.yaml')
    parser.add_argument('--cache-dir',required=True)
    parser.add_argument('--max-gib',type=float,default=1)
    args=parser.parse_args()
    cfg=load_config(args.config)
    validate_training_audit(cfg)
    cfg.update(patient_cache_dir=str(Path(args.cache_dir).resolve()),patient_cache_max_gib=args.max_gib,
               fast_label_mapping=True,reuse_archive_handle=True)
    records=discover_patients(cfg)
    splits=json.loads(Path(cfg['split_file']).read_text())
    # Fixed admission order independent of benchmark's next 100 random samples.
    eligible=[r for r in records if r['patient_id'] in splits['expert_train']]
    eligible += [r for r in records if r['patient_id'] in splits['validation']]
    root=Path(cfg['patient_cache_dir']); root.mkdir(parents=True,exist_ok=True)
    closed=root/f'.admission_closed_{int(args.max_gib*1024**3)}'
    rows=[]
    for record in eligible:
        if closed.exists():break
        from brats_debate.data.patient_cache import cache_path
        existed_before=cache_path(record,cfg).exists()
        start=time.perf_counter(); patient=load_patient(record,cfg)
        cold=time.perf_counter()-start
        start=time.perf_counter(); cached=load_patient(record,cfg)
        warm=time.perf_counter()-start
        import numpy as np
        assert np.array_equal(patient['image'],cached['image'])
        assert np.array_equal(patient['label'],cached['label'])
        assert np.array_equal(patient['affine'],cached['affine'])
        assert patient['header'].binaryblock==cached['header'].binaryblock
        rows.append(dict(patient_id=record['patient_id'],first_read_seconds=cold,second_read_seconds=warm,
                         cache_present_before=existed_before,cache_present_after=cache_path(record,cfg).exists()))
        del patient,cached
        print(json.dumps(rows[-1]),flush=True)
    save_json(root/'build_report.json',dict(cases=len(list(root.glob('*.lz4'))),
              bytes=sum(p.stat().st_size for p in root.glob('*.lz4')),reads=rows,
              note='Cold means absent preprocessing cache, not a purged operating-system page cache.'))


if __name__=='__main__':main()
