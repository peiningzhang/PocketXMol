"""Evaluate bounded N16 extension after its frozen trend gate; no resampling."""
import argparse,fcntl,json,time
from pathlib import Path
from evaluate_stage1 import evaluate
from merge_stage1_n16 import merge
ROOT=Path('outputs_test_time_scaling/stage_1')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--watch',action='store_true');a=p.parse_args()
    decision=json.loads((ROOT/'tables/n16_decision.json').read_text())
    if decision['decision']!='bounded_N16_extension_warranted' or not decision['baseline_complete']:
        raise SystemExit('Await frozen full-core upward trend gate')
    root=ROOT/'n16_extension'
    if not (root/'extension_manifest.json').exists():raise SystemExit('Await extension generation manifest')
    manifest=json.loads((root/'extension_manifest.json').read_text())
    lock=(root/'evaluation.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        records=json.loads((root/'n16_generation_ledger.json').read_text())
        for seed in manifest['seeds']:
            tails=[r for r in records if r['kind']=='pool_tail' and r['seed']==seed and r['status']=='done']
            if len(tails)>1:raise ValueError('Ambiguous authoritative tail')
            if tails:
                merge(ROOT/f'pool_t100_n8_r0_s{seed}',Path(tails[0]['outdir']),
                      root/f'pool_t100_n16_r0_s{seed}',manifest['pocket_ids'])
        evaluate(root,vina_workers=1)
        planned=[Path(r['outdir']) for r in records if r['status']=='done' and r['kind']!='pool_tail']
        planned += [root/f'pool_t100_n16_r0_s{seed}' for seed in manifest['seeds']]
        if (root/'GENERATION_COMPLETE').exists() and len(planned)==9 and all((p/'EVALUATED').exists() for p in planned):
            (root/'EVALUATION_COMPLETE').touch();return
        if not a.watch:return
        time.sleep(20)
if __name__=='__main__':main()
