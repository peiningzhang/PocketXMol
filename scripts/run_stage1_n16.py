"""Guarded generation for the predeclared dev20/three-seed N16 extension.

Generate only the eight new candidates, plus both B1600 controls. Prefix reuse,
independent evaluation and paired analysis are required separately.
"""
import argparse,fcntl,json,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.stage1_resources import charged_hours
ROOT=Path('outputs_test_time_scaling/stage_1')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--retry_failed',action='store_true');a=p.parse_args()
    decision=json.loads((ROOT/'tables/n16_decision.json').read_text())
    if not decision.get('baseline_complete') or decision['decision']!='bounded_N16_extension_warranted':
        raise SystemExit('N16 generation not authorized by complete baseline trend gate')
    protocol=json.loads((ROOT/'protocol.json').read_text())
    root=ROOT/'n16_extension';root.mkdir(exist_ok=True)
    lock=(root/'driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ids=ROOT/'development_ids.txt';seeds=protocol['seeds'][:3]
    frozen=dict(decision=decision,pocket_ids=protocol['development_ids'],seeds=seeds,
                new_candidate_repeat_indices=list(range(8,16)),prefix_policy='Reuse original raw N8 candidates; never regenerate prefix',
                controls=['steps1600','native extra15'],scope='generation only; no reward tuning')
    manifest=root/'extension_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=frozen:raise ValueError('Extension manifest differs')
    manifest.write_text(json.dumps(frozen,indent=2));(root/'pocket_ids.txt').write_text(ids.read_text())
    path=root/'n16_generation_ledger.json';ledger=json.loads(path.read_text()) if path.exists() else []
    for seed in seeds:
        prefix=ROOT/f'pool_t100_n8_r0_s{seed}'
        if not (prefix/'EVALUATED').exists():raise ValueError('Authoritative N8 prefix not evaluated')
        for kind,steps,repeats,rounds,start in [('pool_tail',100,8,0,8),('steps',1600,1,0,0),('refine',100,1,15,0)]:
            label=f'{kind}_t{steps}_n{repeats}_r{rounds}_s{seed}'
            prior=[r for r in ledger if r['label']==label]
            if any(r['status']=='done' for r in prior):continue
            if prior and not a.retry_failed:raise RuntimeError('Failed task requires explicit retry flag')
            estimate=20*(800 if kind=='pool_tail' else 1600)*.03/3600+.02
            if charged_hours()+estimate>=48:
                (root/'BUDGET_LIMIT').write_text(str(charged_hours()));return
            dest=root/(label if not prior else label+f'_retry{len(prior)}')
            if dest.exists():raise RuntimeError(f'Inspect untracked output: {dest}')
            cmd=[sys.executable,'scripts/sample_test_time_scaling.py','--outdir',str(dest),
                 '--steps',str(steps),'--repeats',str(repeats),'--start_repeat',str(start),
                 '--max_refine_rounds',str(rounds),'--num_pockets','20','--pocket_ids',str(ids),
                 '--seed',str(seed),'--batch_size','4','--warmup','--config_task',
                 'configs/sample/test/sbdd_csd/'+('base.yml' if kind=='refine' else 'simple.yml')]
            rec=dict(label=label,kind=kind,seed=seed,status='running',outdir=str(dest),
                     started=time.time(),command=cmd,authoritative_prefix=str(prefix))
            ledger.append(rec);path.write_text(json.dumps(ledger,indent=2))
            with (root/(dest.name+'.log')).open('w') as stream:
                result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
            rec.update(status='done' if result.returncode==0 else 'failed',returncode=result.returncode,
                       wall_seconds=time.time()-rec['started'])
            if (dest/'run.json').exists():rec['graph_nfe']=json.loads((dest/'run.json').read_text())['graph_forward_calls']
            path.write_text(json.dumps(ledger,indent=2))
            if result.returncode:raise SystemExit(result.returncode)
    (root/'GENERATION_COMPLETE').touch()
if __name__=='__main__':main()
