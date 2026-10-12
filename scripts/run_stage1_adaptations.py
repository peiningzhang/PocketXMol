"""Bounded developer matrix; 20 pockets x 3 seeds, two adaptation families."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('outputs_test_time_scaling/stage_1')


def tasks():
    for high in [False,True]:
        for seed in [2026,12026,22026]:
            particles=8 if high else 4
            for interval in [20,50]:
                for strength in [2.,5.]:
                    yield dict(method='fk',particles=particles,interval=interval,strength=strength,seed=seed,
                               label=f'fk_p{particles}_i{interval}_b{strength:g}_s{seed}',budget=particles*100)
            rounds=3 if high else 1
            for selection in ['random','confidence']:
                yield dict(method='partial',rounds=rounds,selection=selection,seed=seed,
                           label=f'partial_{selection}_r{rounds}_s{seed}',budget=100*(1+rounds))


def consumed_hours():
    seconds=0.
    for ledger in ROOT.rglob('*ledger.json'):
        for record in json.loads(ledger.read_text()):
            seconds+=record.get('wall_seconds',max(0,time.time()-record['started']) if record.get('status')=='running' else 0)
    return seconds/3600


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--retry_failed',action='store_true');a=p.parse_args()
    root=ROOT/'development';root.mkdir(exist_ok=True)
    lock=(root/'driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=root/'adaptation_ledger.json';ledger=json.loads(path.read_text()) if path.exists() else []
    for task in tasks():
        prior=[r for r in ledger if r['label']==task['label']]
        if any(r['status']=='done' for r in prior):continue
        if prior and not a.retry_failed:continue
        # Conservative wall estimate leaves room for startup and warm-up.
        estimate=20*task['budget']*.03/3600+.02
        if consumed_hours()+estimate>=48:
            (root/'BUDGET_LIMIT').write_text(f'Charged hours {consumed_hours():.3f}; next estimated {estimate:.3f}');return
        dest=root/task['label']
        if dest.exists():
            dest=root/(task['label']+f'_retry{len(prior)}')
            if dest.exists():raise RuntimeError(f'Inspect existing output: {dest}')
        script='scripts/sample_fk_particles.py' if task['method']=='fk' else 'scripts/sample_partial_refinement.py'
        cmd=[sys.executable,script,'--outdir',str(dest),'--pocket_ids',str(ROOT/'development_ids.txt'),
             '--seed',str(task['seed']),'--batch_size','4','--steps','100','--warmup']
        if task['method']=='fk':cmd+=['--particles',str(task['particles']),'--interval',str(task['interval']),'--strength',str(task['strength'])]
        else:cmd+=['--selection',task['selection'],'--rounds',str(task['rounds']),'--noise_start','1.0']
        record=dict(task,status='running',started=time.time(),command=cmd,outdir=str(dest));ledger.append(record)
        path.write_text(json.dumps(ledger,indent=2))
        with (root/(dest.name+'.log')).open('w') as f:
            result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        record.update(status='done' if result.returncode==0 else 'failed',returncode=result.returncode,wall_seconds=time.time()-record['started'])
        if (dest/'run.json').exists():record['graph_nfe']=json.loads((dest/'run.json').read_text())['graph_forward_calls']
        path.write_text(json.dumps(ledger,indent=2))
        if result.returncode:raise SystemExit(result.returncode)
    (root/'GENERATION_COMPLETE').touch()

if __name__=='__main__':main()
