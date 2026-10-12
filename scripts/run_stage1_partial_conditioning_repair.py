"""Replace only incompatible fixed-context feature trials, preserving old data.

Wait for the existing GPU driver PID to exit, then execute missing version2
partial cells. Original candidate files and scores remain diagnostic evidence.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_stage1_adaptations import tasks
from scripts.stage1_resources import charged_hours

ROOT=Path('outputs_test_time_scaling/stage_1')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--after_pid',type=int,required=True);p.add_argument('--retry_failed',action='store_true');a=p.parse_args()
    root=ROOT/'development_conditioning_v2';root.mkdir(exist_ok=True)
    lock=(root/'driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while Path(f'/proc/{a.after_pid}/cmdline').exists():
        command=Path(f'/proc/{a.after_pid}/cmdline').read_bytes()
        if b'scripts/run_stage1_adaptations.py' not in command:break
        time.sleep(20)
    original=json.loads((ROOT/'development/adaptation_ledger.json').read_text())
    complete={r['label']:Path(r['outdir']) for r in original if r['status']=='done'}
    path=root/'partial_conditioning_repair_ledger.json';ledger=json.loads(path.read_text()) if path.exists() else []
    for task in tasks():
        if task['method']!='partial':continue
        old=complete.get(task['label'])
        if old and json.loads((old/'run.json').read_text()).get('partial_conditioning_version',1)==2:continue
        prior=[r for r in ledger if r['label']==task['label']]
        if any(r['status']=='done' for r in prior):continue
        if prior and not a.retry_failed:continue
        if charged_hours()+20*task['budget']*.03/3600+.02>=48:
            (root/'BUDGET_LIMIT').write_text(str(charged_hours()));return
        folder=root/task['label']
        if folder.exists():
            folder=root/(task['label']+f'_retry{len(prior)}')
            if folder.exists():raise RuntimeError(f'Inspect existing output: {folder}')
        cmd=[sys.executable,'scripts/sample_partial_refinement.py','--outdir',str(folder),'--pocket_ids',str(ROOT/'development_ids.txt'),
             '--seed',str(task['seed']),'--batch_size','4','--steps','100','--warmup','--selection',task['selection'],'--rounds',str(task['rounds']),'--noise_start','1.0']
        rec=dict(task,status='running',started=time.time(),outdir=str(folder),command=cmd,
                 replaces_incompatible_conditioning=str(old) if old else None);ledger.append(rec);path.write_text(json.dumps(ledger,indent=2))
        if old:
            (old/'INCOMPATIBLE_CONDITIONING.json').write_text(json.dumps({'reason':'Exact fixed context was enforced, but teacher fixed_halfdist features for context internal edges were absent','formal_comparison':False,'replacement':str(folder),'candidate_files_preserved':True},indent=2))
        with (root/(folder.name+'.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        rec.update(status='done' if r.returncode==0 else 'failed',returncode=r.returncode,wall_seconds=time.time()-rec['started'])
        if (folder/'run.json').exists():rec['graph_nfe']=json.loads((folder/'run.json').read_text())['graph_forward_calls']
        path.write_text(json.dumps(ledger,indent=2))
        if r.returncode:raise SystemExit(r.returncode)
    (root/'GENERATION_COMPLETE').touch()

if __name__=='__main__':main()
