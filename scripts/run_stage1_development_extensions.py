"""Fill selected-method development cells after immutable configuration freeze.

Never reads extension outcomes for configuration selection. Reuses the original
three-seed development cells; fills missing budgets/seeds for full100 summaries.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.stage1_resources import charged_hours

ROOT=Path('outputs_test_time_scaling/stage_1')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--retry_failed',action='store_true');a=p.parse_args()
    manifest_path=ROOT/'confirmation/confirmation_manifest.json';manifest=json.loads(manifest_path.read_text())
    for file,digest in manifest['code_sha256'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen implementation changed')
    root=ROOT/'development_extensions';root.mkdir(exist_ok=True)
    lock=(root/'driver.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    path=root/'extension_ledger.json';ledger=json.loads(path.read_text()) if path.exists() else []
    original=json.loads((ROOT/'development/adaptation_ledger.json').read_text())
    repairs=ROOT/'development_conditioning_v2/partial_conditioning_repair_ledger.json'
    if repairs.exists():
        repaired=json.loads(repairs.read_text()); labels={r['label'] for r in repaired if r['status']=='done'}
        original=[r for r in original if r['label'] not in labels]+repaired
    original=[r for r in original if r['method']!='partial' or (Path(r['outdir'])/'run.json').exists() and json.loads((Path(r['outdir'])/'run.json').read_text()).get('partial_conditioning_version',1)==2]
    for budget in manifest['budgets']:
        for seed in manifest['seeds']:
            for config in manifest['selected']:
                task=config['task'];label=f"{config['name']}_b{budget}_s{seed}"
                matching=[r for r in original if r['status']=='done' and r['seed']==seed and r['budget']==budget and r['method']==config['family'] and
                    ((r['method']=='fk' and r['interval']==task['interval'] and r['strength']==task['strength']) or
                     (r['method']=='partial' and r['selection']==task['selection']))]
                if matching:
                    if len(matching)!=1:raise ValueError('Ambiguous original development cell')
                    continue
                prior=[r for r in ledger if r['label']==label]
                if any(r['status']=='done' for r in prior):continue
                if prior and not a.retry_failed:continue
                if charged_hours()+20*budget*.03/3600+.02>=48:
                    (root/'BUDGET_LIMIT').write_text(str(charged_hours()));return
                dest=root/label
                if dest.exists():
                    dest=root/(label+f'_retry{len(prior)}')
                    if dest.exists():raise RuntimeError(f'Inspect output: {dest}')
                script='scripts/sample_fk_particles.py' if config['family']=='fk' else 'scripts/sample_partial_refinement.py'
                cmd=[sys.executable,script,'--outdir',str(dest),'--pocket_ids',str(ROOT/'development_ids.txt'),
                     '--seed',str(seed),'--batch_size','4','--steps','100','--warmup']
                if config['family']=='fk':cmd+=['--particles',str(budget//100),'--interval',str(task['interval']),'--strength',str(task['strength'])]
                else:cmd+=['--selection',task['selection'],'--rounds',str(budget//100-1),'--noise_start','1.0','--no_diagnostic_states']
                rec=dict(label=label,status='running',started=time.time(),method=config['name'],budget=budget,seed=seed,
                         command=cmd,outdir=str(dest),configuration_selected_before_extensions=True)
                ledger.append(rec);path.write_text(json.dumps(ledger,indent=2))
                with (root/(dest.name+'.log')).open('w') as stream:result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
                rec.update(status='done' if result.returncode==0 else 'failed',returncode=result.returncode,wall_seconds=time.time()-rec['started'])
                if (dest/'run.json').exists():rec['graph_nfe']=json.loads((dest/'run.json').read_text())['graph_forward_calls']
                path.write_text(json.dumps(ledger,indent=2))
                if result.returncode:raise SystemExit(result.returncode)
    (root/'GENERATION_COMPLETE').touch()

if __name__=='__main__':main()
