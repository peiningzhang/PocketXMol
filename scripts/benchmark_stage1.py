"""Repeated same-GPU prototype latency, after configuration freeze.

Eight confirmation pockets fixed by index before timing; same seed repeated
three times. Full generation includes reconstruction/I/O. Warm-up, startup,
data handling and cold receptor preparation are reported separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.stage1_resources import charged_hours

ROOT=Path('outputs_test_time_scaling/stage_1')
ENV=Path('/shared/healthinfolab/phz24002/anaconda3/envs')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--trials',type=int,default=3);p.add_argument('--baseline_only',action='store_true');p.add_argument('--trace_control',action='store_true');a=p.parse_args()
    if a.baseline_only:
        protocol=json.loads((ROOT/'protocol.json').read_text())
        manifest=dict(pocket_ids=protocol['confirmation_ids'],selected=[],code_sha256={})
    else:
        manifest=json.loads((ROOT/'confirmation/confirmation_manifest.json').read_text())
    for file,digest in manifest['code_sha256'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen implementation changed')
    root=ROOT/('timing_trace_control' if a.trace_control else 'timing');root.mkdir(exist_ok=True)
    ids=root/'pocket_ids.txt';pockets=manifest['pocket_ids'][::10];assert len(pockets)==8
    ids.write_text('\n'.join(pockets)+'\n')
    policy=dict(pocket_ids=pockets,trials=a.trials,seed=2026,budgets=[400,800],batch_size=4,
                hardware_policy='Run all trials on the same A10040GB GPU; validate external live allocation/GPU state before launch',
                cold_receptor_preparation='Measure separately; primary Vina timing records existing receptor reuse',
                sampling_bitwise_deterministic=False,record_full_trajectory=not a.trace_control)
    if not a.trace_control:policy.pop('record_full_trajectory')
    policy['scope']='baselines' if a.baseline_only else 'baselines_and_frozen_selected_methods'
    policy_path=root/('timing_protocol_baselines.json' if a.baseline_only else 'timing_protocol.json')
    if policy_path.exists() and json.loads(policy_path.read_text())!=policy:
        raise ValueError('Benchmark protocol already frozen with different parameters')
    policy_path.write_text(json.dumps(policy,indent=2))
    ledger_path=root/'timing_ledger.json';ledger=json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    done={r['label'] for r in ledger if r['status']=='done'}
    for trial in range(a.trials):
        for budget in [400,800]:
            configs=[dict(name='geometry_confidence',script='sample_test_time_scaling.py',args=['--steps','100','--repeats',str(budget//100)]),
                dict(name='more_steps',script='sample_test_time_scaling.py',args=['--steps',str(budget)]),
                dict(name='native_refinement',script='sample_test_time_scaling.py',args=['--steps','100','--max_refine_rounds',str(budget//100-1),'--config_task','configs/sample/test/sbdd_csd/base.yml'])]
            for selected in manifest['selected']:
                t=selected['task']
                if selected['family']=='fk':args=['--steps','100','--particles',str(budget//100),'--interval',str(t['interval']),'--strength',str(t['strength'])];script='sample_fk_particles.py'
                else:args=['--steps','100','--selection',t['selection'],'--rounds',str(budget//100-1),'--noise_start','1.0','--no_diagnostic_states'];script='sample_partial_refinement.py'
                configs.append(dict(name=selected['name'],script=script,args=args))
            for config in configs:
                label=f"{config['name']}_b{budget}_trial{trial}";folder=root/label
                if label in done:continue
                if folder.exists():raise RuntimeError(f'Inspect interrupted benchmark: {folder}; do not overwrite')
                if charged_hours()+8*budget*.03/3600+.02>=48:raise RuntimeError('Aggregate GPU budget limit')
                cmd=[sys.executable,'scripts/'+config['script'],'--outdir',str(folder),'--pocket_ids',str(ids),'--seed','2026','--batch_size','4','--warmup',*config['args']]
                if a.trace_control and config['script']!='sample_fk_particles.py':cmd+=['--no_full_trajectory']
                rec=dict(label=label,trace_control=a.trace_control,method=config['name'],budget=budget,trial=trial,status='running',started=time.time(),outdir=str(folder),command=cmd);ledger.append(rec);ledger_path.write_text(json.dumps(ledger,indent=2))
                with (root/(label+'.sample.log')).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
                rec.update(status='sampled' if r.returncode==0 else 'failed',wall_seconds=time.time()-rec['started'],returncode=r.returncode)
                ledger_path.write_text(json.dumps(ledger,indent=2))
                if r.returncode:raise SystemExit(r.returncode)
                ns=budget//100 if config['name']=='geometry_confidence' or config['name'].startswith('fk_') else 1
                tick=time.perf_counter()
                with (root/(label+'.geometry.log')).open('w') as f:
                    subprocess.run([str(ENV/'reed_molecule/bin/python'),'scripts/evaluate_test_time_scaling.py','--run_dir',str(folder),'--best_of',str(ns)],stdout=f,stderr=subprocess.STDOUT,check=True)
                rec['geometry_pipeline_wall_seconds']=time.perf_counter()-tick
                if config['name']=='geometry_confidence':
                    tick=time.perf_counter()
                    with (root/(label+'.vina.log')).open('w') as f:
                        subprocess.run([str(ENV/'pxm_vina/bin/python'),'evaluate/evaluate_vina_sdf.py','--sdf_dir',str(folder/'SDF'),'--gen_info',str(folder/'gen_info.csv'),'--protein_dir','data/csd/files/proteins','--out_csv',str(folder/'vina.csv'),'--mode','score_only','--n_workers','2','--quiet'],stdout=f,stderr=subprocess.STDOUT,check=True)
                    rec['vina_pipeline_wall_seconds']=time.perf_counter()-tick
                rec['status']='done';ledger_path.write_text(json.dumps(ledger,indent=2))
                (folder/'BENCHMARK_DONE').touch()
    (root/('BASELINE_DONE' if a.baseline_only else 'DONE')).touch()

if __name__=='__main__':main()
