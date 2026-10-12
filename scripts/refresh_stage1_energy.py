"""Migrate early energy evaluations to recorded single-thread protocol.

Preserve previous measurements. Never change candidates or scoring thresholds.
Only completed EVALUATED runs are touched, so active evaluator outputs are safe.
"""
import json
from pathlib import Path
import shutil
import subprocess

root=Path('outputs_test_time_scaling/stage_1')
python='/shared/healthinfolab/phz24002/anaconda3/envs/reed_molecule/bin/python'
for marker in sorted(root.rglob('EVALUATED')):
    folder=marker.parent;out=folder/'tts_eval';protocol=json.loads((out/'energy_protocol.json').read_text())
    log=folder/'energy.log'
    bad_thread_wrapper=log.exists() and 'multiple values for argument' in log.read_text()
    if protocol.get('ensemble_threads')==1 and not bad_thread_wrapper:continue
    archive=out/'energy_previous_protocol';archive.mkdir(exist_ok=False)
    for name in ['energy_candidates.csv','energy_summary.csv','energy_protocol.json']:
        shutil.copy2(out/name,archive/name)
    labels=[p.stem for p in sorted(out.glob('best_of_*.csv'))]
    with (folder/'energy.single_thread.log').open('w') as stream:
        subprocess.run([python,'scripts/evaluate_tts_energy.py','--run_dir',str(folder),'--all_candidates','--selections',*labels],stdout=stream,stderr=subprocess.STDOUT,check=True)
    (archive/'migration.json').write_text(json.dumps({'reason':'Bound CPU threads; fixed RNG seed 42, ensemble50, threshold100 unchanged','previous_wrapper_fault':bad_thread_wrapper,'candidate_regeneration':False},indent=2))
    print('Refreshed',folder,flush=True)
