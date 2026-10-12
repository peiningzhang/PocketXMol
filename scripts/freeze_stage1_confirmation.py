"""Freeze <=2 configurations using only the 20-pocket/3-seed development set.

Predefined choice: one per family; prioritize validity/complete guardrails,
then maximize the smaller co-primary delta at B=400, then joint, then physical.
Configuration selection never reads confirmation-pocket outcomes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_stage1 import load_candidates,ci
from scripts.run_stage1_adaptations import tasks

ROOT=Path('outputs_test_time_scaling/stage_1')


def chosen(folder,label,ids):
    if not (folder/'EVALUATED').exists():raise RuntimeError(f'Await evaluation: {folder}')
    pool=load_candidates(folder)
    keys=pd.read_csv(folder/'tts_eval'/f'{label}.csv')[['data_id','filename']]
    result=keys.merge(pool,on=['data_id','filename'],validate='one_to_one')
    result=result[result.data_id.isin(ids)]
    if set(result.data_id)!=set(ids) or len(result)!=len(ids):raise ValueError('Development selection coverage mismatch')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check_only',action='store_true');a=p.parse_args()
    protocol=json.loads((ROOT/'protocol.json').read_text());ids=protocol['development_ids'];seeds=protocol['seeds'][:3]
    development=ROOT/'development';ledger=json.loads((development/'adaptation_ledger.json').read_text())
    authoritative={r['label']:Path(r['outdir']) for r in ledger if r['status']=='done'}
    repairs=ROOT/'development_conditioning_v2/partial_conditioning_repair_ledger.json'
    if repairs.exists():
        authoritative.update({r['label']:Path(r['outdir']) for r in json.loads(repairs.read_text()) if r['status']=='done'})
    for task in tasks():
        if task['label'] not in authoritative or not (authoritative[task['label']]/'EVALUATED').exists():
            raise RuntimeError(f'Await complete frozen development matrix: {task["label"]}')
        if task['method']=='partial' and json.loads((authoritative[task['label']]/'run.json').read_text()).get('partial_conditioning_version',1)!=2:
            raise RuntimeError(f'Await corrected teacher fixed-distance conditioning: {task["label"]}')
    baseline_ledger=json.loads((ROOT/'generation_ledger.json').read_text())
    baseline_paths={r['label']:Path(r['outdir']) for r in baseline_ledger if r['status']=='done'}
    base=[]
    for seed in seeds:
        folder=baseline_paths.get(f'pool_t100_n4_r0_s{seed}')
        if folder is None:raise RuntimeError(f'Await B400 baseline seed {seed}')
        f=chosen(folder,'best_of_4_geometry_confidence',ids);f['seed']=seed;base.append(f)
    base=pd.concat(base,ignore_index=True)
    configs={}
    for task in tasks():
        if task['budget']!=400:continue
        if task['method']=='fk':
            key=f"fk_i{task['interval']}_b{task['strength']:g}";label='best_of_4_geometry_confidence'
        else:key=f"partial_{task['selection']}";label='best_of_1_confidence'
        f=chosen(authoritative[task['label']],label,ids);f['seed']=task['seed']
        configs.setdefault(key,dict(task=task,frames=[]))['frames'].append(f)
    scored=[];rng=np.random.default_rng(2026)
    for name,config in configs.items():
        f=pd.concat(config['frames'],ignore_index=True)
        if f.duplicated(['data_id','seed']).any() or len(f)!=len(ids)*len(seeds):raise ValueError('Development pairs incomplete')
        match=f.merge(base,on=['data_id','seed'],suffixes=('','_base'),validate='one_to_one')
        score=dict(name=name,family=config['task']['method'],task=config['task'],pairs=len(match),pockets=len(ids))
        for metric in ['P_phys','P_joint','valid','complete']:
            diff=match[metric].astype(float)-match[metric+'_base'].astype(float)
            per_pocket=diff.groupby(match.data_id).mean();lo,hi=ci(per_pocket,rng,5000)
            score[metric+'_delta']=float(per_pocket.mean());score[metric+'_ci']=[float(lo),float(hi)]
            score[metric+'_seed_deltas']={str(seed):float(diff[match.seed==seed].mean()) for seed in seeds}
        score['guardrails_ok']=score['valid_delta']>=-.03 and score['complete_delta']>=-.03
        score['min_primary_delta']=min(score['P_phys_delta'],score['P_joint_delta']);scored.append(score)
    winners=[]
    for family in ['fk','partial']:
        family_scores=[s for s in scored if s['family']==family]
        winner=max(family_scores,key=lambda s:(s['guardrails_ok'],s['min_primary_delta'],s['P_joint_delta'],s['P_phys_delta'],s['name']))
        winners.append(winner)
    print(json.dumps({'selected':winners,'all_development_scores':scored},indent=2))
    if a.check_only:return
    root=ROOT/'confirmation';root.mkdir(exist_ok=True);dest=root/'confirmation_manifest.json'
    if dest.exists():raise ValueError('Confirmation already frozen; do not overwrite')
    source_files=['models/sample.py','scripts/stage1_runtime.py','scripts/fk_particles.py','scripts/partial_refinement.py','scripts/sample_fk_particles.py','scripts/sample_partial_refinement.py']
    result=dict(version=1,pocket_ids=protocol['confirmation_ids'],seeds=protocol['seeds'],budgets=[400,800],
                batch_size=4,selected=winners,all_development_scores=scored,
                selection_rule='one per family; guardrails, max minimum co-primary delta at B400, joint, physical, deterministic name tie-break',
                selection_uses_confirmation_data=False,bootstrap_resamples=5000,
                code_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in source_files},
                partial_noise_start=1.0,partial_fraction=.25,partial_rounds_by_budget={'400':3,'800':7},
                fk_particles_by_budget={'400':4,'800':8},
                negative_development_policy='Complete full predefined confirmation for both chosen families; no further reward or selector tuning')
    result['frozen_development_extensions'] = {
        'pocket_ids': protocol['development_ids'],
        'purpose': 'Complete selected-method full100/five-seed summaries after configuration freeze; never use extension outcomes to retune',
        'seeds': protocol['seeds'], 'budgets': [400,800],
        'reuse': 'Reuse exactly matching completed development tasks; only generate missing method/budget/seed cells',
    }
    dest.write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
