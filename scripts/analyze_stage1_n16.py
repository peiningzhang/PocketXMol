"""Bounded dev20 N8/N16 versus both step/refinement controls; pocket bootstrap."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_stage1 import load_candidates,ci
from scripts.stage1_timing import generation_costs
ROOT=Path('outputs_test_time_scaling/stage_1');root=ROOT/'n16_extension'
decision=json.loads((ROOT/'tables/n16_decision.json').read_text())
if decision['decision']!='bounded_N16_extension_warranted':raise SystemExit('Await N16 trend gate')
if not (root/'EVALUATION_COMPLETE').exists():raise SystemExit('Await complete N16 independent evaluation')
manifest=json.loads((root/'extension_manifest.json').read_text());ids=set(manifest['pocket_ids'])
ledger=json.loads((root/'n16_generation_ledger.json').read_text());records=[]
for seed in manifest['seeds']:
    for budget in [800,1600]:
        pool=ROOT/f'pool_t100_n8_r0_s{seed}' if budget==800 else root/f'pool_t100_n16_r0_s{seed}'
        sources=[(rank,pool,budget//100) for rank in ['random','confidence','geometry_confidence','vina_oracle']]
        for kind,method in [('steps','more_steps'),('refine','native_refinement')]:
            if budget==800:folder=ROOT/(f'steps_t800_n1_r0_s{seed}' if kind=='steps' else f'refine_t100_n1_r7_s{seed}')
            else:
                matches=[r for r in ledger if r['kind']==kind and r['seed']==seed and r['status']=='done']
                if len(matches)!=1:raise ValueError('Ambiguous extension control')
                folder=Path(matches[0]['outdir'])
            sources.append((method,folder,1))
        for method,folder,n in sources:
            if not (folder/'EVALUATED').exists():raise ValueError('Baseline control evaluation incomplete')
            f=load_candidates(folder);f=f[f.data_id.isin(ids)].copy();run=json.loads((folder/'run.json').read_text())
            rank=method if method in ['random','confidence','geometry_confidence','vina_oracle'] else 'confidence'
            name=f'best_of_{n}_{rank}';chosen=pd.read_csv(folder/'tts_eval'/(name+'.csv'))
            denominator=len(chosen);chosen=chosen[chosen.data_id.isin(ids)]
            g=chosen[['data_id','filename']].merge(f,on=['data_id','filename'],validate='one_to_one')
            if len(g)!=20 or set(g.data_id)!=ids:raise ValueError('Planned outputs missing')
            searched=f[f.i_repeat<n];calls=searched.groupby('data_id').nfe.sum();costs=generation_costs(folder)
            generation={pid:sum(costs[(r.data_id,int(r.i_repeat))]['total_generation_seconds'] for r in sf.itertuples()) for pid,sf in searched.groupby('data_id')}
            summary=pd.read_csv(folder/'tts_eval/summary.csv').set_index('method').loc[name]
            verifier=searched.physical_score_seconds.sum()/20 if rank in ['geometry_confidence','vina_oracle'] else 0.
            if rank=='vina_oracle':verifier+=float((folder/'vina_wall_seconds.txt').read_text())*n/run['repeats']/run['num_pockets']
            g['actual_graph_nfe']=g.data_id.map(calls);g['search_seconds']=g.data_id.map(generation)+float(summary.selection_seconds)/denominator+verifier
            g['method']=method;g['budget']=budget;g['seed']=seed;g['source']=str(folder);records.append(g)
f=pd.concat(records,ignore_index=True)
if f.duplicated(['method','budget','seed','data_id']).any():raise ValueError('Duplicate paired unit')
metrics=['P_phys','P_joint','P_joint_minus6','P_joint_minus8','valid','complete','geometry_pass','energy_pass','pocket_no_clash']
rows=[];paired=[]
for (method,budget),g in f.groupby(['method','budget']):
    if len(g)!=60:raise ValueError('Incomplete dev20/three-seed cell')
    row=dict(method=method,budget=int(budget),pockets=20,outputs=60,actual_nfe_mean=float(g.actual_graph_nfe.mean()),search_seconds_mean=float(g.search_seconds.mean()))
    rng=np.random.default_rng(2026)
    for metric in metrics:
        values=g.groupby('data_id')[metric].mean();lo,hi=ci(values,rng,5000);row.update({metric:values.mean(),metric+'_ci_low':lo,metric+'_ci_high':hi})
    for metric in ['n_atoms','qed','energy_ratio','vina_score','ligand_efficiency']:
        v=pd.to_numeric(g[metric],errors='coerce');row[metric+'_mean']=v.mean();row[metric+'_median']=v.median();row[metric+'_coverage']=v.notna().mean()
    rows.append(row)
    base=f[(f.method=='geometry_confidence')&(f.budget==budget)]
    match=g.merge(base,on=['data_id','seed'],suffixes=('','_base'),validate='one_to_one')
    for metric in ['P_phys','P_joint','valid','complete']:
        d=(match[metric].astype(float)-match[metric+'_base'].astype(float)).groupby(match.data_id).mean();lo,hi=ci(d,rng,5000)
        paired.append(dict(method=method,budget=int(budget),reference='geometry_confidence',metric=metric,delta=d.mean(),ci_low=lo,ci_high=hi,pairs=len(match),pockets=20,exact_nfe_pair_fraction=(match.actual_graph_nfe==match.actual_graph_nfe_base).mean()))
out=root/'tables';out.mkdir(exist_ok=True)
f.to_csv(out/'per_pocket_seed_selections.csv',index=False);pd.DataFrame(rows).to_csv(out/'budget_curves.csv',index=False);pd.DataFrame(paired).to_csv(out/'paired_comparisons.csv',index=False)
adjacent=[]
for method,g in f.groupby('method'):
    match=g[g.budget==1600].merge(g[g.budget==800],on=['data_id','seed'],suffixes=('_high','_low'),validate='one_to_one')
    if len(match)!=60:raise ValueError('Adjacent-budget pairing incomplete')
    rng=np.random.default_rng(2026)
    for metric in ['P_phys','P_joint','valid','complete']:
        d=(match[metric+'_high'].astype(float)-match[metric+'_low'].astype(float)).groupby(match.data_id).mean();lo,hi=ci(d,rng,5000)
        adjacent.append(dict(method=method,low_budget=800,high_budget=1600,metric=metric,delta=d.mean(),ci_low=lo,ci_high=hi,pockets=20,pairs=60,actual_nfe_low_mean=match.actual_graph_nfe_low.mean(),actual_nfe_high_mean=match.actual_graph_nfe_high.mean()))
pd.DataFrame(adjacent).to_csv(out/'paired_budget_differences.csv',index=False)
(out/'analysis.json').write_text(json.dumps(dict(complete=True,scope='bounded exploratory development20/three seeds only; no confirmation or full100 claim',bootstrap_resamples=5000,cluster='pocket after seed average',cost_scope='Logical search cost with original prefix amortization; no common-GPU repeated N16 latency benchmark'),indent=2))
print('Complete bounded extension cells',len(rows))
