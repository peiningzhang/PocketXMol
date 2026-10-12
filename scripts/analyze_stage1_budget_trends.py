"""Paired adjacent-budget trends; record whether N16 warrants a bounded extension."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_stage1 import ci

root=Path('outputs_test_time_scaling/stage_1');tables=root/'tables'
f=pd.read_csv(tables/'per_pocket_seed_selections.csv')
for c in ['P_phys','P_joint']:f[c]=f[c].astype(str).str.lower().eq('true')
rng=np.random.default_rng(2026);rows=[]
for subset in ['development','confirmation','all']:
    sf=f if subset=='all' else f[f.subset==subset]
    for method,g in sf.groupby('method'):
        for low,high in [(100,200),(200,400),(400,800)]:
            pair=g[g.budget==high].merge(g[g.budget==low],on=['data_id','seed'],suffixes=('_high','_low'),validate='one_to_one')
            if not len(pair):continue
            for metric in ['P_phys','P_joint']:
                diff=pair[metric+'_high'].astype(float)-pair[metric+'_low'].astype(float)
                d=diff.groupby(pair.data_id).mean();lo,hi=ci(d,rng,5000)
                rows.append(dict(subset=subset,method=method,low_budget=low,high_budget=high,metric=metric,delta=d.mean(),ci_low=lo,ci_high=hi,pockets=len(d),pairs=len(pair)))
frame=pd.DataFrame(rows);frame.to_csv(tables/'paired_budget_differences.csv',index=False)
analysis=json.loads((tables/'analysis.json').read_text())
complete=analysis['baseline_matrix_complete']
strong=frame[(frame.subset=='all')&(frame.method=='geometry_confidence')&(frame.low_budget==400)&(frame.high_budget==800)] if len(frame) else frame
clear=False
if complete and len(strong)==2:
    # Clear upward evidence in at least one co-primary; neither declines >2pp.
    clear=bool(((strong.delta>0)&(strong.ci_low>0)).any() and (strong.delta>=-.02).all())
decision=dict(baseline_complete=complete,decision='await_full_core_matrix' if not complete else 'bounded_N16_extension_warranted' if clear else 'no_clear_B800_upward_trend_no_N16_expansion',
    operational_rule='Geometry-confidence deployed baseline: positive paired95CI lower bound in at least one co-primary, neither decreases >2pp; all100/five seeds must be complete',
    extension_if_triggered={'pocket_ids':'development_ids.txt','seeds':[2026,12026,22026],'budgets':[800,1600],'controls':['independent100-step pool N16','more_steps1600','native_refinement extra15 rounds'],'do_not_extend_rewards':True},
    evidence=strong.to_dict('records'))
(tables/'n16_decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2))
