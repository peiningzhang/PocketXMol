"""Supplemental native AR comparison at identical observed per-output graph NFE.

Pool prefix size is derived from native observed NFE. This retrospective control
is diagnostic, not a deployable budget policy or a replacement for main curves.
"""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_stage1 import load_candidates,ci
from scripts.evaluate_test_time_scaling import select_candidates
ROOT=Path('outputs_test_time_scaling/stage_1');protocol=json.loads((ROOT/'protocol.json').read_text())
ids=set(protocol['pocket_ids']);dev=set(protocol['development_ids']);records=[]
for seed in protocol['seeds']:
    pool=ROOT/f'pool_t100_n8_r0_s{seed}'
    if not (pool/'EVALUATED').exists():raise SystemExit('Await complete independent pools')
    f=load_candidates(pool)
    for budget,rounds in [(200,1),(400,3),(800,7)]:
        native=ROOT/f'refine_t100_n1_r{rounds}_s{seed}'
        if not (native/'EVALUATED').exists():raise SystemExit('Await complete native controls')
        a=load_candidates(native)
        if len(a)!=100 or set(a.data_id)!=ids:raise ValueError('Native planned grid incomplete')
        counts=pd.to_numeric(a.nfe,errors='raise')/100
        if not counts.between(1,8).all() or not (counts==counts.round()).all():raise ValueError('NFE cannot be exactly matched by existing 100-step prefixes')
        a=a.copy();a['matched_n']=counts.astype(int)
        for n,g in a.groupby('matched_n'):
            searched=f[f.data_id.isin(g.data_id)&(f.i_repeat<int(n))]
            reference=select_candidates(searched,int(n),'geometry_confidence')
            total=searched.groupby('data_id').nfe.sum()
            reference=reference.copy();reference['actual_nfe']=reference.data_id.map(total)
            pair=g.merge(reference,on='data_id',suffixes=('_native','_pool'),validate='one_to_one')
            if not (pair.nfe_native==pair.actual_nfe).all():raise ValueError('Actual NFE mismatch')
            for row in pair.to_dict('records'):
                rec=dict(data_id=row['data_id'],seed=seed,nominal_native_budget=budget,matched_pool_n=int(n),
                         actual_graph_nfe=float(row['actual_nfe']),native_source=str(native),pool_source=str(pool),
                         native_filename=row['filename_native'],pool_filename=row['filename_pool'])
                for metric in ['P_phys','P_joint','valid','complete','n_atoms','qed','vina_score','energy_ratio','ligand_efficiency']:
                    rec[metric+'_native']=row[metric+'_native'];rec[metric+'_pool']=row[metric+'_pool']
                records.append(rec)
f=pd.DataFrame(records)
if len(f)!=1500 or f.duplicated(['data_id','seed','nominal_native_budget']).any():raise ValueError('Matched grid incomplete')
rows=[]
for subset in ['development','confirmation','all']:
    sf=f if subset=='all' else f[f.data_id.isin(dev if subset=='development' else ids-dev)]
    for budget,g in sf.groupby('nominal_native_budget'):
        rng=np.random.default_rng(2026)
        for metric in ['P_phys','P_joint','valid','complete']:
            d=(g[metric+'_native'].astype(float)-g[metric+'_pool'].astype(float)).groupby(g.data_id).mean();lo,hi=ci(d,rng,5000)
            rows.append(dict(subset=subset,nominal_native_budget=int(budget),metric=metric,
                             native_minus_matched_pool=d.mean(),ci_low=lo,ci_high=hi,pockets=len(d),pairs=len(g),
                             actual_nfe_mean=g.actual_graph_nfe.mean(),exact_nfe_pair_fraction=1.0,
                             native_rate=g.groupby('data_id')[metric+'_native'].mean().mean(),
                             matched_pool_rate=g.groupby('data_id')[metric+'_pool'].mean().mean()))
out=ROOT/'tables';f.to_csv(out/'native_actual_nfe_matched_selections.csv',index=False);pd.DataFrame(rows).to_csv(out/'native_actual_nfe_matched_comparisons.csv',index=False)
(out/'native_actual_nfe_matched_protocol.json').write_text(json.dumps(dict(complete=True,pairs=1500,
    bootstrap_resamples=5000,cluster='pocket after seed average',all_planned_native_outputs_retained=True,
    selector='Frozen geometry-confidence ranking; no energy or Vina used',
    prefix_rule='N=observed native graph-NFE/100, including diagnostic N3/5/6/7 prefixes; same raw N8 pool',
    scope='Retrospective resource-matched diagnostic. Requires native observed NFE to choose reference prefix; not a deployable standalone allocation policy, not a replacement for predeclared N1/2/4/8 main curves.'),indent=2))
print('Verified exact-NFE pairs:',len(f))
