"""Evaluate predeclared Stage2 quantitative gates after complete statistics.

Size, diversity and mechanism interpretation still require explicit report review.
"""
import json
from pathlib import Path
import pandas as pd

root=Path('outputs_test_time_scaling/stage_1');tables=root/'tables'
analysis=json.loads((tables/'analysis.json').read_text())
if analysis.get('provisional') is not False:
    raise SystemExit('Await complete full100 and confirmation statistics')
manifest=json.loads((root/'confirmation/confirmation_manifest.json').read_text())
paired=pd.read_csv(tables/'paired_comparisons.csv')
timing=pd.read_csv(tables/'trace_control_repeated_timing_comparisons.csv')
results=[]
for config in manifest['selected']:
    for budget in manifest['budgets']:
        f=paired[(paired.subset=='confirmation')&(paired.method==config['name'])&(paired.budget==budget)].set_index('metric')
        required=['P_phys','P_joint','valid','complete']
        if any(metric not in f.index for metric in required):raise ValueError('Missing paired gate metrics')
        if not (f.pairs==400).all() or not (f.pockets==80).all():raise ValueError('Confirmation grid incomplete')
        matched=bool((f.exact_nfe_pair_fraction==1).all())
        guard=bool((f.loc[['valid','complete'],'delta']>=-.03).all())
        primary=f.loc[['P_phys','P_joint']]
        quality_gain=bool(((primary.delta>=.05)&(primary.ci_low>0)).any() and (primary.delta>=-.02).all())
        similar=bool((primary.delta.abs()<=.02).all())
        t=timing[(timing.method==config['name'])&(timing.budget==budget)]
        if len(t)!=1:raise ValueError('Missing repeated controlled timing')
        t=t.iloc[0]
        repeated=bool(t.paired_trials==3 and str(t.same_hardware).lower()=='true' and
                      t.cost_ratio<=.8 and t.trial_cost_ratio_max<1)
        results.append(dict(method=config['name'],budget=int(budget),actual_nfe_all_pairs_matched=matched,
            validity_complete_guardrails=guard,criterion1_quality=matched and guard and quality_gain,
            criterion2_cost=matched and guard and similar and repeated,
            co_primary=primary[['delta','ci_low','ci_high']].to_dict('index'),
            controlled_cost_ratio=float(t.cost_ratio),controlled_trial_ratio_max=float(t.trial_cost_ratio_max),
            size_coverage_diversity_review='required',criterion3_mechanism='requires reproducible evidence and effective control; report review'))
(tables/'stage2_quantitative_gates.json').write_text(json.dumps(dict(results=results,
    decision_scope='Quantitative gates only; no automatic Stage2 authorization without size/coverage/diversity and mechanism review'),indent=2))
print(json.dumps(results,indent=2))
