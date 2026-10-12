"""Full confirmation quality vs same-hardware trace-off prototype cost.

Quality uses80 pockets/five seeds; throughput uses8 fixed pockets/three trials.
Timing x-error is trial SD, quality y-error is pocket-cluster95CI. Native early
stopping can differ by pocket subset; actual NFE for each scope is retained.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('outputs_test_time_scaling/stage_1')
def main():
    manifest=json.loads((ROOT/'confirmation/confirmation_manifest.json').read_text())
    quality=pd.read_csv(ROOT/'tables/budget_curves.csv')
    quality=quality[quality.subset=='confirmation']
    timing=pd.read_csv(ROOT/'tables/trace_control_repeated_timing_trials.csv')
    methods=['geometry_confidence','vina_oracle','more_steps','native_refinement']+[s['name'] for s in manifest['selected']]
    rows=[]
    for method in methods:
        for budget in [400,800]:
            q=quality[(quality.method==method)&(quality.budget==budget)]
            t=timing[(timing.method==method)&(timing.budget==budget)]
            if len(q)!=1 or not bool(q.iloc[0].complete_matrix) or len(t)!=3:
                raise RuntimeError(f'Await full confirmation quality and three trace-off timing trials: {method}/{budget}')
            assert t[['hostname','cuda_visible_devices','gpu_name']].drop_duplicates().shape[0]==1
            row=q.iloc[0].to_dict();row.update(controlled_search_seconds=t.search_seconds_per_output.mean(),controlled_search_seconds_trial_sd=t.search_seconds_per_output.std(),benchmark_actual_nfe=t.actual_graph_nfe_per_output.mean(),timing_trials=len(t));rows.append(row)
    f=pd.DataFrame(rows);f.to_csv(ROOT/'tables/controlled_cost_quality_points.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    for ax,metric in zip(axes,['P_phys','P_joint']):
        for method,g in f.groupby('method',sort=False):
            g=g.sort_values('budget')
            ax.errorbar(g.controlled_search_seconds,100*g[metric],xerr=g.controlled_search_seconds_trial_sd,
                yerr=[100*(g[metric]-g[metric+'_ci_low']),100*(g[metric+'_ci_high']-g[metric])],marker='o',capsize=3,label=method)
        ax.set(xlabel='Trace-off prototype search seconds per output (including selector verifiers)',ylabel=metric+' (%)',ylim=(0,100));ax.grid(alpha=.2)
    axes[1].legend(bbox_to_anchor=(1.03,1),loc='upper left',fontsize=8)
    provisional=json.loads((ROOT/'tables/analysis.json').read_text())['provisional']
    fig.suptitle(('PROVISIONAL: ' if provisional else '')+'Confirmation quality (80 pockets, 5 seeds); prototype timing (8 pockets, 3 trials)')
    fig.text(.5,.01,'Quality bars: pocket-cluster95CI. Cost bars: trial SD. Both scopes retain their actual NFE in the source table.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.05,1,1]);out=ROOT/'figures';out.mkdir(exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(out/('confirmation_controlled_time_curves.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)
if __name__=='__main__':main()
