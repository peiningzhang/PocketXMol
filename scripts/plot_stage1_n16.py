"""Export bounded extension curves only after complete dev20/three-seed analysis."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
root=Path('outputs_test_time_scaling/stage_1/n16_extension')
path=root/'tables/analysis.json'
if not path.exists() or json.loads(path.read_text()).get('complete') is not True:
    raise SystemExit('Await complete bounded extension analysis')
f=pd.read_csv(root/'tables/budget_curves.csv');out=root/'figures';out.mkdir(exist_ok=True)
for x,name in [('actual_nfe_mean','actual_nfe'),('search_seconds_mean','logical_search_time')]:
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,metric in zip(axes,['P_phys','P_joint']):
        for method,g in f.groupby('method'):
            g=g.sort_values('budget')
            ax.errorbar(g[x],g[metric],yerr=[g[metric]-g[metric+'_ci_low'],g[metric+'_ci_high']-g[metric]],marker='o',capsize=3,label=method)
        ax.set(xlabel='Actual graph-NFE per output (mean)' if x=='actual_nfe_mean' else 'Logical amortized search seconds per output',ylabel=metric,ylim=(0,1))
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8,loc='best')
    fig.suptitle('Bounded exploratory extension: development20, three seeds; B800 / B1600')
    fig.text(.5,.01,'95% pocket-bootstrap CIs after seed averaging. Time reuses measured prefix costs; not a common-GPU repeated latency benchmark.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.05,1,.94])
    for suffix in ['png','pdf']:fig.savefig(out/(name+'_curves.'+suffix),dpi=180)
    plt.close(fig)
print('Exported bounded N16 NFE/time curves')
