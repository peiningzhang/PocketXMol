"""Export standalone frozen-protocol NFE/time and component figures."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs_test_time_scaling/stage_1'));a=p.parse_args()
    root=a.root; out=root/'figures';out.mkdir(exist_ok=True)
    global_provisional=json.loads((root/'tables/analysis.json').read_text())['provisional']
    curves=pd.read_csv(root/'tables/budget_curves.csv')
    for subset in curves.subset.unique():
        f=curves[curves.subset==subset]
        for cost,label,filename in [('actual_graph_nfe_mean','Actual graph-NFE per output','nfe'),('search_seconds_per_output_mean','Search seconds per output (incl. selector verifiers)','time')]:
            fig,axes=plt.subplots(1,2,figsize=(13,4.5))
            for ax,metric in zip(axes,['P_phys','P_joint']):
                for method,g in f.groupby('method'):
                    g=g.sort_values(cost)
                    ax.errorbar(g[cost],100*g[metric],yerr=[100*(g[metric]-g[metric+'_ci_low']),100*(g[metric+'_ci_high']-g[metric])],marker='o',capsize=3,label=method)
                ax.set(xlabel=label,ylabel=f'{metric} (%)',ylim=(0,100));ax.grid(alpha=.2)
            provisional=global_provisional or not f.complete_matrix.all()
            fig.suptitle(f'{subset}: '+('PROVISIONAL, incomplete planned matrix' if provisional else 'frozen Stage 1 matrix'))
            axes[1].legend(bbox_to_anchor=(1.03,1),loc='upper left',fontsize=8)
            fig.tight_layout();fig.savefig(out/f'{subset}_{filename}_curves.png',dpi=180,bbox_inches='tight');fig.savefig(out/f'{subset}_{filename}_curves.pdf',bbox_inches='tight');plt.close(fig)
    components=pd.read_csv(root/'tables/component_failures.csv')
    for subset,g in components.groupby('subset'):
        g=g.copy();g['label']=g.method+' / '+g.budget.astype(str)
        pivot=g.pivot(index='label',columns='component',values='pass_all')
        fig,ax=plt.subplots(figsize=(10,max(3,len(pivot)*.3)))
        im=ax.imshow(pivot.values,vmin=0,vmax=1,cmap='viridis',aspect='auto');ax.set_yticks(range(len(pivot)),pivot.index);ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=35,ha='right');fig.colorbar(im,label='Pass fraction, all planned outputs represented in completed runs');ax.set_title(subset+(': PROVISIONAL, incomplete planned matrix' if global_provisional else ''))
        fig.tight_layout();fig.savefig(out/f'{subset}_component_pass.png',dpi=180,bbox_inches='tight');fig.savefig(out/f'{subset}_component_pass.pdf',bbox_inches='tight');plt.close(fig)

if __name__=='__main__':main()
