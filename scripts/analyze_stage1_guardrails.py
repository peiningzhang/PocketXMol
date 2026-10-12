"""Stratified size/coverage/diversity checks and FK ancestor diagnostics."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs_test_time_scaling/stage_1'));a=p.parse_args()
    root=a.root;tables=root/'tables';f=pd.read_csv(tables/'per_pocket_seed_selections.csv')
    for name in ['P_phys','P_joint','valid','complete']:
        f[name]=f[name].astype(str).str.lower().eq('true')
    # Boundaries fixed independently of observed method gains.
    f['size_bin']=pd.cut(pd.to_numeric(f.n_atoms,errors='coerce'),[-np.inf,20,30,40,50,np.inf],labels=['<=20','21-30','31-40','41-50','>50']).astype(object).fillna('unknown_reconstruction')
    rows=[]
    for key,g in f.groupby(['subset','method','budget','size_bin'],dropna=False):
        row=dict(zip(['subset','method','budget','size_bin'],key));row['outputs']=len(g);row['pockets']=g.data_id.nunique()
        for col in ['P_phys','P_joint','valid','complete']:row[col]=g[col].mean()
        row['energy_coverage']=pd.to_numeric(g.energy_ratio,errors='coerce').notna().mean()
        row['vina_coverage']=pd.to_numeric(g.vina_score,errors='coerce').notna().mean()
        row['ligand_efficiency_mean']=pd.to_numeric(g.ligand_efficiency,errors='coerce').mean()
        rows.append(row)
    pd.DataFrame(rows).to_csv(tables/'size_stratified_quality.csv',index=False)
    particle=[]
    for path in sorted(root.glob('development/*/particle_diagnostics.json'))+sorted(root.glob('confirmation/*/particle_diagnostics.json'))+sorted(root.glob('development_extensions/*/particle_diagnostics.json')):
        if not (path.parent/'DONE').exists():continue
        run=json.loads((path.parent/'run.json').read_text())
        for event in json.loads(path.read_text()):
            row=dict(run=path.parent.name,seed=run['seed'],particles=run['particles'],interval=run['interval'],strength=run['strength'],**event)
            row['ancestor_fraction']=event['original_ancestor_count']/run['particles']
            row['ESS_fraction']=event['ESS']/run['particles'];particle.append(row)
    pd.DataFrame(particle).drop(columns=['reward','weights','parents'],errors='ignore').to_csv(tables/'fk_ancestor_ess.csv',index=False)
    if particle:
        pd.DataFrame(particle).groupby(['particles','interval','strength','call'])[['ancestor_fraction','ESS_fraction','unique_parents']].agg(['mean','min','max']).to_csv(tables/'fk_ancestor_ess_summary.csv')
    print('Wrote stratified size/coverage and FK ancestry/ESS tables')

if __name__=='__main__':main()
