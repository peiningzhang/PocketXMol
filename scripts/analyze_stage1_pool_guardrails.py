"""Frozen selected FK vs independent pools: paired diversity/coverage guardrails.

Uses complete geometry candidate tables, not energy availability or selection.
Undefined diversity stays missing; planned counts/valid coverage remain explicit.
"""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem,RDLogger
from rdkit.Chem import AllChem,DataStructs
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_stage1 import ci,boolean
ROOT=Path('outputs_test_time_scaling/stage_1')
def describe(g):
    valid=g[boolean(g.valid)];smiles=valid.smiles.dropna()
    mols=[Chem.MolFromSmiles(s) for s in smiles]
    fps=[AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048) for m in mols if m]
    distances=[1-DataStructs.TanimotoSimilarity(fps[i],fps[j]) for i in range(len(fps)) for j in range(i)]
    return dict(planned_candidates=len(g),valid_candidates=len(valid),valid_candidate_coverage=len(valid)/len(g),
        duplicate_fraction=1-smiles.nunique()/len(smiles) if len(smiles) else np.nan,
        fingerprint_diversity=np.mean(distances) if distances else np.nan,
        mean_candidate_heavy_atoms=pd.to_numeric(g.n_atoms,errors='coerce').mean(),
        atom_count_coverage=pd.to_numeric(g.n_atoms,errors='coerce').notna().mean())
def main():
    RDLogger.DisableLog('rdApp.*')
    protocol=json.loads((ROOT/'protocol.json').read_text());manifest=json.loads((ROOT/'confirmation/confirmation_manifest.json').read_text())
    selected=[c for c in manifest['selected'] if c['family']=='fk']
    if len(selected)!=1:raise ValueError('Expected one frozen FK configuration')
    name=selected[0]['name'];rows=[]
    for budget in manifest['budgets']:
        n=budget//100
        for seed in protocol['seeds']:
            independent=ROOT/f'pool_t100_n8_r0_s{seed}'
            fk=ROOT/'confirmation'/f'{name}_b{budget}_s{seed}'
            for folder,kind in [(independent,'independent'),(fk,'FK')]:
                f=pd.read_csv(folder/'tts_eval/candidate_metrics.csv')
                if kind=='independent':f=f[f.i_repeat<n]
                f=f[f.data_id.isin(protocol['confirmation_ids'])]
                if len(f)!=80*n or f.groupby('data_id').size().ne(n).any():raise ValueError('Planned candidate grid incomplete')
                for pid,g in f.groupby('data_id'):
                    rows.append(dict(data_id=pid,seed=seed,budget=budget,pool_kind=kind,source=str(folder),**describe(g)))
    f=pd.DataFrame(rows);out=ROOT/'tables';f.to_csv(out/'selected_pool_guardrails_per_pocket.csv',index=False)
    comparisons=[]
    for budget,g in f.groupby('budget'):
        pair=g[g.pool_kind=='FK'].merge(g[g.pool_kind=='independent'],on=['data_id','seed','budget'],suffixes=('_fk','_base'),validate='one_to_one')
        if len(pair)!=400:raise ValueError('Paired grid differs')
        for metric in ['valid_candidate_coverage','duplicate_fraction','fingerprint_diversity','mean_candidate_heavy_atoms','atom_count_coverage']:
            defined=pair[metric+'_fk'].notna()&pair[metric+'_base'].notna()
            common=pair.loc[defined]
            d=(common[metric+'_fk']-common[metric+'_base']).groupby(common.data_id).mean().dropna()
            lo,hi=ci(d,np.random.default_rng(2026),5000)
            comparisons.append(dict(budget=int(budget),metric=metric,FK_mean=common.groupby('data_id')[metric+'_fk'].mean().mean(),independent_mean=common.groupby('data_id')[metric+'_base'].mean().mean(),delta=d.mean(),ci_low=lo,ci_high=hi,defined_pairs=int(defined.sum()),planned_pairs=400,defined_pockets=len(d)))
    pd.DataFrame(comparisons).to_csv(out/'selected_pool_guardrails_comparisons.csv',index=False)
    (out/'selected_pool_guardrails_protocol.json').write_text(json.dumps(dict(complete=True,scope='80 confirmation pockets, five seeds; frozen FK and exact independent prefixes P4/P8; all planned candidates retained',bootstrap_resamples=5000,cluster='pocket after seed average',conditional_metric_note='Duplicate/fingerprint values condition on chemically valid candidates. Undefined values remain NaN; defined pairs/pockets reported separately. Energy availability is not used.'),indent=2))
    print(pd.DataFrame(comparisons).to_string(index=False))
if __name__=='__main__':main()
