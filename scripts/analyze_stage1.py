"""Pocket-cluster bootstrap, nested pool oracle, independent energy outcomes."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.stage1_timing import generation_costs

BOOL = ['valid','complete','geometry_pass','pocket_no_clash','qualified','energy_pass']
RDLogger.DisableLog('rdApp.*')

def boolean(series):
    return series.astype(str).str.lower().eq('true')

def load_candidates(folder):
    f = pd.read_csv(folder/'tts_eval/candidate_metrics.csv')
    energy = pd.read_csv(folder/'tts_eval/energy_candidates.csv')
    if set(f.filename)!=set(energy.filename):
        raise ValueError(f'Energy evaluation records incomplete: {folder}')
    energy_protocol=json.loads((folder/'tts_eval/energy_protocol.json').read_text())
    if energy_protocol.get('ensemble_threads')!=1:
        raise ValueError(f'Energy threading protocol not frozen: {folder}')
    module=[m for m in energy_protocol['config']['modules'] if m['function']=='energy_ratio'][0]
    if module['parameters']['threshold_energy_ratio']!=100 or module['parameters']['ensemble_number_conformations']!=50:
        raise ValueError('Energy threshold/ensemble differs from frozen plan')
    f = f.merge(energy, on='filename', how='left', validate='one_to_one')
    for col in BOOL:
        f[col] = boolean(f[col])
    f['P_phys'] = f.qualified & f.energy_pass
    f['P_joint'] = f.P_phys & pd.to_numeric(f.vina_score,errors='coerce').le(-7)
    f['P_joint_minus6'] = f.P_phys & pd.to_numeric(f.vina_score,errors='coerce').le(-6)
    f['P_joint_minus8'] = f.P_phys & pd.to_numeric(f.vina_score,errors='coerce').le(-8)
    f['clashes_per_heavy_atom'] = f.num_pairwise_clashes / f.n_atoms
    return f

def ci(values, rng, resamples=5000):
    v=np.asarray(values,dtype=float)
    if not len(v): return (np.nan,np.nan)
    boot=v[rng.integers(0,len(v),size=(resamples,len(v)))].mean(axis=1)
    return tuple(np.quantile(boot,[.025,.975]))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs_test_time_scaling/stage_1'))
    p.add_argument('--resamples',type=int,default=5000)
    a=p.parse_args(); root=a.root; out=root/'tables'; out.mkdir(exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text())
    confirmation_path=root/'confirmation/confirmation_manifest.json'
    selected_names=set()
    if confirmation_path.exists():
        selected_names={c['name'] for c in json.loads(confirmation_path.read_text())['selected']}
    dev=set(protocol['development_ids']); allids=set(protocol['pocket_ids'])
    selections=[]; obsolete_selections=[]; candidates=[]; oracles=[]; diversity=[]
    markers = list(root.glob('*/EVALUATED')) + list(root.glob('development/*/EVALUATED')) + list(root.glob('confirmation/*/EVALUATED')) + list(root.glob('development_extensions/*/EVALUATED')) + list(root.glob('development_conditioning_v2/*/EVALUATED'))
    for marker in sorted(markers):
        folder=marker.parent; run=json.loads((folder/'run.json').read_text())
        f=load_candidates(folder); f['seed']=run['seed']; f['run']=folder.name
        obsolete_partial=run.get('method')=='partial_refinement' and run.get('partial_conditioning_version',1)!=2
        f['analysis_role']='diagnostic_incomplete_conditioning' if obsolete_partial else 'formal'
        timing=generation_costs(folder)
        for field in ['total_generation_seconds','gpu_sampling_seconds','reconstruction_io_data_seconds']:
            f[field]=[timing[(r.data_id,int(r.i_repeat))][field] for r in f.itertuples()]
        if run.get('start_repeat',0): continue
        # Prefer complete N=8 view when available; avoid double counting overlapping pools.
        if folder.name.startswith('pool_t100_n4_') and (root/f"pool_t100_n8_r0_s{run['seed']}"/'EVALUATED').exists(): continue
        candidates.append(f)
        pool=folder.name.startswith('pool_')
        fk=run.get('method')=='fk_particles'
        partial='selection' in run and 'rounds' in run
        metadata={r['method']:r for r in pd.read_csv(folder/'tts_eval/summary.csv').to_dict('records')}
        for selection in sorted((folder/'tts_eval').glob('best_of_*.csv')):
            n=int(selection.stem.split('_')[2]); ranker=selection.stem.split('_',3)[3]
            if fk and (n!=run['particles'] or ranker!='geometry_confidence'): continue
            if not pool and not fk and ranker!='confidence': continue
            chosen=pd.read_csv(selection)[['data_id','filename']]
            selected=chosen.merge(f,on=['data_id','filename'],validate='one_to_one')
            method=ranker if pool else f"fk_i{run['interval']}_b{run['strength']:g}" if fk else f"partial_{run['selection']}" if partial else ('native_refinement' if folder.name.startswith('refine_') else 'more_steps')
            if partial and run.get('partial_conditioning_version',1)!=2:
                method += '_v1_incomplete_conditioning'
            budget=100*n if pool or fk else 100*(1+run['rounds']) if partial else 100*(1+run['max_refine_rounds']) if method=='native_refinement' else run['steps']
            selected['method']=method; selected['budget']=budget
            summary=metadata[selection.stem]
            searched=f[f.i_repeat<n] if pool or fk else f
            actual_nfe=searched.assign(_nfe=pd.to_numeric(searched.nfe,errors='raise')).groupby('data_id')._nfe.sum()
            selected['actual_graph_nfe']=selected.data_id.map(actual_nfe)
            generation=searched.groupby('data_id').total_generation_seconds.sum()
            selected['generation_seconds_per_output']=selected.data_id.map(generation)
            selected['gpu_sampling_seconds_per_output']=selected.data_id.map(searched.groupby('data_id').gpu_sampling_seconds.sum())
            selected['selection_seconds_per_output']=summary['selection_seconds']/len(selected)
            # Geometry/Vina rankers incur verifier cost for every searched candidate.
            verifier=0.0
            if ranker in ['geometry_confidence','vina_oracle'] and (pool or fk):
                verifier=f.loc[f.i_repeat<n,'physical_score_seconds'].sum()/len(selected)
            if ranker=='vina_oracle' and pool:
                wall=float((folder/'vina_wall_seconds.txt').read_text())
                verifier+=wall*n/run['repeats']/len(selected)
            selected['search_seconds_per_output']=selected.generation_seconds_per_output+selected.selection_seconds_per_output+verifier
            selected['search_verifier_seconds_per_output']=verifier
            if obsolete_partial:
                obsolete_selections.append(selected)
            else:
                selections.append(selected)
        if pool or fk:
            for n in [1,2,4,8]:
                if n>run['repeats'] or fk and n!=run['particles']:continue
                subset=f[f.i_repeat<n]
                for pid,g in subset.groupby('data_id'):
                    if pool:
                        oracles.append(dict(data_id=pid,seed=run['seed'],budget=100*n,P_phys_oracle=bool(g.P_phys.any()),P_joint_oracle=bool(g.P_joint.any()),run=folder.name))
                    mols=[Chem.MolFromSmiles(s) for s in g.loc[g.valid,'smiles'].dropna()]
                    fps=[AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=2048) for m in mols if m]
                    distances=[1-DataStructs.TanimotoSimilarity(fps[i],fps[j]) for i in range(len(fps)) for j in range(i)]
                    smiles=g.loc[g.valid,'smiles'].dropna()
                    diversity.append(dict(data_id=pid,seed=run['seed'],budget=100*n,run=folder.name,pool_kind='FK' if fk else 'independent',valid_candidates=len(smiles),planned_candidates=len(g),valid_candidate_coverage=len(smiles)/len(g),duplicate_fraction=1-smiles.nunique()/len(smiles) if len(smiles) else np.nan,fingerprint_diversity=np.mean(distances) if distances else np.nan,scaffold_count=g.loc[g.complete,'scaffold'].nunique()))
    if not selections: print('No evaluated runs yet');return
    selected=pd.concat(selections,ignore_index=True)
    if obsolete_selections:
        pd.concat(obsolete_selections,ignore_index=True).to_csv(out/'obsolete_partial_v1_selections.csv',index=False)
    if selected.duplicated(['method','budget','data_id','seed']).any():
        raise ValueError('Duplicate method/budget/pocket/seed output; resolve authoritative retry/pool')
    selected['subset']=np.where(selected.data_id.isin(dev),'development','confirmation')
    selected.to_csv(out/'per_pocket_seed_selections.csv',index=False)
    pd.concat(candidates,ignore_index=True).to_csv(out/'all_candidates.csv',index=False)
    oracle=pd.DataFrame(oracles); oracle.to_csv(out/'pool_oracle.csv',index=False)
    pd.DataFrame(diversity).to_csv(out/'pool_diversity.csv',index=False)
    rng=np.random.default_rng(2026); reports=[]; paired=[]; gaps=[]; components=[]; correlations=[]
    metrics=['P_phys','P_joint','P_joint_minus6','P_joint_minus8','valid','complete','geometry_pass','energy_pass','pocket_no_clash']
    for subset in ['development','confirmation','all']:
        sf=selected if subset=='all' else selected[selected.subset==subset]
        targetids=allids if subset=='all' else dev if subset=='development' else allids-dev
        for (method,budget),g in sf.groupby(['method','budget']):
            adaptation=method.startswith('fk_') or method.startswith('partial_')
            global_cell=not adaptation or (method in selected_names and budget in [400,800])
            if not global_cell and subset!='development':
                continue  # Exploratory dev-only configurations do not represent all100.
            expected_seeds=protocol['seeds'] if global_cell else protocol['seeds'][:3]
            expected_pairs={(pid,seed) for pid in targetids for seed in expected_seeds}
            observed_pairs=set(zip(g.data_id,g.seed))
            pocket=g.groupby('data_id')[metrics].mean()
            rep=dict(subset=subset,method=method,budget=budget,pockets=len(pocket),outputs=len(g),seed_count=g.seed.nunique(),expected_outputs=len(expected_pairs),scope='full_protocol' if global_cell else 'exploratory_development',complete_matrix=observed_pairs==expected_pairs and len(g)==len(expected_pairs))
            for m in metrics:
                lo,hi=ci(pocket[m],rng,a.resamples);rep.update({m:float(pocket[m].mean()),m+'_ci_low':lo,m+'_ci_high':hi})
            for m in ['energy_ratio','vina_score','n_atoms','ligand_efficiency','qed','num_pairwise_clashes','clashes_per_heavy_atom','actual_graph_nfe','search_seconds_per_output','generation_seconds_per_output']:
                v=pd.to_numeric(g[m],errors='coerce');rep[m+'_mean']=v.mean();rep[m+'_median']=v.median();rep[m+'_p10']=v.quantile(.1);rep[m+'_p90']=v.quantile(.9);rep[m+'_coverage']=v.notna().mean()
            rep['qualified_outputs_per_search_second']=g.P_phys.sum()/g.search_seconds_per_output.sum()
            rep['joint_outputs_per_search_second']=g.P_joint.sum()/g.search_seconds_per_output.sum()
            rep['unique_smiles_fraction_valid']=g.loc[g.valid,'smiles'].nunique()/max(int(g.valid.sum()),1)
            rep['unique_scaffolds_complete']=g.loc[g.complete,'scaffold'].nunique()
            rep['acyclic_fraction_complete']=g.loc[g.complete,'scaffold'].fillna('').eq('').mean()
            rep['energy_pass_given_complete']=g.loc[g.complete,'energy_pass'].mean()
            rep['geometry_pass_given_complete']=g.loc[g.complete,'geometry_pass'].mean()
            reports.append(rep)
            base=sf[(sf.method=='geometry_confidence')&(sf.budget==budget)]
            match=g.merge(base,on=['data_id','seed'],suffixes=('','_base'),validate='one_to_one')
            for m in ['P_phys','P_joint','valid','complete']:
                if not len(match): continue
                d=(match[m].astype(float)-match[m+'_base'].astype(float)).groupby(match.data_id).mean()
                lo,hi=ci(d,rng,a.resamples)
                paired.append(dict(subset=subset,method=method,budget=budget,reference='geometry_confidence',metric=m,delta=d.mean(),ci_low=lo,ci_high=hi,pockets=len(d),pairs=len(match),
                    method_actual_nfe_mean=match.actual_graph_nfe.mean(),reference_actual_nfe_mean=match.actual_graph_nfe_base.mean(),
                    exact_nfe_pair_fraction=np.isclose(match.actual_graph_nfe,match.actual_graph_nfe_base).mean()))
            if len(oracle) and method in ['random','confidence','geometry_confidence','vina_oracle']:
                match=g.merge(oracle,on=['data_id','seed','budget'],validate='one_to_one')
                for m in ['P_phys','P_joint']:
                    gaps.append(dict(subset=subset,method=method,budget=budget,metric=m,oracle=match[m+'_oracle'].mean(),selected=match[m].mean(),gap=(match[m+'_oracle'].astype(float)-match[m].astype(float)).mean()))
            for m in ['bond_lengths','bond_angles','internal_steric_clash','aromatic_ring_flatness','double_bond_flatness','energy_pass','pocket_no_clash']:
                vals=boolean(g[m]); components.append(dict(subset=subset,method=method,budget=budget,component=m,pass_all=vals.mean(),pass_given_complete=vals[g.complete].mean(),failures=int((~vals).sum()),denominator=len(g)))
        cf=pd.concat(candidates,ignore_index=True)
        cf=cf[cf.analysis_role=='formal']
        cf=cf if subset=='all' else cf[cf.data_id.isin(targetids)]
        for score in ['P_phys','P_joint','vina_score','energy_ratio']:
            values=[]
            for _,g in cf.groupby(['data_id','seed','run']):
                if len(g)<2:continue
                corr=g[['cfd_traj',score]].astype(float).corr(method='spearman').iloc[0,1]
                if np.isfinite(corr):values.append(corr)
            correlations.append(dict(subset=subset,outcome=score,mean_within_pool_spearman=np.mean(values) if values else np.nan,pools_with_defined_correlation=len(values)))
    pd.DataFrame(reports).to_csv(out/'budget_curves.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'paired_comparisons.csv',index=False)
    pd.DataFrame(gaps).to_csv(out/'oracle_gap.csv',index=False)
    pd.DataFrame(components).to_csv(out/'component_failures.csv',index=False)
    pd.DataFrame(correlations).to_csv(out/'confidence_correlations.csv',index=False)
    cf=pd.concat(candidates,ignore_index=True)
    cf=cf[cf.analysis_role=='formal']
    # Reliability bins describe the teacher signal; they do not fit a selector on confirmation pockets.
    cf['confidence_sigmoid_display']=1/(1+np.exp(-pd.to_numeric(cf.cfd_traj,errors='coerce')))
    cf['confidence_bin']=pd.cut(cf.confidence_sigmoid_display,bins=np.linspace(0,1,11),include_lowest=True)
    calibration=cf.groupby('confidence_bin',observed=True).agg(n=('filename','size'),
        mean_signal=('confidence_sigmoid_display','mean'),P_phys=('P_phys','mean'),P_joint=('P_joint','mean'))
    calibration.to_csv(out/'confidence_reliability.csv')
    (out/'confidence_reliability_protocol.json').write_text(json.dumps(dict(display_transform='sigmoid of raw teacher position confidence; monotone visualization only',calibrated_success_probability=False,bins='fixed ten equal sigmoid-display bins, not fit to confirmation outcomes',aggregation='candidate-weighted diagnostic across formal evaluated runs; not a fitted deployment calibration model'),indent=2))
    if 'failure_kind' in cf:
        cf.groupby(['run','failure_kind'],dropna=False).size().rename('count').reset_index().to_csv(out/'chemical_failure_counts.csv',index=False)
    observed={(r['subset'],r['method'],r['budget']):r['complete_matrix'] for r in reports}
    core={(s,m,b) for s in ['development','confirmation','all'] for m in ['random','confidence','geometry_confidence','vina_oracle'] for b in [100,200,400,800]}
    core|={(s,m,b) for s in ['development','confirmation','all'] for m in ['more_steps','native_refinement'] for b in [200,400,800]}
    development={('development',f'fk_i{i}_b{beta}',b) for i in [20,50] for beta in [2,5] for b in [400,800]}
    development|={('development',f'partial_{selection}',b) for selection in ['random','confidence'] for b in [200,400]}
    confirmation={(s,m,b) for s in ['development','confirmation','all'] for m in selected_names for b in [400,800]}
    missing=sorted(k for k in core|development|confirmation if not observed.get(k,False))
    (out/'analysis.json').write_text(json.dumps(dict(bootstrap_resamples=a.resamples,bootstrap_seed=2026,cluster='pocket after seed average',evaluated_runs=len(candidates),
        baseline_matrix_complete=all(observed.get(k,False) for k in core),
        development_matrix_complete=all(observed.get(k,False) for k in development),
        selected_confirmation_complete=bool(selected_names) and all(observed.get(k,False) for k in confirmation),
        missing_or_incomplete_cells=missing,provisional=bool(missing) or not bool(selected_names)),indent=2))
    print(pd.DataFrame(reports)[['subset','method','budget','outputs','P_phys','P_joint','complete_matrix']].to_string(index=False))

if __name__=='__main__':main()
