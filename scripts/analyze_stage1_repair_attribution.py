"""Consecutive reconstructed-state changes; raw invariants remain separate."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem,RDLogger
RDLogger.DisableLog('rdApp.*')
ROOT=Path('outputs_test_time_scaling/stage_1')
def bonds(mol):
    return {tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))):str(b.GetBondType()) for b in mol.GetBonds()}
def main():
    rows=[]
    for parent in sorted(list((ROOT/'development').glob('partial_*'))+list((ROOT/'development_conditioning_v2').glob('partial_*'))):
        if not (parent/'DONE').exists():continue
        run=json.loads((parent/'run.json').read_text())
        if run.get('partial_conditioning_version',1)!=2:continue
        states=parent/'diagnostic_states'
        if not (states/'round0/gen_info.csv').exists():continue
        initial=pd.read_csv(states/'round0/gen_info.csv');events=json.loads((parent/'partial_diagnostics.json').read_text())
        counts={e['data_id']:e['atom_count'] for e in events if e['round']==1};offset=0;offsets={}
        for pid in initial.data_id:
            offsets[pid]=offset;offset+=counts[pid]
        for event in events:
            pid=event['data_id'];round_=event['round']
            if pid not in offsets:continue
            before=states/f'round{round_-1}';after=states/f'round{round_}'
            if not (before/'gen_info.csv').exists() or not (after/'gen_info.csv').exists():continue
            info_a=pd.read_csv(before/'gen_info.csv').set_index('data_id').loc[pid]
            info_b=pd.read_csv(after/'gen_info.csv').set_index('data_id').loc[pid]
            pa=before/'SDF'/info_a.filename;pb=after/'SDF'/info_b.filename
            selected=event.get('selected_atom_indices_local',[i-offsets[pid] for i in event['selected_atom_indices']])
            row=dict(run=parent.name,data_id=pid,seed=run['seed'],selection=run['selection'],configured_rounds=run['rounds'],round=round_,selected_atom_indices_local=json.dumps(selected),
                before_sdf=str(pa.resolve()),after_sdf=str(pb.resolve()),raw_atom_type_changes=event['atom_type_changes'],raw_edge_type_changes=event['edge_type_changes'],
                raw_position_rms_displacement=event['position_rms_displacement'],raw_fixed_context_exact=event['fixed_context_exact'],raw_pocket_exact=event['pocket_exact'],alignment_verified=False,error='')
            try:
                a=Chem.MolFromMolFile(str(pa),sanitize=False,removeHs=False,strictParsing=False);b=Chem.MolFromMolFile(str(pb),sanitize=False,removeHs=False,strictParsing=False)
                assert a is not None and b is not None,'unreadable reconstructed state'
                n=event['atom_count'];assert a.GetNumAtoms()==b.GetNumAtoms()==n,'masked/removed atoms: mapping not established'
                assert len(selected)==event['selected_count'] and all(0<=i<n for i in selected),'invalid graph-local selection'
                mask=np.zeros(n,dtype=bool);mask[selected]=True
                za=np.array([x.GetAtomicNum() for x in a.GetAtoms()]);zb=np.array([x.GetAtomicNum() for x in b.GetAtoms()])
                displacement=np.linalg.norm(a.GetConformer().GetPositions()-b.GetConformer().GetPositions(),axis=1)
                ba,bb=bonds(a),bonds(b);changed=[dict(atoms=list(k),before=ba.get(k,'none'),after=bb.get(k,'none'),incident_to_selected=bool(mask[list(k)].any())) for k in sorted(set(ba)|set(bb)) if ba.get(k)!=bb.get(k)]
                try:
                    ca,cb=Chem.Mol(a),Chem.Mol(b);Chem.SanitizeMol(ca);Chem.SanitizeMol(cb)
                    na,nb=bonds(ca),bonds(cb)
                    normalized=[dict(atoms=list(k),before=na.get(k,'none'),after=nb.get(k,'none'),incident_to_selected=bool(mask[list(k)].any())) for k in sorted(set(na)|set(nb)) if na.get(k)!=nb.get(k)]
                    row.update(aromatic_normalization_available=True,sdf_normalized_changed_bonds=json.dumps(normalized),sdf_normalized_changed_bonds_unselected_internal=sum(not x['incident_to_selected'] for x in normalized))
                except Exception as exc:
                    row.update(aromatic_normalization_available=False,aromatic_normalization_error=str(exc))
                row['sdf_unselected_connectivity_changes']=sum(not x['incident_to_selected'] and (x['before']=='none' or x['after']=='none') for x in changed)
                row.update(alignment_verified=True,sdf_atom_type_changed_indices=json.dumps(np.where(za!=zb)[0].tolist()),
                    sdf_changed_atom_types_unselected=int(((za!=zb)&~mask).sum()),sdf_selected_position_rms=float(np.sqrt(np.mean(displacement[mask]**2))),
                    sdf_unselected_position_max=float(displacement[~mask].max()) if (~mask).any() else 0,
                    sdf_bond_change_space='serialized before sanitization',sdf_changed_bonds=json.dumps(changed),sdf_changed_bonds_unselected_internal=sum(not x['incident_to_selected'] for x in changed))
            except Exception as exc:row['error']=str(exc)
            rows.append(row)
    out=ROOT/'tables';out.mkdir(exist_ok=True);f=pd.DataFrame(rows);f.to_csv(out/'repair_state_attribution_v2.csv',index=False)
    (out/'repair_state_attribution_protocol.json').write_text(json.dumps(dict(comparisons=len(f),consecutive_rounds=True,index_origin='historical batch-global indices converted using first saved batch graph offsets; explicit local indices preferred',
        mapping_condition='Both decoded SDF atom counts equal raw graph count; decoder and reconstruction add atoms in original order',
        limitation='SDF bond/charge changes can include reconstruction chemistry; raw fixed-context and pocket assertions are distinct evidence. Coordinate values are rounded by SDF serialization.'),indent=2))
    print('Consecutive-state comparisons:',len(f),'verified alignment:',int(f.alignment_verified.sum()) if len(f) else 0)
if __name__=='__main__':main()
