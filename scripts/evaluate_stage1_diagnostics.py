"""Evaluate saved refinement states; count repairs and newly introduced failures.

Diagnostic verifiers are separate analysis overhead, never selector rewards.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_stage1 import load_candidates
from scripts.evaluate_stage1 import evaluate

def diagnostic_metadata(metadata, state, info):
    derived=dict(metadata,outdir=str(state),num_generated=len(info),repeats=1,start_repeat=0,
                 pocket_ids=list(info.data_id),num_pockets=info.data_id.nunique(),
                 diagnostic_state=state.name,inference_nfe_not_measured_here=True,
                 parent_run_outdir=metadata['outdir'],
                 parent_run_graph_forward_calls=metadata.get('graph_forward_calls'))
    for field in ['graph_forward_calls','forward_calls','generation_wall_seconds','sampling_seconds',
                  'startup_seconds','import_seconds','warmup_seconds','warmup_calls',
                  'data_setup_seconds','data_loader_and_stream_setup_seconds','reconstruction_seconds']:
        derived[field]=None
    derived['diagnostic_cost_note']='Saved intermediate states, not independently generated runs; parent totals are not per-state measurements.'
    return derived


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('outputs_test_time_scaling/stage_1/development'))
    a=p.parse_args(); comparisons=[]
    for parent_done in sorted(a.root.glob('partial_*/DONE')):
        parent=parent_done.parent;metadata=json.loads((parent/'run.json').read_text())
        if metadata.get('partial_conditioning_version',1)!=2:
            continue
        states=parent/'diagnostic_states'
        if not states.exists():continue
        for marker in states.glob('*/DIAGNOSTIC_READY'):
            state=marker.parent
            if not (state/'run.json').exists():
                info=pd.read_csv(state/'gen_info.csv')
                derived=diagnostic_metadata(metadata,state,info)
                (state/'run.json').write_text(json.dumps(derived,indent=2))
                (state/'DONE').touch()
        evaluate(states,vina_workers=1)
        initial=load_candidates(states/'round0')
        for final_marker in sorted(states.glob('round*/EVALUATED')):
            final_folder=final_marker.parent
            if final_folder.name=='round0':continue
            final=load_candidates(final_folder)
            previous_folder=states/f'round{int(final_folder.name[5:])-1}'
            previous=load_candidates(previous_folder)
            previous=previous.rename(columns={c:c+'_previous' for c in previous if c!='data_id'})
            pair=initial.merge(final,on='data_id',suffixes=('_initial','_final'),validate='one_to_one').merge(previous,on='data_id',validate='one_to_one')
            for record in pair.to_dict('records'):
                row=dict(run=parent.name,data_id=record['data_id'],seed=metadata['seed'],selection=metadata['selection'],configured_rounds=metadata['rounds'],round=int(final_folder.name[5:]))
                for metric in ['P_phys','P_joint','valid','complete','geometry_pass','energy_pass','pocket_no_clash']:
                    before=bool(record[metric+'_initial']);after=bool(record[metric+'_final'])
                    row[metric+'_initial']=before;row[metric+'_final']=after
                    row[metric+'_repair']=not before and after;row[metric+'_damage']=before and not after
                    prev=bool(record[metric+'_previous']);row[metric+'_previous']=prev
                    row[metric+'_round_repair']=not prev and after;row[metric+'_round_damage']=prev and not after
                for metric in ['num_pairwise_clashes','energy_ratio','vina_score','n_atoms']:
                    row[metric+'_initial']=record[metric+'_initial'];row[metric+'_final']=record[metric+'_final'];row[metric+'_previous']=record[metric+'_previous']
                row['previous_sdf']=str((previous_folder/'SDF'/record['filename_previous']).resolve())
                row['initial_sdf']=str((states/'round0/SDF'/record['filename_initial']).resolve())
                row['final_sdf']=str((final_folder/'SDF'/record['filename_final']).resolve())
                comparisons.append(row)
    out=a.root.parent/'tables';out.mkdir(exist_ok=True)
    frame=pd.DataFrame(comparisons)
    frame.to_csv(out/(a.root.name+'_repair_damage_examples_v2.csv'),index=False)
    if len(frame):
        columns=[c for c in frame if c.endswith('_repair') or c.endswith('_damage')]
        frame.groupby(['selection','configured_rounds','round'])[columns].sum().to_csv(out/(a.root.name+'_repair_damage_summary_v2.csv'))
        print(frame.groupby(['selection','configured_rounds','round'])[columns].sum().to_string())

if __name__=='__main__':main()
