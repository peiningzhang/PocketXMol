"""Repeated prototype search cost and startup/warm-up breakdown, same hardware."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--trace_control',action='store_true');args=parser.parse_args()
root=Path('outputs_test_time_scaling/stage_1');folder=root/('timing_trace_control' if args.trace_control else 'timing');path=folder/'timing_ledger.json'
prefix='trace_control_' if args.trace_control else ''
ledger=json.loads(path.read_text());rows=[]
for record in ledger:
    if record['status']!='done':continue
    run_dir=Path(record['outdir']);run=json.loads((run_dir/'run.json').read_text())
    ids=run['pocket_ids'];outputs=len(ids);method=record['method']
    ns=record['budget']//100 if method=='geometry_confidence' or method.startswith('fk_') else 1
    ranker='geometry_confidence' if ns>1 else 'confidence'
    summary=pd.read_csv(run_dir/'tts_eval/summary.csv').set_index('method').loc[f'best_of_{ns}_{ranker}']
    metrics=pd.read_csv(run_dir/'tts_eval/candidate_metrics.csv')
    physical=metrics.physical_score_seconds.sum() if ranker=='geometry_confidence' else 0
    selector=float(summary.selection_seconds)
    row=dict(method=method,budget=record['budget'],trial=record['trial'],pockets=outputs,
        actual_graph_nfe_per_output=run['graph_forward_calls']/outputs,
        hot_generation_seconds=run['generation_wall_seconds'],gpu_sampling_seconds=run['sampling_seconds'],
        selector_seconds=selector,search_physical_verifier_seconds=physical,
        search_seconds_per_output=(run['generation_wall_seconds']+selector+physical)/outputs,
        startup_seconds=run.get('startup_seconds'),import_seconds=run.get('import_seconds'),
        warmup_seconds=run['warmup_seconds'],warmup_graph_nfe=run.get('warmup_calls',{}).get('graph_forward_calls'),
        data_setup_seconds=run.get('data_setup_seconds'),data_loader_seconds=run.get('data_loader_and_stream_setup_seconds'),
        reconstruction_seconds=run.get('reconstruction_seconds'),hostname=run.get('runtime',{}).get('hostname'),
        cuda_visible_devices=run.get('runtime',{}).get('cuda_visible_devices'),gpu_name=run['gpu_name'],run=str(run_dir))
    rows.append(row)
    if method=='geometry_confidence' and 'vina_pipeline_wall_seconds' in record:
        vino=dict(row,method='vina_oracle',search_vina_verifier_seconds=record['vina_pipeline_wall_seconds'],
            search_seconds_per_output=(run['generation_wall_seconds']+selector+physical+record['vina_pipeline_wall_seconds'])/outputs)
        proto_path=run_dir/'vina.csv.protocol.json'
        if proto_path.exists():
            proto=json.loads(proto_path.read_text());vino['receptor_preparation_seconds']=proto['receptor_preparation_seconds'];vino['receptors_reused']=sum(r['cache_reused'] for r in proto['prepared_receptors'])
        rows.append(vino)
tables=root/'tables';tables.mkdir(exist_ok=True);f=pd.DataFrame(rows);f.to_csv(tables/(prefix+'repeated_timing_trials.csv'),index=False)
if len(f):
    if f.groupby(['method','budget']).apply(lambda g:g[['hostname','cuda_visible_devices','gpu_name']].drop_duplicates().shape[0]).max()!=1:
        raise ValueError('Repeated timing hardware not consistent within a method/budget')
    summary=f.groupby(['method','budget']).search_seconds_per_output.agg(['mean','median','std','min','max','count']).reset_index();summary.to_csv(tables/(prefix+'repeated_timing_summary.csv'),index=False)
    comparisons=[]
    for (method,budget),g in f.groupby(['method','budget']):
        base=f[(f.method=='geometry_confidence')&(f.budget==budget)]
        paired=g.merge(base,on=['budget','trial'],suffixes=('','_base'),validate='one_to_one')
        if len(paired):
            comparisons.append(dict(method=method,budget=budget,paired_trials=len(paired),
                cost_ratio=paired.search_seconds_per_output.sum()/paired.search_seconds_per_output_base.sum(),
                trial_cost_ratio_min=(paired.search_seconds_per_output/paired.search_seconds_per_output_base).min(),
                trial_cost_ratio_max=(paired.search_seconds_per_output/paired.search_seconds_per_output_base).max(),
                same_hardware=bool((paired.hostname==paired.hostname_base).all() and (paired.cuda_visible_devices==paired.cuda_visible_devices_base).all()),
                trace_materialization_control=args.trace_control))
    pd.DataFrame(comparisons).to_csv(tables/(prefix+'repeated_timing_comparisons.csv'),index=False)
    print(summary.to_string(index=False))
(folder/'analysis.json').write_text(json.dumps(dict(completed_sampling_trials=len([x for x in ledger if x['status']=='done']),baseline_complete=(folder/'BASELINE_DONE').exists(),
    selected_complete=(folder/'DONE').exists(),scope='prototype throughput per output at batch4, not single-request production latency',
    trace_materialization_control=args.trace_control,
    limitation='Common full-trajectory recording disabled; confidence histories retained; throughput is not single-request latency' if args.trace_control else 'Original baseline sampler materializes unused full trajectories; FK and partial loops differ in diagnostic overhead. Cost-only stage2 claims require a common trace-overhead control, not raw timing ratios alone'),indent=2))
