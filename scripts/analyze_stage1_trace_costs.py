"""Compare all completed paired recording-on/off timing trials."""
import json
from pathlib import Path
import pandas as pd

root = Path('outputs_test_time_scaling/stage_1/tables')
on = pd.read_csv(root / 'repeated_timing_trials.csv')
off = pd.read_csv(root / 'trace_control_repeated_timing_trials.csv')
keys = ['method', 'budget', 'trial']
f = on.merge(off, on=keys, suffixes=('_on', '_off'), validate='one_to_one')
assert len(f) == len(on) == len(off), 'Unpaired timing trials'
for field in ['hostname', 'cuda_visible_devices', 'gpu_name', 'pockets']:
    assert (f[field+'_on'] == f[field+'_off']).all(), field
f['full_trace_cost_ratio'] = f.search_seconds_per_output_on / f.search_seconds_per_output_off
f['exact_nfe_match'] = f.actual_graph_nfe_per_output_on == f.actual_graph_nfe_per_output_off
f.to_csv(root / 'trace_materialization_comparison_trials.csv', index=False)
rows = []
for (method, budget), g in f.groupby(['method', 'budget']):
    rows.append(dict(method=method, budget=budget,
        trace_on_mean=g.search_seconds_per_output_on.mean(),
        trace_off_mean=g.search_seconds_per_output_off.mean(), trials=len(g),
        trace_on_to_off_ratio=g.search_seconds_per_output_on.sum()/g.search_seconds_per_output_off.sum(),
        actual_nfe_on_mean=g.actual_graph_nfe_per_output_on.mean(),
        actual_nfe_off_mean=g.actual_graph_nfe_per_output_off.mean(),
        exact_nfe_trial_fraction=g.exact_nfe_match.mean()))
pd.DataFrame(rows).to_csv(root / 'trace_materialization_comparison.csv', index=False)
(root / 'trace_materialization_comparison_protocol.json').write_text(json.dumps(dict(
    hardware_identical=True, paired_trials=len(f), methods=sorted(f.method.unique()),
    scope='prototype batch4 search throughput', confidence_histories_retained=True,
    interpretation='Empirical full recording on/off costs, not a pure causal estimate of recording overhead: GPU generation is not bitwise deterministic, reconstruction cost varies and native refinement can stop early. Actual NFE and matched fraction must be inspected.'), indent=2))
print(f'{len(f)} paired trials across {len(rows)} method/budget cells')
