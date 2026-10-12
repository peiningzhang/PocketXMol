"""Recover total generation costs per candidate from batch boundaries.

Older runs record GPU sampling seconds plus total generation wall time. Their
log end timestamps identify batch durations; the first interval is determined
from total wall minus the remaining intervals. Total charge is conserved.
"""
from datetime import datetime
import json
from pathlib import Path
import re
import pandas as pd


def generation_costs(folder):
    folder=Path(folder);run=json.loads((folder/'run.json').read_text())
    if run.get('merged_view'):
        result={}
        for shard in run['shards']:
            costs=generation_costs(shard)
            if result.keys() & costs.keys():raise ValueError('Timing shards overlap')
            result.update(costs)
        return result
    info=pd.read_csv(folder/'gen_info.csv')
    costs={}
    if run.get('inference_nfe_not_measured_here'):return costs
    batches=run['batch_costs'];ends=[]
    for line in (folder/'log.txt').read_text().splitlines():
        if '] repeat=' not in line and '] pocket=' not in line:continue
        match=re.match(r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})::',line)
        if match:ends.append(datetime.strptime(match.group(1),'%Y-%m-%d %H:%M:%S,%f').timestamp())
    if len(ends)!=len(batches):raise ValueError(f'Timing log/batch count mismatch: {folder}')
    walls=[run['generation_wall_seconds']-(ends[-1]-ends[0])]+[b-a for a,b in zip(ends,ends[1:])]
    for batch,wall in zip(batches,walls):
        if wall<0:raise ValueError('Negative recovered generation interval')
        frame=info[info.data_id.isin(batch['data_ids'])]
        if 'repeat' in batch:frame=frame[frame.i_repeat==batch['repeat']]
        if not len(frame):raise ValueError('Timing batch has no candidates')
        for r in frame.itertuples():
            key=(r.data_id,int(r.i_repeat))
            if key in costs:raise ValueError('Candidate assigned multiple generation intervals')
            costs[key]=dict(total_generation_seconds=wall/len(frame),
                            gpu_sampling_seconds=batch['seconds']/len(frame),
                            reconstruction_io_data_seconds=(wall-batch['seconds'])/len(frame),
                            timing_source='batch wall recovered from log boundaries and measured total wall')
    if len(costs)!=len(info):raise ValueError('Timing coverage incomplete')
    total=sum(v['total_generation_seconds'] for v in costs.values())
    if abs(total-run['generation_wall_seconds'])>1e-6:raise ValueError('Generation time not conserved')
    return costs
