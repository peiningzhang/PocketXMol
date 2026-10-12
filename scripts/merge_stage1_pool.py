"""Build nested N=8 view from disjoint 0:4 and 4:8 candidate shards."""
import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def merge(first, tail, dest):
    if (dest / 'DONE').exists():
        return
    if dest.exists():
        raise ValueError('Incomplete merged directory requires manual inspection')
    runs = [json.loads((p / 'run.json').read_text()) for p in [first, tail]]
    for key in ['checkpoint_sha256', 'pocket_ids', 'seed', 'steps', 'batch_size', 'config_task']:
        if runs[0][key] != runs[1][key]:
            raise ValueError(f'Pool mismatch: {key}')
    frames = [pd.read_csv(p / 'gen_info.csv', keep_default_na=False) for p in [first, tail]]
    pairs = pd.concat(frames)[['data_id', 'i_repeat']]
    if pairs.duplicated().any() or not pairs.groupby('data_id').i_repeat.apply(lambda x: set(x)==set(range(8))).all():
        raise ValueError('Expected exactly 8 disjoint candidates per pocket')
    dest.mkdir(); (dest / 'SDF').mkdir()
    for part, (src, frame) in enumerate(zip([first, tail], frames)):
        names = []
        for old in frame.filename:
            name = f'shard{part}_{old}'
            os.symlink((src / 'SDF' / old).resolve(), dest / 'SDF' / name)
            names.append(name)
        frame['filename'] = names
    pd.concat(frames, ignore_index=True).to_csv(dest / 'gen_info.csv', index=False)
    result = dict(runs[0], repeats=8, start_repeat=0, shards=[str(first),str(tail)], merged_view=True)
    for key in ['graph_forward_calls','forward_calls','sampling_seconds','generation_wall_seconds','num_generated','warmup_seconds']:
        result[key] = sum(r[key] for r in runs)
    result['actual_nfe_per_candidate'] = result['graph_forward_calls']/result['num_generated']
    result['batch_costs'] = runs[0]['batch_costs'] + runs[1]['batch_costs']
    from scripts.stage1_merged_metadata import merged_setup_metadata
    result.update(merged_setup_metadata(runs))
    (dest / 'run.json').write_text(json.dumps(result,indent=2))
    (dest / 'DONE').touch()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['first','tail','dest']: p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args(); merge(a.first,a.tail,a.dest)
