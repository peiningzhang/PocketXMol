"""Build dev20 N16 views with exact original N8 prefixes and new repeats8:16."""
import argparse,json,os,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.stage1_timing import generation_costs

def merge(prefix,tail,dest,ids):
    prefix,tail,dest=map(Path,[prefix,tail,dest])
    if (dest/'DONE').exists():return
    if dest.exists():raise ValueError('Inspect incomplete merged view')
    for source in [prefix,tail]:
        if not (source/'DONE').exists():raise ValueError('Source generation incomplete')
    runs=[json.loads((p/'run.json').read_text()) for p in [prefix,tail]]
    for field in ['checkpoint_sha256','seed','steps','batch_size','config_task']:
        if runs[0][field]!=runs[1][field]:raise ValueError('Pool identity differs: '+field)
    if not set(ids)<=set(runs[0]['pocket_ids']) or set(ids)!=set(runs[1]['pocket_ids']):raise ValueError('Subset identity differs')
    frames=[pd.read_csv(p/'gen_info.csv',keep_default_na=False) for p in [prefix,tail]]
    frames=[f[f.data_id.isin(ids)].copy() for f in frames]
    for index,f in enumerate(frames):
        expected={(pid,i) for pid in ids for i in range(index*8,(index+1)*8)}
        if len(f)!=len(expected) or set(zip(f.data_id,f.i_repeat))!=expected:raise ValueError('Repeat grid differs')
    costs={};source_totals=[]
    for source,f in zip([prefix,tail],frames):
        allcost=generation_costs(source)
        selected={(row.data_id,int(row.i_repeat)):allcost[(row.data_id,int(row.i_repeat))] for row in f.itertuples()}
        if costs.keys() & selected.keys():raise ValueError('Timing overlap')
        costs.update(selected)
        source_totals.append(dict(source=str(source),full_graph_calls=json.loads((source/'run.json').read_text())['graph_forward_calls'],selected_candidates=len(f)))
    dest.mkdir();(dest/'SDF').mkdir()
    for index,(source,f) in enumerate(zip([prefix,tail],frames)):
        names=[]
        for name in f.filename:
            new=f'shard{index}_{name}';os.symlink((source/'SDF'/name).resolve(),dest/'SDF'/new);names.append(new)
        f['filename']=names
    info=pd.concat(frames,ignore_index=True);info.to_csv(dest/'gen_info.csv',index=False)
    result=dict(runs[0],outdir=str(dest),pocket_ids=list(ids),num_pockets=len(ids),num_generated=len(info),
                repeats=16,start_repeat=0,merged_view=True,shards=[str(prefix),str(tail)],
                subset_prefix_reused=True,source_full_generation_counts=source_totals,
                graph_forward_calls=int(pd.to_numeric(info.nfe).sum()),forward_calls=None,
                generation_wall_seconds=sum(v['total_generation_seconds'] for v in costs.values()),
                sampling_seconds=sum(v['gpu_sampling_seconds'] for v in costs.values()),batch_costs=[],
                actual_nfe_per_candidate=float(pd.to_numeric(info.nfe).mean()),
                cost_scope='Logical search cost for selected dev20 candidates. Prefix uses measured per-batch amortized costs from original full100 run; no prefix regeneration. Full source GPU calls recorded separately.')
    result['source_runtime']=[run.get('runtime') for run in runs]
    result['runtime']={'merged_source_view':True,'source_runtime':result['source_runtime']}
    result['setup_metadata_complete']=False
    for field in ['startup_seconds','import_seconds','warmup_seconds','warmup_calls','data_setup_seconds',
                  'model_noiser_setup_seconds','data_loader_and_stream_setup_seconds','reconstruction_seconds']:
        result[field]=None
    (dest/'run.json').write_text(json.dumps(result,indent=2));(dest/'DONE').touch()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ['prefix','tail','dest','pocket_ids']:p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args();merge(a.prefix,a.tail,a.dest,a.pocket_ids.read_text().splitlines())
