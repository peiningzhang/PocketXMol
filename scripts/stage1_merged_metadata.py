"""Merge measured setup counters without inventing missing early-run fields."""
def merged_setup_metadata(runs):
    result={}
    fields=['import_seconds','startup_seconds','data_setup_seconds','model_noiser_setup_seconds','data_loader_and_stream_setup_seconds','reconstruction_seconds']
    for field in fields:
        values=[r.get(field) for r in runs]
        result[field]=sum(values) if all(v is not None for v in values) else None
        result['shard_'+field]=values
    values=[r.get('warmup_calls') for r in runs]
    result['shard_warmup_calls']=values
    keys=['forward_calls','graph_forward_calls']
    result['warmup_calls']={k:sum(v[k] for v in values) for k in keys} if all(v is not None for v in values) else None
    result['warmup_calls_known_lower_bound']={k:sum(v.get(k,0) for v in values if v is not None) for k in keys}
    result['setup_metadata_complete']=all(result[field] is not None for field in fields) and result['warmup_calls'] is not None
    result['merged_metadata_note']='Setup and warm-up are summed across sampler subprocesses only where every shard has measured fields. Null means an early shard did not record the measurement; known lower bounds are separate.'
    return result

if __name__=='__main__':
    import json
    from pathlib import Path
    root=Path('outputs_test_time_scaling/stage_1')
    repaired=0
    for path in root.glob('pool_t100_n8*/run.json'):
        run=json.loads(path.read_text())
        if not run.get('merged_view'):continue
        sources=[json.loads((Path(p)/'run.json').read_text()) for p in run['shards']]
        backup=path.with_name('run_before_setup_metadata_repair.json')
        if not backup.exists():backup.write_text(path.read_text())
        run.update(merged_setup_metadata(sources));path.write_text(json.dumps(run,indent=2));repaired+=1
    print('Merged metadata views checked:',repaired)
