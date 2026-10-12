"""Verify identical raw prefixes and selections across N4 and merged N8 views."""
import hashlib
import json
from pathlib import Path
import pandas as pd
ROOT=Path('outputs_test_time_scaling/stage_1')
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    protocol=json.loads((ROOT/'protocol.json').read_text());records=[]
    for seed in protocol['seeds']:
        merged=ROOT/f'pool_t100_n8_r0_s{seed}'
        if not (merged/'EVALUATED').exists():records.append(dict(seed=seed,status='await_evaluation'));continue
        m=json.loads((merged/'run.json').read_text());first=Path(m['shards'][0]);a=pd.read_csv(first/'gen_info.csv');b=pd.read_csv(merged/'gen_info.csv');prefix=b[b.i_repeat<4]
        assert len(a)==len(prefix)==400 and len(b)==800
        assert set(zip(a.data_id,a.i_repeat))==set(zip(prefix.data_id,prefix.i_repeat))
        pairs=a.merge(prefix,on=['data_id','i_repeat'],suffixes=('_first','_merged'),validate='one_to_one')
        assert all(digest(first/'SDF'/x.filename_first)==digest(merged/'SDF'/x.filename_merged) for x in pairs.itertuples())
        selections=[]
        for n in [1,2,4]:
            for ranker in ['random','confidence','geometry_confidence','vina_oracle']:
                label=f'best_of_{n}_{ranker}';x=pd.read_csv(first/'tts_eval'/(label+'.csv'));y=pd.read_csv(merged/'tts_eval'/(label+'.csv'))
                q=x[['data_id','filename']].merge(y[['data_id','filename']],on='data_id',suffixes=('_first','_merged'),validate='one_to_one')
                same=sum(digest(first/'SDF'/z.filename_first)==digest(merged/'SDF'/z.filename_merged) for z in q.itertuples())
                selections.append(dict(label=label,identical_selected_sdf=same,outputs=len(q)))
                assert same==len(q)==100,f'Prefix selection differs: {seed}/{label}'
        records.append(dict(seed=seed,status='verified',raw_prefix_candidates=400,merged_candidates=800,selections=selections,first_source=str(first)))
    result=dict(complete=all(r['status']=='verified' for r in records),records=records)
    (ROOT/'nested_pool_audit.json').write_text(json.dumps(result,indent=2));print('Verified nested seeds',sum(r['status']=='verified' for r in records),'/5')
if __name__=='__main__':main()
