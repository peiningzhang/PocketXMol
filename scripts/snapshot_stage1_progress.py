"""Snapshot all generation ledgers without interpreting live rows as completion."""
import csv,json,time
from pathlib import Path
from stage1_resources import charged_hours
root=Path('outputs_test_time_scaling/stage_1');rows=[]
for path in sorted(root.rglob('*ledger.json')):
    for rec in json.loads(path.read_text()):
        dest=Path(rec.get('outdir',''))
        rows.append(dict(ledger=str(path.relative_to(root)),label=rec.get('label',dest.name),
            status=rec['status'],outdir=rec.get('outdir'),wall_seconds=rec.get('wall_seconds'),
            graph_nfe=rec.get('graph_nfe'),returncode=rec.get('returncode'),
            generation_marker=(dest/'DONE').exists(),evaluation_marker=(dest/'EVALUATED').exists()))
with (root/'progress_all_tasks.csv').open('w') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(root/'progress_snapshot.json').write_text(json.dumps(dict(timestamp_unix=time.time(),
    charged_gpu_hours=charged_hours(),rows=len(rows),
    scope='Ledger and marker snapshot; running processes must be verified separately. Failed and superseded attempts retained.'),indent=2))
print('Recorded generation attempts:',len(rows))
