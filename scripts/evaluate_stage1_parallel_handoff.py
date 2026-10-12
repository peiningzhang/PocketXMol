"""Evaluate explicit disjoint folders while an existing queue parent is stopped.

The current child keeps running. Always resume the parent after this finite batch.
"""
import argparse,json,signal,subprocess
from pathlib import Path
from evaluate_stage1 import evaluate
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--folders_file',type=Path,required=True)
p.add_argument('--resume_host',required=True)
p.add_argument('--resume_pid',type=int,required=True)
a=p.parse_args()
def stop(signum,frame):raise KeyboardInterrupt('handoff interrupted')
signal.signal(signal.SIGTERM,stop)
try:
    folders=[Path(x) for x in a.folders_file.read_text().splitlines() if x]
    for folder in folders:
        evaluate(folder.parent,[folder],vina_workers=1)
        print('Completed handoff folder:',folder,flush=True)
finally:
    result=subprocess.run(['ssh','-F','/dev/null','-o','ConnectTimeout=10',a.resume_host,
                           'kill','-CONT',str(a.resume_pid)],timeout=30)
    a.folders_file.with_suffix('.resume.json').write_text(json.dumps(dict(
        host=a.resume_host,pid=a.resume_pid,returncode=result.returncode),indent=2))
    if result.returncode:raise RuntimeError('Queue parent needs explicit resume')
