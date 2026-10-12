"""Wait for complete dev evidence, freeze once, then run confirmation and extensions."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path('outputs_test_time_scaling/stage_1')
def main():
    lock=(ROOT/'confirmation_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while not (ROOT/'confirmation/confirmation_manifest.json').exists():
        r=subprocess.run([sys.executable,'scripts/freeze_stage1_confirmation.py','--check_only'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        (ROOT/'confirmation_readiness.log').write_text(r.stdout)
        if r.returncode==0:
            subprocess.run([sys.executable,'scripts/freeze_stage1_confirmation.py'],check=True)
            print('Complete development evidence; immutable configuration frozen',flush=True)
            break
        if 'RuntimeError: Await' not in r.stdout:raise RuntimeError(r.stdout)
        time.sleep(60)
    while True:
        state=subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
        memory,util=[int(x.strip()) for x in state.split(',')]
        if memory==0 and util==0:break
        print('Waiting for idle GPU2:',state,flush=True);time.sleep(60)
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    (ROOT/'confirmation_queue_gpu_state.json').write_text(json.dumps(dict(host=os.uname().nodename,gpu_index=2,idle_memory_mb=memory,idle_utilization=util,timestamp=time.time()),indent=2))
    for script in ['scripts/run_stage1_confirmation.py','scripts/run_stage1_development_extensions.py']:
        result=subprocess.run([sys.executable,script,'--retry_failed'])
        if result.returncode:raise SystemExit(result.returncode)
    (ROOT/'CONFIRMATION_AND_EXTENSIONS_GENERATED').touch()
if __name__=='__main__':main()
