"""Append frozen-method timing after baseline protocols release the same GPU."""
import fcntl
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path('outputs_test_time_scaling/stage_1')
lock=(ROOT/'selected_timing_queue.lock').open('w')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:raise SystemExit(0)
while not all((ROOT/p).exists() for p in ['confirmation/confirmation_manifest.json','timing/BASELINE_DONE','timing_trace_control/BASELINE_DONE']):time.sleep(60)
for extra in [[],['--trace_control']]:
    subprocess.run([sys.executable,'scripts/benchmark_stage1.py',*extra],check=True)
