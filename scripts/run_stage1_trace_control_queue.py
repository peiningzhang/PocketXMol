"""Run common trace-off timing after the current baseline timing releases GPU1."""
import os
from pathlib import Path
import subprocess
import sys
import time
pid=4539
while True:
    try:command=Path('/proc/%d/cmdline'%pid).read_bytes()
    except FileNotFoundError:break
    if b'benchmark_stage1.py' not in command:break
    time.sleep(30)
raise SystemExit(subprocess.call([sys.executable,'scripts/benchmark_stage1.py','--baseline_only','--trace_control']))
