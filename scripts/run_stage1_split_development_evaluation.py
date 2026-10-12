"""Exclusive final development evaluation split; retain existing active children."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
p=argparse.ArgumentParser();p.add_argument('--family',choices=['fk','partial'],required=True);a=p.parse_args()
r=Path('outputs_test_time_scaling/stage_1/development')
subprocess.run([sys.executable,'scripts/evaluate_stage1.py','--root',str(r),'--folders_file',str(r/(a.family+'_final_evaluation_folders.txt'))],check=True)
(r/(a.family+'_FINAL_EVALUATION_DONE')).touch()
if a.family=='fk':
    while not (r/'partial_FINAL_EVALUATION_DONE').exists():time.sleep(60)
    subprocess.run(['ssh','-F','/dev/null','gpu39','kill -CONT 18760'],check=True)
