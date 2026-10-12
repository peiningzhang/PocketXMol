"""Exclusive unstarted folder handoff; preserve active CPU29 child."""
import subprocess
import sys
try:
    code=subprocess.call([sys.executable,'scripts/evaluate_stage1.py','--root','outputs_test_time_scaling/stage_1/development','--folders_file','outputs_test_time_scaling/stage_1/development/gpu39_remaining_evaluation_folders.txt'])
finally:
    subprocess.run(['ssh','-F','/dev/null','gpu29','kill -CONT 1209318'],check=True)
raise SystemExit(code)
