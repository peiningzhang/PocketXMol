"""Independent verification of corrected saved repair states; no new sampling."""
from pathlib import Path
import subprocess
import sys
for root in ['development_conditioning_v2','development']:
    subprocess.run([sys.executable,'scripts/evaluate_stage1_diagnostics.py','--root','outputs_test_time_scaling/stage_1/'+root],check=True)
Path('outputs_test_time_scaling/stage_1/DIAGNOSTIC_EVALUATION_COMPLETE').touch()
