"""Evaluate confirmation and frozen dev extensions sequentially after selection."""
from pathlib import Path
import time
from evaluate_stage1 import evaluate
ROOT=Path('outputs_test_time_scaling/stage_1')
while not (ROOT/'confirmation/confirmation_manifest.json').exists():time.sleep(60)
roots=[ROOT/'confirmation',ROOT/'development_extensions']
while True:
    for root in roots:
        if root.exists():
            evaluate(root)
            done=list(root.glob('*/DONE'))
            if (root/'GENERATION_COMPLETE').exists() and done and all((p.parent/'EVALUATED').exists() for p in done):
                (root/'EVALUATION_COMPLETE').touch()
    if all((root/'EVALUATION_COMPLETE').exists() for root in roots):break
    time.sleep(20)
