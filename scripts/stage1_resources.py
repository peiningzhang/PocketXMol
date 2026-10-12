"""Conservative aggregate wall charge across Stage 1 GPU drivers.

Charged time includes startup/reconstruction while a card is reserved; this is
an upper bound on model-active GPU-hours, not an estimate of GPU utilization.
"""
import json
import time
from pathlib import Path

ROOT=Path('outputs_test_time_scaling/stage_1')


def charged_hours():
    seconds=0.
    for path in ROOT.rglob('*ledger.json'):
        try:records=json.loads(path.read_text())
        except (OSError,json.JSONDecodeError):continue
        for rec in records:
            seconds+=rec.get('wall_seconds',max(0,time.time()-rec['started']) if rec.get('status')=='running' else 0.)
    return seconds/3600


def check_budget(outdir):
    if ROOT.resolve() not in Path(outdir).resolve().parents:return
    hours=charged_hours()
    if hours>=48:
        (ROOT/'BUDGET_LIMIT').write_text(f'Conservative aggregate charged GPU-hours: {hours:.6f}\n')
        raise RuntimeError('Stage 1 aggregate 48 GPU-hours limit reached; preserve outputs and summarize')
