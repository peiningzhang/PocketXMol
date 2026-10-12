"""Resumable frozen Stage 1 generation driver; launch only on SSH GPU node."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SEEDS = [2026, 12026, 22026, 32026, 42026]
ROOT = Path('outputs_test_time_scaling/stage_1')

def tasks(smoke=False):
    seeds = SEEDS[:1] if smoke else SEEDS
    # Finish all seeds at B <= 400 before B=800 tasks.
    for high in [False, True]:
        if smoke and high:
            continue
        for seed in seeds:
            specs = [('pool_tail' if high else 'pool', 100, 4, 0), ('steps', 800 if high else 200, 1, 0)]
            if not high:
                specs += [('steps', 400, 1, 0)]
            specs += [('refine', 100, 1, r) for r in ([7] if high else [1, 3])]
            if smoke:
                specs = [('pool', 100, 2, 0), ('refine', 100, 1, 1)]
            for kind, steps, repeats, rounds in specs:
                label = f'{kind}_t{steps}_n{repeats}_r{rounds}_s{seed}'
                yield dict(label=label, kind=kind, steps=steps, repeats=repeats, rounds=rounds, seed=seed)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--retry_failed', action='store_true')
    p.add_argument('--gpu_hours', type=float, default=48)
    a = p.parse_args()
    root = ROOT / 'smoke' if a.smoke else ROOT
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'driver.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    ledger_path = root / 'generation_ledger.json'
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else []
    for task in tasks(a.smoke):
        prior = [r for r in ledger if r['label'] == task['label']]
        if any(r['status'] == 'done' for r in prior):
            continue
        if prior and not a.retry_failed:
            continue
        if sum(r.get('wall_seconds', 0) for r in ledger) / 3600 >= a.gpu_hours:
            break
        dest = root / task['label']
        if dest.exists():
            if (dest / 'DONE').exists():
                continue
            dest = root / (task['label'] + f'_retry{len(prior)}')
            if dest.exists():
                raise RuntimeError(f'Untracked output exists: {dest}; inspect before retry')
        cmd = [sys.executable, 'scripts/sample_test_time_scaling.py', '--outdir', str(dest),
               '--steps', str(task['steps']), '--repeats', str(task['repeats']),
               '--num_pockets', '3' if a.smoke else '100', '--seed', str(task['seed']),
               '--batch_size', '4', '--warmup', '--max_refine_rounds', str(task['rounds']),
               '--config_task', 'configs/sample/test/sbdd_csd/' + ('base.yml' if task['kind'] == 'refine' else 'simple.yml')]
        if task['kind'] == 'pool_tail':
            cmd += ['--start_repeat', '4']
        if (ROOT / 'pocket_ids.txt').exists() and not a.smoke:
            cmd += ['--pocket_ids', str(ROOT / 'pocket_ids.txt')]
        rec = dict(task, outdir=str(dest), status='running', command=cmd, started=time.time())
        ledger.append(rec)
        ledger_path.write_text(json.dumps(ledger, indent=2))
        with (root / (dest.name + '.log')).open('w') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        rec.update(status='done' if result.returncode == 0 else 'failed', returncode=result.returncode,
                   wall_seconds=time.time() - rec['started'])
        if (dest / 'run.json').exists():
            rec['graph_nfe'] = json.loads((dest / 'run.json').read_text())['graph_forward_calls']
        ledger_path.write_text(json.dumps(ledger, indent=2))
        with (root / 'progress.csv').open('w') as f:
            fields = ['label', 'status', 'outdir', 'wall_seconds', 'graph_nfe', 'returncode']
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(ledger)
        if result.returncode:
            raise SystemExit(result.returncode)

if __name__ == '__main__':
    main()
