"""Evaluate completed Stage 1 runs; never regenerate failed molecules."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('outputs_test_time_scaling/stage_1')
ENV = Path('/shared/healthinfolab/phz24002/anaconda3/envs')

def run(cmd, log):
    with log.open('w') as stream:
        subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)

def evaluate(root, folders=None, vina_workers=2):
    from merge_stage1_pool import merge
    for tail_done in root.glob('pool_tail_*/DONE'):
        tail = tail_done.parent
        first = root / tail.name.replace('pool_tail_', 'pool_', 1)
        dest = root / tail.name.replace('pool_tail_t100_n4_', 'pool_t100_n8_', 1)
        if not (first / 'DONE').exists():
            retries = [p for p in root.glob(first.name + '_retry*') if (p / 'DONE').exists()]
            if len(retries) > 1:
                raise RuntimeError('Multiple completed retries; resolve authoritative source')
            if retries:
                first = retries[0]
        if (first / 'DONE').exists():
            merge(first, tail, dest)
    for done in sorted(root.glob('*/DONE')) if folders is None else [p/'DONE' for p in folders]:
        folder = done.parent
        if not (folder / 'run.json').exists():
            if folders is not None:
                raise FileNotFoundError(folder / 'run.json')
            continue
        _evaluate_folder(folder, vina_workers)

def _evaluate_folder(folder, vina_workers):
    with (folder / 'tts_evaluation.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (folder / 'EVALUATED').exists():
            return
        metadata = json.loads((folder / 'run.json').read_text())
        if metadata.get('start_repeat', 0):
            return  # Tail pool must be merged with initial candidates before nested selection.
        ns = [n for n in [1, 2, 4, 8, 16] if n <= metadata['repeats']]
        phy = str(ENV / 'reed_molecule/bin/python')
        vina = str(ENV / 'pxm_vina/bin/python')
        args = [phy, 'scripts/evaluate_test_time_scaling.py', '--run_dir', str(folder), '--best_of', *map(str, ns)]
        run(args, folder / 'geometry.log')
        tick = time.perf_counter()
        run([vina, 'evaluate/evaluate_vina_sdf.py', '--sdf_dir', str(folder / 'SDF'), '--gen_info',
             str(folder / 'gen_info.csv'), '--protein_dir', 'data/csd/files/proteins', '--out_csv',
             str(folder / 'vina.csv'), '--mode', 'score_only', '--n_workers', str(vina_workers), '--quiet'], folder / 'vina.log')
        (folder / 'vina_wall_seconds.txt').write_text(str(time.perf_counter()-tick))
        run(args + ['--reuse'], folder / 'selection.log')
        labels = [f'best_of_{n}_{r}' for n in ns for r in ['random','confidence','geometry_confidence','vina_oracle']]
        run([phy, 'scripts/evaluate_tts_energy.py', '--run_dir', str(folder), '--all_candidates',
             '--selections', *labels], folder / 'energy.log')
        (folder / 'EVALUATED').touch()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--watch', action='store_true')
    p.add_argument('--folders_file',type=Path)
    p.add_argument('--vina_workers',type=int,default=2)
    a = p.parse_args()
    while True:
        folders=[Path(x) for x in a.folders_file.read_text().splitlines() if x] if a.folders_file else None
        evaluate(a.root,folders,vina_workers=a.vina_workers)
        if not a.watch:
            return
        time.sleep(20)

if __name__ == '__main__':
    main()
