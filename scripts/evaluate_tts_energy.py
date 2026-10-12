"""Independent PoseBusters energy check on selected TTS outputs.

Neither confidence nor geometry selection uses this 50-conformer energy ratio.
Missing energy values are failures and coverage is reported separately.
"""
import argparse
import copy
import json
import logging
import time
import hashlib
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from posebusters import PoseBusters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--eval_dir', default='', help='Evaluation directory when evaluating historical subsets')
    parser.add_argument('--reuse', action='store_true', help='Reuse measured energies for the same run and protocol')
    parser.add_argument('--all_candidates', action='store_true', help='Evaluate entire pool for physical oracle')
    parser.add_argument('--selections', nargs='+', default=['best_of_1_confidence', 'best_of_4_geometry_confidence'])
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')
    energy_module = importlib.import_module('posebusters.modules.energy_ratio')
    original_new_conformation = energy_module.new_conformation
    def single_thread_ensemble(mol, n_confs=1, num_threads=0, energy_minimization=True):
        return original_new_conformation(mol, n_confs, 1, energy_minimization)
    energy_module.new_conformation = single_thread_ensemble
    root = Path(args.run_dir)
    out = Path(args.eval_dir) if args.eval_dir else root / 'tts_eval'
    cfg = copy.deepcopy(PoseBusters('mol').config)
    cfg['modules'] = [m for m in cfg['modules'] if m['function'] in ('loading', 'energy_ratio')]
    selections = {label: pd.read_csv(out / (label + '.csv')) for label in args.selections}
    if args.all_candidates:
        selections['all_candidates'] = pd.read_csv(out / 'candidate_metrics.csv')
    pool = pd.concat(selections.values()).drop_duplicates('filename')
    candidate_hashes = {name: hashlib.sha256((root / 'SDF' / name).read_bytes()).hexdigest()
                        for name in pool.filename}
    records = []
    cache = {}
    run_manifest = json.loads((root / 'run.json').read_text())
    if not args.reuse and run_manifest.get('merged_view'):
        for shard_index, shard_path in enumerate(run_manifest['shards']):
            shard = Path(shard_path)
            source_protocol = shard / 'tts_eval/energy_protocol.json'
            source_records = shard / 'tts_eval/energy_candidates.csv'
            if not source_protocol.exists() or not source_records.exists():
                continue
            previous = json.loads(source_protocol.read_text())
            source_run = json.loads((shard / 'run.json').read_text())
            if previous['config'] != cfg or previous.get('ensemble_threads') != 1 or source_run['checkpoint_sha256'] != run_manifest['checkpoint_sha256']:
                continue
            for record in pd.read_csv(source_records).to_dict('records'):
                old_name = record['filename']
                new_name = f'shard{shard_index}_{old_name}'
                if new_name in candidate_hashes and previous.get('candidate_sha256', {}).get(old_name) == candidate_hashes[new_name]:
                    cache[new_name] = dict(record, filename=new_name, cache_source=str(shard))
    if args.reuse:
        protocol = json.loads((out / 'energy_protocol.json').read_text())
        if protocol['config'] != cfg:
            raise ValueError('Cached energy protocol differs')
        if protocol.get('ensemble_threads') != 1:
            raise ValueError('Cached energy threading protocol differs')
        if any(protocol.get('candidate_sha256', {}).get(name) != digest
               for name, digest in candidate_hashes.items()):
            raise ValueError('Cached energy candidates differ')
        cache = {r['filename']: r for r in pd.read_csv(out / 'energy_candidates.csv').to_dict('records')}
    for row in pool.to_dict('records'):
        if row['filename'] in cache:
            records.append(cache[row['filename']])
            continue
        tick = time.perf_counter()
        record = {'filename': row['filename'], 'energy_ratio': np.nan,
                  'energy_pass': False, 'error': ''}
        warning_messages=[]
        class WarningCapture(logging.Handler):
            def emit(self, warning):
                warning_messages.append(warning.getMessage())
        warning_capture=WarningCapture()
        energy_logger=logging.getLogger(energy_module.__name__)
        energy_logger.addHandler(warning_capture)
        try:
            mol = Chem.MolFromMolFile(str(root / 'SDF' / row['filename']))
            if mol is None or row.get('tag') == 'bad':
                raise ValueError('invalid input')
            report = PoseBusters(copy.deepcopy(cfg)).bust(mol, full_report=True).iloc[0].to_dict()
            ratio = report.get('energy_ratio', np.nan)
            record['energy_ratio'] = float(ratio)
            if not np.isfinite(ratio):
                record['error'] = 'ensemble_or_generated_conformer_energy_unavailable'
            record['energy_pass'] = bool(report.get('internal_energy', False)) if np.isfinite(ratio) else False
        except Exception as exc:
            record['error'] = str(exc)
        finally:
            energy_logger.removeHandler(warning_capture)
        record['energy_warning_messages'] = json.dumps(warning_messages)
        record['seconds'] = time.perf_counter()-tick
        records.append(record)
        pd.DataFrame(records).to_csv(out / 'energy_candidates.csv', index=False)
        print(f'Energy checked {len(records)}/{len(pool)}', flush=True)
    energy = pd.DataFrame(records)
    energy.to_csv(out / 'energy_candidates.csv', index=False)
    summaries = []
    for label, selection in selections.items():
        frame = selection.merge(energy, on='filename', how='left', validate='one_to_one')
        known = frame.energy_ratio.dropna()
        summaries.append({'selection': label, 'n': len(frame),
                          'energy_pass_rate_all_attempts': float(frame.energy_pass.fillna(False).mean()),
                          'energy_coverage': len(known)/len(frame),
                          'energy_ratio_median': float(known.median()) if len(known) else None,
                          'qualified_and_energy_pass': float((frame.qualified & frame.energy_pass).mean()),
                          'qualified_energy_and_vina_le_minus7': float((frame.qualified & frame.energy_pass & pd.to_numeric(frame.vina_score, errors='coerce').le(-7)).mean()),
                          'energy_verifier_seconds': float(frame.seconds.sum())})
    pd.DataFrame(summaries).to_csv(out / 'energy_summary.csv', index=False)
    (out / 'energy_protocol.json').write_text(json.dumps({'config': cfg, 'selection_uses_energy': False,
        'candidate_sha256': candidate_hashes, 'ETKDG_randomSeed': 42,
        'ensemble_threads': 1}, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    main()
