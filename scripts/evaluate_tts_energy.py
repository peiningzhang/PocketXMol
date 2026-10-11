"""Independent PoseBusters energy check on selected TTS outputs.

Neither confidence nor geometry selection uses this 50-conformer energy ratio.
Missing energy values are failures and coverage is reported separately.
"""
import argparse
import copy
import json
import time
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
    parser.add_argument('--selections', nargs='+', default=['best_of_1_confidence', 'best_of_4_geometry_confidence'])
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')
    root = Path(args.run_dir)
    out = Path(args.eval_dir) if args.eval_dir else root / 'tts_eval'
    cfg = copy.deepcopy(PoseBusters('mol').config)
    cfg['modules'] = [m for m in cfg['modules'] if m['function'] in ('loading', 'energy_ratio')]
    selections = {label: pd.read_csv(out / (label + '.csv')) for label in args.selections}
    pool = pd.concat(selections.values()).drop_duplicates('filename')
    records = []
    cache = {}
    if args.reuse:
        protocol = json.loads((out / 'energy_protocol.json').read_text())
        if protocol['config'] != cfg:
            raise ValueError('Cached energy protocol differs')
        cache = {r['filename']: r for r in pd.read_csv(out / 'energy_candidates.csv').to_dict('records')}
    for row in pool.to_dict('records'):
        if row['filename'] in cache:
            records.append(cache[row['filename']])
            continue
        tick = time.perf_counter()
        record = {'filename': row['filename'], 'energy_ratio': np.nan,
                  'energy_pass': False, 'error': ''}
        try:
            mol = Chem.MolFromMolFile(str(root / 'SDF' / row['filename']))
            if mol is None or row.get('tag') == 'bad':
                raise ValueError('invalid input')
            report = PoseBusters(copy.deepcopy(cfg)).bust(mol, full_report=True).iloc[0].to_dict()
            ratio = report.get('energy_ratio', np.nan)
            record['energy_ratio'] = float(ratio)
            record['energy_pass'] = bool(report.get('internal_energy', False)) if np.isfinite(ratio) else False
        except Exception as exc:
            record['error'] = str(exc)
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
    (out / 'energy_protocol.json').write_text(json.dumps({'config': cfg, 'selection_uses_energy': False}, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    main()
