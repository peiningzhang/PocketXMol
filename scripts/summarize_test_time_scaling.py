"""Combine completed TTS runs, preserving generation and verifier budgets."""
import argparse
import json
from pathlib import Path
import pandas as pd


def markdown_table(frame):
    # Avoid adding tabulate to the cluster's scientific environments.
    def cell(value):
        if pd.isna(value):
            return 'NA'
        if isinstance(value, float):
            return f'{value:.4g}'
        return str(value).replace('|', '\\|')
    header = '| ' + ' | '.join(frame.columns) + ' |'
    separator = '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |'
    rows = ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)]
    return '\n'.join([header, separator] + rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = []
    for path in sorted(root.glob('*/tts_eval/summary.csv')):
        run_dir = path.parent.parent
        run = json.loads((run_dir / 'run.json').read_text())
        energy_path = run_dir / 'tts_eval' / 'energy_summary.csv'
        energy_rows = {r['selection']: r for r in pd.read_csv(energy_path).to_dict('records')} if energy_path.exists() else {}
        wall_path = run_dir / 'vina_wall_seconds.txt'
        vina_seconds = float(wall_path.read_text()) if wall_path.exists() else 0.0
        for row in pd.read_csv(path).to_dict('records'):
            if row['method'] == 'all_candidates':
                continue
            n = int(row['candidates_per_output'])
            row['run'] = run_dir.name
            energy_row = energy_rows.get(row['method'], energy_rows.get('best_of_1_confidence', {}) if n == 1 else {})
            for metric in ['qualified_and_energy_pass', 'qualified_energy_and_vina_le_minus7', 'energy_ratio_median', 'energy_coverage']:
                row[metric] = energy_row.get(metric)
            row['nfe_per_output'] = run['actual_nfe_per_candidate'] * n
            row['generation_seconds_estimate'] = run['generation_wall_seconds'] * n / run['repeats']
            row['sampling_seconds_estimate'] = run['sampling_seconds'] * n / run['repeats']
            row['search_vina_seconds_estimate'] = vina_seconds * n / run['repeats'] if n > 1 and 'vina_oracle' in row['method'] else 0
            row['search_total_seconds_estimate'] = row['generation_seconds_estimate'] + row['physical_verifier_seconds'] + row['selection_seconds'] + row['search_vina_seconds_estimate']
            row['qualified_outputs_per_search_second'] = row['qualified'] * row['n'] / row['search_total_seconds_estimate']
            rows.append(row)
    if not rows:
        raise ValueError('No completed evaluation summaries')
    frame = pd.DataFrame(rows)
    frame.to_csv(root / 'comparison.csv', index=False)
    columns = ['run', 'method', 'n', 'nfe_per_output', 'valid', 'complete',
               'geometry_pass', 'qualified', 'qualified_and_energy_pass',
               'qualified_energy_and_vina_le_minus7', 'vina_score_mean', 'search_total_seconds_estimate']
    lines = ['# PocketXMol test-time scaling results', '',
             'Geometry: PoseBusters bond lengths, bond angles, internal clashes, aromatic and double-bond flatness. '
             'Qualified: complete AND geometry pass AND no protein clash. All attempts remain in the denominator.', '',
             'Generation timings are amortized from the shared candidate run; model NFE is measured. '
             'Physical verifier timing is measured per candidate; Vina search time is amortized from the scoring run. '
             'Offline evaluation cost is excluded from confidence-only search cost. Initialization/import costs are excluded. '
             'These are pilot measurements, not matched end-to-end latency benchmarks.', '',
             markdown_table(frame[columns]), '',
             'Vina-oracle selection optimizes the reported Vina score, so its Vina improvement is not independent validation. '
             'Confidence/geometry selectors do not use Vina. Energy-ensemble checks are omitted from geometry_pass. '
             'Diversity and atom count are reported in comparison.csv.', '']
    (root / 'report.md').write_text('\n'.join(lines))
    energy = root / 'steps100_candidates4' / 'tts_eval' / 'energy_summary.csv'
    if energy.exists():
        with (root / 'report.md').open('a') as file:
            file.write('\n## Independent conformer-energy diagnostic\n\n')
            file.write(markdown_table(pd.read_csv(energy)) + '\n')
    print(frame[columns].to_string(index=False))


if __name__ == '__main__':
    main()
