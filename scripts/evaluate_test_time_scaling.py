"""Identity-free geometry/pocket evaluation and nested Best-of-N baselines.

Run in an environment with RDKit and PoseBusters. Fast geometry checks omit
the costly energy ensemble: geometry_pass is NOT the full PoseBusters pass.
All pass rates use all attempted outputs; missing checks count as failures.
"""
import argparse
import copy
import json
import sys
import time
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from posebusters import PoseBusters
from utils.buster_tools import check_intermolecular_distance

GEOMETRY = ['bond_lengths', 'bond_angles', 'internal_steric_clash',
            'aromatic_ring_flatness', 'double_bond_flatness']


def summarize(frame):
    result = {'n': len(frame), 'pockets': frame.data_id.nunique()}
    for col in ['valid', 'complete', 'geometry_pass', 'pocket_no_clash', 'qualified',
                'qualified_vina_le_minus7']:
        result[col] = float(frame[col].fillna(False).mean())
    for col in ['qed', 'n_atoms', 'vina_score', 'ligand_efficiency', 'num_pairwise_clashes']:
        values = pd.to_numeric(frame[col], errors='coerce').dropna()
        result[col + '_mean'] = float(values.mean()) if len(values) else None
        result[col + '_median'] = float(values.median()) if len(values) else None
        result[col + '_coverage'] = len(values) / len(frame)
    result['unique_smiles_fraction'] = frame.loc[frame.valid, 'smiles'].nunique() / max(int(frame.valid.sum()), 1)
    result['unique_scaffolds'] = frame.loc[frame.complete, 'scaffold'].nunique()
    result['geometry_given_complete'] = float(frame.loc[frame.complete, 'geometry_pass'].mean()) if frame.complete.any() else None
    for col in GEOMETRY:
        result[col + '_pass_rate'] = float(frame[col].fillna(False).mean())
    return result


def select_candidates(frame, n, ranker):
    counts = frame.groupby('data_id').i_repeat.nunique()
    if (counts < n).any():
        raise ValueError(f'Best-of-{n} requires {n} candidates for every pocket')
    pool = frame.loc[frame.i_repeat < n].copy()
    pool['_confidence'] = pd.to_numeric(pool.cfd_traj, errors='coerce').fillna(-np.inf)
    if ranker == 'confidence':
        columns, asc = ['_confidence', 'i_repeat'], [False, True]
    elif ranker == 'random':
        pool['_random'] = [int.from_bytes(hashlib.sha256(
            f"{r.data_id}:{getattr(r, 'base_seed', 2026)}:{n}:{r.i_repeat}".encode()).digest()[:8], 'big')
            for r in pool.itertuples()]
        columns, asc = ['_random', 'i_repeat'], [True, True]
    elif ranker == 'geometry_confidence':
        columns, asc = ['qualified', 'geometry_score', 'complete', '_confidence', 'i_repeat'], [False, False, False, False, True]
    elif ranker == 'vina_oracle':
        pool['_vina'] = pd.to_numeric(pool.vina_score, errors='coerce').fillna(np.inf)
        columns, asc = ['qualified', '_vina', 'i_repeat'], [False, True, True]
    else:
        raise ValueError(ranker)
    return pool.sort_values(columns, ascending=asc, kind='stable').groupby('data_id', sort=False).head(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--outdir', default='')
    parser.add_argument('--protein_dir', default='data/csd/files/proteins')
    parser.add_argument('--num_pockets', type=int, default=0)
    parser.add_argument('--max_repeats', type=int, default=0)
    parser.add_argument('--best_of', type=int, nargs='+', default=[1])
    parser.add_argument('--energy', action='store_true')
    parser.add_argument('--reuse', action='store_true', help='Reuse this output directory candidate_metrics.csv')
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')
    root = Path(args.run_dir)
    out = Path(args.outdir) if args.outdir else root / 'tts_eval'
    out.mkdir(parents=True, exist_ok=True)
    info = pd.read_csv(root / 'gen_info.csv', keep_default_na=False)
    run_manifest = json.loads((root / 'run.json').read_text())
    info['base_seed'] = run_manifest['seed']
    if args.num_pockets:
        info = info.loc[info.data_id.isin(info.data_id.drop_duplicates().iloc[:args.num_pockets])]
    if args.max_repeats:
        info = info.loc[pd.to_numeric(info.i_repeat) < args.max_repeats]
    tick = time.perf_counter()
    cfg = copy.deepcopy(PoseBusters(config='mol').config)
    if not args.energy:
        cfg['modules'] = [m for m in cfg['modules'] if m['function'] != 'energy_ratio']
    candidate_protocol = {
        'config': cfg, 'pocket_clash_cutoff': 0.75,
        'checkpoint_sha256': run_manifest['checkpoint_sha256'],
        'sdf_sha256': {name: hashlib.sha256((root / 'SDF' / name).read_bytes()).hexdigest() for name in info.filename},
        'protein_sha256': {pid: hashlib.sha256((Path(args.protein_dir) / (pid + '_pro.pdb')).read_bytes()).hexdigest() for pid in info.data_id.unique()},
    }
    if args.reuse:
        if json.loads((out / 'candidate_protocol.json').read_text()) != candidate_protocol:
            raise ValueError('Geometry cache protocol, checkpoint or molecular inputs differ')
        frame = pd.read_csv(out / 'candidate_metrics.csv', keep_default_na=False)
        for col in ['valid', 'complete', 'geometry_pass', 'pocket_no_clash', 'qualified'] + GEOMETRY:
            frame[col] = frame[col].astype(str).str.lower().eq('true')
        if set(zip(frame.data_id, frame.filename)) != set(zip(info.data_id, info.filename)):
            raise ValueError('Cached candidates do not match requested evaluation subset')
    else:
        rows = []
        proteins = {}
        for i, row in enumerate(info.to_dict('records')):
            candidate_tick = time.perf_counter()
            record = dict(row, valid=False, complete=False, geometry_pass=False,
                          pocket_no_clash=False, qualified=False, geometry_score=0,
                          qed=np.nan, n_atoms=np.nan, scaffold='', num_pairwise_clashes=np.nan,
                          error='', failure_kind='')
            record.update({col: False for col in GEOMETRY})
            try:
                mol = Chem.MolFromMolFile(str(root / 'SDF' / row['filename']))
                if mol is None or row.get('tag') == 'bad':
                    raw = Chem.MolFromMolFile(str(root / 'SDF' / row['filename']), sanitize=False)
                    record['failure_kind'] = 'reconstruction_failure'
                    if raw is not None:
                        record['n_atoms'] = raw.GetNumHeavyAtoms()
                        try:
                            Chem.SanitizeMol(raw)
                        except Exception as exc:
                            message = str(exc)
                            record['failure_kind'] = 'valence' if 'valence' in message.lower() else 'aromatic_bond' if 'kekul' in message.lower() or 'aromatic' in message.lower() else 'sanitization_other'
                            record['error'] = message
                    raise ValueError('invalid molecule or upstream bad tag')
                record['valid'] = True
                record['complete'] = len(Chem.GetMolFrags(mol)) == 1
                record['smiles'] = Chem.MolToSmiles(mol)
                record['n_atoms'] = mol.GetNumHeavyAtoms()
                record['qed'] = QED.qed(mol)
                record['scaffold'] = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
                # New buster instance prevents result accumulation across calls.
                report = PoseBusters(config=copy.deepcopy(cfg)).bust(mol, full_report=True).iloc[0].to_dict()
                # PoseBusters normalizes its configured display names to these keys.
                for col in GEOMETRY:
                    record[col] = bool(report.get(col, False)) if pd.notna(report.get(col)) else False
                for col in ['number_short_outlier_bonds', 'number_long_outlier_bonds',
                            'number_outlier_angles', 'number_clashes',
                            'number_aromatic_rings_checked', 'number_double_bonds_checked']:
                    record[col] = report.get(col, np.nan)
                record['geometry_score'] = sum(int(record[col]) for col in GEOMETRY)
                record['geometry_pass'] = all(record[col] for col in GEOMETRY)
                if args.energy:
                    record['energy_ratio_passes'] = bool(report.get('internal_energy', False))
                data_id = row['data_id']
                if data_id not in proteins:
                    proteins[data_id] = Chem.MolFromPDBFile(str(Path(args.protein_dir) / (data_id + '_pro.pdb')),
                                                           sanitize=False, proximityBonding=False)
                if proteins[data_id] is None:
                    raise ValueError('protein missing or failed to load')
                clash = check_intermolecular_distance(mol, proteins[data_id],
                            ignore_types={'hydrogens', 'organic_cofactors', 'inorganic_cofactors', 'waters'},
                            clash_cutoff=0.75)['results']
                record['pocket_no_clash'] = bool(clash['no_clashes'])
                record['num_pairwise_clashes'] = int(clash['num_pairwise_clashes'])
                record['qualified'] = record['complete'] and record['geometry_pass'] and record['pocket_no_clash']
            except Exception as exc:
                record['error'] = record['error'] or str(exc)
                if not record['failure_kind']:
                    record['failure_kind'] = 'evaluation_error'
            record['physical_score_seconds'] = time.perf_counter() - candidate_tick
            rows.append(record)
            if (i + 1) % 20 == 0:
                print(f'Evaluated {i+1}/{len(info)}', flush=True)
        frame = pd.DataFrame(rows)
    # Vina is a separate measured verifier. Never silently replace missing scores.
    vina_path = root / 'vina.csv'
    frame = frame.drop(columns=['vina_score', 'ligand_efficiency', 'qualified_vina_le_minus7'], errors='ignore')
    if vina_path.exists():
        vina = pd.read_csv(vina_path)[['filename', 'vina_score']]
        frame = frame.merge(vina, on='filename', how='left', validate='one_to_one')
    else:
        frame['vina_score'] = np.nan
    frame['vina_score'] = pd.to_numeric(frame.vina_score, errors='coerce')
    frame['ligand_efficiency'] = -frame.vina_score / pd.to_numeric(frame.n_atoms, errors='coerce')
    frame['qualified_vina_le_minus7'] = frame.qualified & frame.vina_score.le(-7)
    frame.to_csv(out / 'candidate_metrics.csv', index=False)
    (out / 'candidate_protocol.json').write_text(json.dumps(candidate_protocol, indent=2))
    reports = [{'method': 'all_candidates', **summarize(frame)}]
    for n in args.best_of:
        for ranker in ['random', 'confidence', 'geometry_confidence', 'vina_oracle']:
            selection_tick = time.perf_counter()
            selected = select_candidates(frame, n, ranker)
            selector_seconds = time.perf_counter() - selection_tick
            selected.to_csv(out / f'best_of_{n}_{ranker}.csv', index=False)
            reports.append({'method': f'best_of_{n}_{ranker}', 'candidates_per_output': n,
                            'nfe_per_output': float(pd.to_numeric(frame.nfe).mean()*n) if 'nfe' in frame else None,
                            'selection_seconds': selector_seconds,
                            'physical_verifier_seconds': float(frame.loc[frame.i_repeat < n, 'physical_score_seconds'].sum())
                                if n > 1 and ranker != 'confidence' else 0.0,
                            **summarize(selected)})
    pd.DataFrame(reports).to_csv(out / 'summary.csv', index=False)
    manifest = {'arguments': vars(args), 'geometry_checks': GEOMETRY, 'energy_included': args.energy,
                'denominator': 'all attempted outputs; missing checks fail',
                'evaluation_seconds_this_invocation': time.perf_counter()-tick,
                'vina_oracle_note': 'selection uses the reported Vina metric; optimistic search baseline, not independent validation'}
    if args.reuse and (out / 'evaluation.json').exists():
        manifest['initial_evaluation'] = json.loads((out / 'evaluation.json').read_text())
    (out / 'evaluation.json').write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(reports)[['method', 'n', 'valid', 'complete', 'geometry_pass', 'qualified', 'vina_score_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
