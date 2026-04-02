#!/usr/bin/env python
"""
Standalone Vina docking evaluation from SDF files.

Minimal dependencies: rdkit, meeko, vina
No OpenBabel, AutoDockTools, or pdb2pqr required.
"""

import argparse
import os
import sys
import subprocess
import tempfile
import contextlib
import pickle
import warnings

# Minimal deps: rdkit, meeko, vina, pandas
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation
from vina import Vina
import pandas as pd
import lmdb
import torch

# Suppress noisy meeko deprecation warnings in batch runs.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"meeko\..*")

# Optional for parallel & progress
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from multiprocessing import Pool
    HAS_POOL = True
except ImportError:
    HAS_POOL = False


def _suppress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


def sdf_to_pdbqt(sdf_path, pdbqt_path):
    """Convert SDF to PDBQT using RDKit + meeko (no OpenBabel)."""
    mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to load molecule from {sdf_path}")
    # Ensure 3D and explicit H
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    mol = Chem.AddHs(mol, addCoords=True)
    preparator = MoleculePreparation()
    setups = preparator.prepare(mol)
    if not setups:
        raise ValueError(f"MoleculePreparation failed for {sdf_path}")
    preparator.write_pdbqt_file(pdbqt_path)
    return mol


def get_box_from_mol(mol, size_factor=1.0, buffer=5.0):
    """Compute docking box center and size from molecule coordinates."""
    pos = mol.GetConformer(0).GetPositions()
    pos = [p for p in pos]
    xs, ys, zs = [p[0] for p in pos], [p[1] for p in pos], [p[2] for p in pos]
    center = [
        (max(xs) + min(xs)) / 2,
        (max(ys) + min(ys)) / 2,
        (max(zs) + min(zs)) / 2,
    ]
    extents = [
        (max(xs) - min(xs)) * size_factor + buffer,
        (max(ys) - min(ys)) * size_factor + buffer,
        (max(zs) - min(zs)) * size_factor + buffer,
    ]
    return center, extents


def prepare_receptor_pdbqt(pdb_path, out_base_path, work_dir=None):
    """
    Prepare receptor PDB to PDBQT using meeko CLI.
    Depending on meeko version/options, output can be:
    - {out_base}.pdbqt
    - {out_base}_rigid.pdbqt
    """
    work_dir = os.path.abspath(work_dir or os.path.dirname(out_base_path) or '.')
    base = os.path.splitext(os.path.basename(out_base_path))[0]
    pdb_abs = os.path.abspath(pdb_path)
    if not os.path.exists(pdb_abs):
        raise FileNotFoundError(f"Receptor PDB not found: {pdb_abs}")
    # Use basename for -o so meeko writes to cwd (work_dir)
    ret = subprocess.run(
        [sys.executable, '-m', 'meeko.cli.mk_prepare_receptor',
         '--read_pdb', pdb_abs, '-o', base, '-p'],
        capture_output=True, text=True, cwd=work_dir
    )
    if ret.returncode != 0:
        err = ret.stderr.strip()
        first_line = err.split('\n')[0] if err else "unknown error"
        raise RuntimeError(f"meeko receptor prep failed: {first_line}")
    candidates = [
        os.path.join(work_dir, base + '.pdbqt'),
        os.path.join(work_dir, base + '_rigid.pdbqt'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f"Expected receptor output not found. Tried: {candidates[0]} and {candidates[1]}"
    )


def find_existing_receptor_pdbqt(protein_path, rec_cache_dir=None):
    """
    Prefer pre-prepared receptor PDBQT files.
    Search order:
      1) sibling files near protein_path
      2) cached files under rec_cache_dir
    """
    rec_stem = os.path.splitext(os.path.basename(protein_path))[0]
    sibling_dir = os.path.dirname(protein_path)
    candidates = [
        os.path.join(sibling_dir, rec_stem + '.pdbqt'),
        os.path.join(sibling_dir, rec_stem + '_rigid.pdbqt'),
    ]
    if rec_cache_dir:
        candidates.extend([
            os.path.join(rec_cache_dir, rec_stem + '.pdbqt'),
            os.path.join(rec_cache_dir, rec_stem + '_rigid.pdbqt'),
        ])
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


@_suppress_stdout
def dock_one(sdf_path, protein_path, center=None, box_size=None, exhaustiveness=16,
             tmp_dir=None, rec_cache_dir=None, mode='all'):
    """
    Dock a single SDF against a protein. Returns dict with vina_score, vina_min, vina_dock.
    """
    tmp_dir = tmp_dir or tempfile.mkdtemp()
    rec_cache_dir = rec_cache_dir or tmp_dir
    pid = os.getpid()
    base_name = os.path.splitext(os.path.basename(sdf_path))[0]
    prefix = os.path.join(tmp_dir, f"dock_{pid}_{base_name}")
    lig_pdbqt = prefix + '_lig.pdbqt'

    # Ligand: SDF -> PDBQT
    mol = sdf_to_pdbqt(sdf_path, lig_pdbqt)
    if center is None or box_size is None:
        center, box_size = get_box_from_mol(mol)

    # Receptor: prefer existing prepared pdbqt, then cache
    rec_pdbqt = find_existing_receptor_pdbqt(protein_path, rec_cache_dir=rec_cache_dir)
    if rec_pdbqt is None:
        rec_stem = os.path.splitext(os.path.basename(protein_path))[0]
        rec_candidates = [
            os.path.join(rec_cache_dir, rec_stem + '.pdbqt'),
            os.path.join(rec_cache_dir, rec_stem + '_rigid.pdbqt'),
        ]
        raise FileNotFoundError(
            f"Receptor PDBQT not found in cache for {rec_stem}. "
            f"Tried: {rec_candidates[0]} and {rec_candidates[1]}"
        )

    v = Vina(sf_name='vina', seed=0, verbosity=0)
    v.set_receptor(rec_pdbqt)
    v.set_ligand_from_file(lig_pdbqt)
    v.compute_vina_maps(center=center, box_size=box_size)

    import numpy as np
    vina_score = np.nan
    vina_min = np.nan
    vina_dock = np.nan
    if mode in ['all', 'score_only']:
        vina_score = v.score()[0]
    if mode in ['all', 'minimize']:
        vina_min = v.optimize()[0]
    if mode in ['all', 'dock']:
        v.dock(exhaustiveness=exhaustiveness, n_poses=1)
        vina_dock = v.energies(n_poses=1)[0][0]

    return {'vina_score': vina_score, 'vina_min': vina_min, 'vina_dock': vina_dock}


def _wrap_dock(args):
    """Worker for multiprocessing."""
    sdf_path, protein_path, exhaustiveness, tmp_base, rec_cache, mode = args
    import numpy as np
    try:
        mol = Chem.MolFromMolFile(sdf_path)
        if mol is None:
            raise ValueError("Failed to load mol")
        if mol.HasSubstructMatch(Chem.MolFromSmarts('[#5]')):
            raise ValueError("Contains Boron")
        res = dock_one(sdf_path, protein_path, center=None, box_size=None,
                       exhaustiveness=exhaustiveness, tmp_dir=tmp_base,
                       rec_cache_dir=rec_cache, mode=mode)
        return {'filename': os.path.basename(sdf_path), **res}
    except Exception as e:
        err_msg = str(e)
        if len(err_msg) > 200:
            err_msg = err_msg.split('\n')[0][:197] + "..."
        return {'filename': os.path.basename(sdf_path),
                'vina_score': np.nan, 'vina_min': np.nan, 'vina_dock': np.nan,
                'error': err_msg}


def collect_sdf_pairs(sdf_dir, protein_dir, mapping_csv=None, gen_info_csv=None):
    """
    Collect (sdf_path, protein_path) pairs.

    - gen_info_csv: CSV with (filename, data_id); protein = {data_id}_pro.pdb
    - mapping_csv: CSV with (filename, data_id) or (sdf_name, protein_name)
    - Else: sdf stem = data_id, protein = {data_id}_pro.pdb
    """
    pairs = []
    sdf_files = []
    for root, _, files in os.walk(sdf_dir):
        for f in files:
            if f.endswith('.sdf') and '-all' not in f and '-in' not in f and '-out' not in f:
                sdf_files.append((os.path.join(root, f), f))

    fn2data_id = {}
    fn2prot = {}
    if gen_info_csv and os.path.exists(gen_info_csv):
        df = pd.read_csv(gen_info_csv)
        if 'filename' in df.columns and 'data_id' in df.columns:
            fn2data_id = df.set_index('filename')['data_id'].astype(str).to_dict()
    if mapping_csv and os.path.exists(mapping_csv):
        df = pd.read_csv(mapping_csv)
        fn_col = 'filename' if 'filename' in df.columns else 'sdf_name'
        if 'data_id' in df.columns:
            fn2data_id = df.set_index(fn_col)['data_id'].astype(str).to_dict()
        elif 'protein_name' in df.columns or 'protein_path' in df.columns:
            prot_col = 'protein_name' if 'protein_name' in df.columns else 'protein_path'
            fn2prot = df.set_index(fn_col)[prot_col].astype(str).to_dict()

    for sdf_path, fn in sdf_files:
        fid = fn.replace('.sdf', '')
        prot_path = None
        if fn in fn2prot or fid in fn2prot:
            p = fn2prot.get(fn, fn2prot.get(fid))
            prot_path = p if os.path.isabs(p) else os.path.join(protein_dir, p)
        elif fn in fn2data_id or fid in fn2data_id:
            data_id = fn2data_id.get(fn, fn2data_id.get(fid))
            for suf in ['_pro.pdb', '.pdb']:
                p = os.path.join(protein_dir, data_id + suf)
                if os.path.exists(p):
                    prot_path = p
                    break
            prot_path = prot_path or os.path.join(protein_dir, data_id + '_pro.pdb')
        else:
            for suf in ['_pro.pdb', '.pdb']:
                p = os.path.join(protein_dir, fid + suf)
                if os.path.exists(p):
                    prot_path = p
                    break
            prot_path = prot_path or os.path.join(protein_dir, fid + '_pro.pdb')
        pairs.append((sdf_path, prot_path))
    return pairs


def _protein_rel_to_rec_rel(protein_rel):
    """
    Convert CrossDocked pocket filename to receptor filename when needed.
    Example:
      XXX/2z3h_A_rec_1wn6_bst_lig_tt_docked_3_pocket10.pdb
    -> XXX/2z3h_A_rec.pdb
    """
    if protein_rel.endswith('_pocket10.pdb') and '_rec_' in protein_rel:
        stem = protein_rel[:-len('_pocket10.pdb')]
        left = stem.split('_rec_')[0]
        return left + '_rec.pdb'
    return protein_rel


def _load_lmdb_item(lmdb_env, idx):
    with lmdb_env.begin() as txn:
        raw = txn.get(str(idx).encode())
    if raw is None:
        return None
    return pickle.loads(raw)


def collect_sdf_pairs_from_lmdb(
    sdf_dir, gen_info_csv, lmdb_path, split_path, protein_root, split_key='test'
):
    """
    Build (sdf_path, protein_path) by:
      csd data_id -> split order -> LMDB index -> LMDB protein_filename.
    """
    if not gen_info_csv or not os.path.exists(gen_info_csv):
        raise FileNotFoundError("LMDB mode requires --gen_info")
    if not os.path.exists(lmdb_path):
        raise FileNotFoundError(f"LMDB file not found: {lmdb_path}")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")
    if not os.path.exists(protein_root):
        raise FileNotFoundError(f"Protein root not found: {protein_root}")

    df = pd.read_csv(gen_info_csv)
    if 'filename' not in df.columns or 'data_id' not in df.columns:
        raise ValueError("--gen_info must contain columns: filename,data_id")
    fn2id = df.set_index('filename')['data_id'].astype(str).to_dict()

    split_obj = torch.load(split_path)
    if split_key not in split_obj:
        raise KeyError(f"split key '{split_key}' not found in {split_path}")
    split_indices = split_obj[split_key]

    lmdb_env = lmdb.open(
        lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False
    )

    pairs = []
    skipped = 0
    for root, _, files in os.walk(sdf_dir):
        for f in files:
            if not f.endswith('.sdf'):
                continue
            if any(x in f for x in ['-all.sdf', '-in.sdf', '-out.sdf', '-raw.sdf']):
                continue
            data_id = fn2id.get(f, fn2id.get(f.replace('.sdf', '')))
            if data_id is None or not data_id.startswith('csd_'):
                skipped += 1
                continue
            try:
                order = int(data_id.split('_')[1]) - 100000
            except Exception:
                skipped += 1
                continue
            if order < 0 or order >= len(split_indices):
                skipped += 1
                continue
            lmdb_idx = split_indices[order]
            item = _load_lmdb_item(lmdb_env, lmdb_idx)
            if item is None or 'protein_filename' not in item:
                skipped += 1
                continue

            protein_rel = item['protein_filename']
            protein_path = os.path.join(protein_root, protein_rel)
            if not os.path.exists(protein_path):
                # fallback for test_set-style receptor files
                protein_rel_alt = _protein_rel_to_rec_rel(protein_rel)
                protein_path_alt = os.path.join(protein_root, protein_rel_alt)
                protein_path = protein_path_alt if os.path.exists(protein_path_alt) else protein_path

            pairs.append((os.path.join(root, f), protein_path))

    lmdb_env.close()
    if skipped:
        print(f"LMDB mapping skipped {skipped} SDF files (no valid mapping)")
    return pairs


def collect_sdf_pairs_from_split_by_name(
    sdf_dir, gen_info_csv, split_by_name_path, test_set_root
):
    """
    Build (sdf_path, protein_path) from:
      - gen_info.csv (filename -> csd_xxxxxx data_id)
      - split_by_name.pt['test'] ordered list
      - test_set root containing family folders
    """
    if not gen_info_csv or not os.path.exists(gen_info_csv):
        raise FileNotFoundError("split_by_name mode requires --gen_info")
    if not os.path.exists(split_by_name_path):
        raise FileNotFoundError(f"split_by_name file not found: {split_by_name_path}")
    if not os.path.exists(test_set_root):
        raise FileNotFoundError(f"test_set root not found: {test_set_root}")

    df = pd.read_csv(gen_info_csv)
    if 'filename' not in df.columns or 'data_id' not in df.columns:
        raise ValueError("--gen_info must contain columns: filename,data_id")
    fn2id = df.set_index('filename')['data_id'].astype(str).to_dict()

    split_obj = torch.load(split_by_name_path)
    if 'test' not in split_obj:
        raise KeyError(f"'test' key not found in {split_by_name_path}")
    split_test = split_obj['test']

    pairs = []
    skipped = 0
    for root, _, files in os.walk(sdf_dir):
        for f in files:
            if not f.endswith('.sdf'):
                continue
            if any(x in f for x in ['-all.sdf', '-in.sdf', '-out.sdf', '-raw.sdf']):
                continue
            data_id = fn2id.get(f, fn2id.get(f.replace('.sdf', '')))
            if data_id is None or not data_id.startswith('csd_'):
                skipped += 1
                continue
            try:
                order = int(data_id.split('_')[1]) - 100000
            except Exception:
                skipped += 1
                continue
            if order < 0 or order >= len(split_test):
                skipped += 1
                continue

            protein_rel, _ = split_test[order]
            protein_rel = _protein_rel_to_rec_rel(protein_rel)
            protein_path = os.path.join(test_set_root, protein_rel)
            pairs.append((os.path.join(root, f), protein_path))

    if skipped:
        print(f"split_by_name mapping skipped {skipped} SDF files (no valid mapping)")
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Vina docking evaluation from SDF')
    parser.add_argument('--sdf_dir', type=str, required=True, help='Directory with SDF files')
    parser.add_argument('--protein_dir', type=str, default=None,
                        help='Directory with receptor PDB files (omit if --protein_path)')
    parser.add_argument('--protein_path', type=str, default=None,
                        help='Single receptor PDB for all SDFs (overrides protein_dir mapping)')
    parser.add_argument('--out_csv', type=str, default=None, help='Output CSV path (default: sdf_dir/vina.csv)')
    parser.add_argument('--mapping_csv', type=str, default=None,
                        help='CSV with sdf_name, protein_name or filename, data_id')
    parser.add_argument('--gen_info', type=str, default=None,
                        help='Path to gen_info.csv (filename, data_id) for mapping')
    parser.add_argument('--lmdb_path', type=str, default=None,
                        help='Use LMDB mapping mode (CrossDocked processed lmdb)')
    parser.add_argument('--split_path', type=str, default=None,
                        help='Split file with test indices (e.g. crossdocked_pocket10_pose_split.pt)')
    parser.add_argument('--split_key', type=str, default='test',
                        help='Split key in split file (default: test)')
    parser.add_argument('--protein_root', type=str, default=None,
                        help='Protein root for LMDB protein_filename resolution')
    parser.add_argument('--split_by_name_path', type=str, default=None,
                        help='Use split_by_name.pt test list for mapping')
    parser.add_argument('--test_set_root', type=str, default=None,
                        help='Root folder of test_set used with --split_by_name_path')
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--n_workers', type=int, default=1, help='Parallel workers (1=sequential)')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['all', 'score_only', 'minimize', 'dock'],
                        help='Docking mode: all (default) is slowest; score_only is fastest')
    parser.add_argument('--max_mols', type=int, default=0,
                        help='If >0, only evaluate first N SDFs after mapping')
    parser.add_argument('--quiet', action='store_true', help='Only print short error (no verbose meeko output)')
    args = parser.parse_args()

    if args.split_by_name_path:
        test_set_root = args.test_set_root or args.protein_root or args.protein_dir
        if not test_set_root:
            print("Error: split_by_name mode needs --test_set_root (or --protein_root / --protein_dir)")
            return 1
        pairs = collect_sdf_pairs_from_split_by_name(
            sdf_dir=args.sdf_dir,
            gen_info_csv=args.gen_info,
            split_by_name_path=args.split_by_name_path,
            test_set_root=test_set_root,
        )
    elif args.lmdb_path:
        protein_root = args.protein_root or args.protein_dir
        if not protein_root:
            print("Error: LMDB mode needs --protein_root (or --protein_dir)")
            return 1
        if not args.split_path:
            print("Error: LMDB mode needs --split_path")
            return 1
        pairs = collect_sdf_pairs_from_lmdb(
            sdf_dir=args.sdf_dir,
            gen_info_csv=args.gen_info,
            lmdb_path=args.lmdb_path,
            split_path=args.split_path,
            protein_root=protein_root,
            split_key=args.split_key,
        )
    elif args.protein_path:
        if not os.path.exists(args.protein_path):
            print(f"Error: --protein_path {args.protein_path} not found")
            return 1
        pairs = []
        for root, _, files in os.walk(args.sdf_dir):
            for f in files:
                if f.endswith('.sdf') and '-all' not in f and '-in' not in f and '-out' not in f:
                    pairs.append((os.path.join(root, f), args.protein_path))
    else:
        if not args.protein_dir:
            print("Error: need --protein_dir or --protein_path")
            return 1
        pairs = collect_sdf_pairs(args.sdf_dir, args.protein_dir,
                                  mapping_csv=args.mapping_csv,
                                  gen_info_csv=args.gen_info)
    if not pairs:
        print("No SDF-protein pairs found. Check --sdf_dir, --protein_dir, and mapping.")
        return 1
    if args.max_mols and args.max_mols > 0:
        pairs = pairs[:args.max_mols]
        print(f"Using first {len(pairs)} pairs (--max_mols)")

    # Skip pairs where protein file does not exist
    valid_pairs = [(s, p) for s, p in pairs if os.path.exists(p)]
    skipped = len(pairs) - len(valid_pairs)
    if skipped:
        print(f"Skipping {skipped} pairs (protein not found)")
    pairs = valid_pairs
    if not pairs:
        print("No valid SDF-protein pairs (all proteins missing)")
        return 1

    print(f"Found {len(pairs)} SDF-protein pairs")

    tmp_base = tempfile.mkdtemp(prefix='vina_eval_')
    rec_cache = os.path.join(tmp_base, 'receptor_cache')
    os.makedirs(rec_cache, exist_ok=True)

    # Pre-prepare receptors once per unique protein to avoid repeated work.
    unique_proteins = sorted({p for _, p in pairs})
    failed_proteins = set()
    prep_iter = unique_proteins
    if HAS_TQDM:
        prep_iter = tqdm(unique_proteins, desc='Preparing receptors')
    for protein_path in prep_iter:
        if find_existing_receptor_pdbqt(protein_path, rec_cache_dir=rec_cache):
            continue
        rec_stem = os.path.splitext(os.path.basename(protein_path))[0]
        candidates = [
            os.path.join(rec_cache, rec_stem + '.pdbqt'),
            os.path.join(rec_cache, rec_stem + '_rigid.pdbqt'),
        ]
        if any(os.path.exists(c) for c in candidates):
            continue
        try:
            prepare_receptor_pdbqt(
                protein_path,
                os.path.join(rec_cache, rec_stem + '.pdbqt'),
                work_dir=rec_cache,
            )
        except Exception as e:
            failed_proteins.add(protein_path)
            err_msg = str(e).split('\n')[0]
            if args.quiet and len(err_msg) > 120:
                err_msg = err_msg[:117] + "..."
            print(f"Skip receptor {os.path.basename(protein_path)}: {err_msg}")

    if failed_proteins:
        before = len(pairs)
        pairs = [(s, p) for s, p in pairs if p not in failed_proteins]
        print(f"Filtered {before - len(pairs)} pairs due to receptor prep failure")
        if not pairs:
            print("No pairs left after receptor preparation filtering")
            return 1

    if args.n_workers <= 1:
        results = []
        it = pairs
        if HAS_TQDM:
            it = tqdm(it, desc='Docking')
        for sdf_path, prot_path in it:
            try:
                res = dock_one(sdf_path, prot_path, center=None, box_size=None,
                              exhaustiveness=args.exhaustiveness, tmp_dir=tmp_base,
                              rec_cache_dir=rec_cache, mode=args.mode)
                results.append({'filename': os.path.basename(sdf_path), **res})
            except Exception as e:
                import numpy as np
                err_msg = str(e)
                if args.quiet and len(err_msg) > 120:
                    err_msg = err_msg[:117] + "..."
                print(f"Error {os.path.basename(sdf_path)}: {err_msg}")
                results.append({'filename': os.path.basename(sdf_path),
                               'vina_score': np.nan, 'vina_min': np.nan, 'vina_dock': np.nan})
    else:
        import numpy as np
        task_args = [(s, p, args.exhaustiveness, tmp_base, rec_cache, args.mode) for s, p in pairs]
        with Pool(args.n_workers) as pool:
            results = list(tqdm(pool.imap_unordered(_wrap_dock, task_args), total=len(pairs))) if HAS_TQDM else pool.map(_wrap_dock, task_args)

    df = pd.DataFrame(results)
    if args.out_csv:
        out_path = args.out_csv
    elif os.path.basename(args.sdf_dir.rstrip('/')) == 'SDF':
        out_path = os.path.join(os.path.dirname(args.sdf_dir.rstrip('/')), 'vina.csv')
    else:
        out_path = os.path.join(args.sdf_dir, 'vina.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} results to {out_path}")

    # Summary (mode-aware valid column)
    metric_col = {
        'all': 'vina_dock',
        'dock': 'vina_dock',
        'score_only': 'vina_score',
        'minimize': 'vina_min',
    }[args.mode]
    valid = df[metric_col].notna()
    if valid.any():
        print(f"  {metric_col} mean: {df.loc[valid, metric_col].mean():.2f}")
        print(f"  {metric_col} median: {df.loc[valid, metric_col].median():.2f}")
    print(f"  Failed: {(~valid).sum()}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
