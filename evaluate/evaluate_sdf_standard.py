#!/usr/bin/env python
"""
SDF-based standard evaluation aligned to IPDiff eval_split style.

This script evaluates reconstructed SDF molecules and mimics eval_split outputs
as closely as possible. Metrics that require trajectory tensors are marked NA.

Output (metrics_*.pt):
  - stability: mol_stable, atm_stable (NA in SDF mode), recon_success, eval_success, complete
  - bond_length: raw bond-length list for JSD
  - all_results: [{mol, smiles, ligand_filename, pred_pos, pred_v, chem_results, vina}]
  - bond_js, pair_js, atom_type_js: JSD metrics

NA fields in SDF mode:
  - mol_stable, atm_stable: require trajectory tensors (pred_ligand_pos_traj, pred_ligand_v_traj).
    Use recon_success / eval_success / complete instead.

Usage (smoke test):
  python evaluate/evaluate_sdf_standard.py \\
    --sdf_dir outputs_test/sbdd_csd/base_pxm_20260304_002527/SDF \\
    --gen_info outputs_test/sbdd_csd/base_pxm_20260304_002527/gen_info.csv \\
    --split_by_name_path ../AliDiff/data/split_by_name.pt \\
    --test_set_root ../AliDiff/data/test_set \\
    --max_mols 50

Usage (full 10k):
  python evaluate/evaluate_sdf_standard.py \\
    --sdf_dir ... --gen_info ... --split_by_name_path ... --test_set_root ... \\
    --n_workers 4
"""

import argparse
import os
import sys
import tempfile
import warnings
from collections import Counter
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.QED import qed
from scipy import spatial as sci_spatial

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from vina import Vina

sys.path.append(".")

from utils.misc import get_logger

# IPDiff bond-length reference (optional; set IPDIFF_ROOT if not default)
IPDIFF_ROOT = os.environ.get("IPDIFF_ROOT", "/shared/healthinfolab/phz24002/IPDiff_main")
from evaluate.evaluate_vina_sdf import (
    collect_sdf_pairs_from_split_by_name,
    find_existing_receptor_pdbqt,
    prepare_receptor_pdbqt,
    sdf_to_pdbqt,
)

# Suppress meeko deprecation spam in batch runs.
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"meeko\..*")

# Atom-type reference from IPDiff.
ATOM_TYPE_DISTRIBUTION = {
    6: 0.6715020339893559,
    7: 0.11703509510732567,
    8: 0.16956379168491933,
    9: 0.01307879304486639,
    15: 0.01113716146426898,
    16: 0.01123926340861198,
    17: 0.006443861300651673,
}

# Bond-length reference imported from IPDiff config (avoid PocketXMol utils collision).
if os.path.exists(IPDIFF_ROOT):
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "ipdiff_bond_cfg",
        os.path.join(IPDIFF_ROOT, "utils", "evaluation", "eval_bond_length_config.py"),
    )
    ipdiff_bond_cfg = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(ipdiff_bond_cfg)
else:
    raise ImportError(
        f"IPDiff bond config not found at {IPDIFF_ROOT}. "
        "Set IPDIFF_ROOT env var to your IPDiff_main path."
    )

BOND_TYPE_MAP = {
    Chem.rdchem.BondType.UNSPECIFIED: 0,
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}


def get_distribution(distances, bins):
    if len(distances) == 0:
        return np.ones(len(bins) + 1) / (len(bins) + 1)
    idx = np.searchsorted(bins, distances)
    counts = np.bincount(idx, minlength=len(bins) + 1).astype(float)
    total = counts.sum()
    if total <= 0:
        return np.ones(len(bins) + 1) / (len(bins) + 1)
    return counts / total


def bond_distance_from_mol(mol):
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    all_dist = []
    for bond in mol.GetBonds():
        s_atom = bond.GetBeginAtom()
        e_atom = bond.GetEndAtom()
        s_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        dist = float(np.linalg.norm(pos[s_idx] - pos[e_idx]))
        bond_cat = BOND_TYPE_MAP.get(bond.GetBondType(), 0)
        all_dist.append(((s_atom.GetAtomicNum(), e_atom.GetAtomicNum(), bond_cat), dist))
    return all_dist


def pair_distance_from_mol(mol):
    conf = mol.GetConformer()
    pos = conf.GetPositions()
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    pair_dist = []
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos[i] - pos[j]))
            pair_dist.append(((atoms[i], atoms[j]), d))
    return pair_dist


def get_bond_length_profile(bond_lengths):
    grouped = {}
    for btype, blen in bond_lengths:
        a1, a2, b = btype
        if a1 > a2:
            a1, a2 = a2, a1
        key = (a1, a2, b)
        grouped.setdefault(key, []).append(blen)
    return {
        k: get_distribution(v, bins=ipdiff_bond_cfg.DISTANCE_BINS)
        for k, v in grouped.items()
    }


def eval_bond_length_profile(profile):
    metrics = {}
    for btype, gt in ipdiff_bond_cfg.EMPIRICAL_DISTRIBUTIONS.items():
        name = f"JSD_{btype[0]}-{btype[1]}|{btype[2]}"
        if btype not in profile:
            metrics[name] = None
        else:
            metrics[name] = float(sci_spatial.distance.jensenshannon(gt, profile[btype]))
    return metrics


def get_pair_length_profile(pair_lengths):
    cc = [d for (t, d) in pair_lengths if t == (6, 6) and d < 2.0]
    all_12 = [d for (_, d) in pair_lengths if d < 12.0]
    return {
        "CC_2A": get_distribution(cc, bins=np.linspace(0, 2, 100)),
        "All_12A": get_distribution(all_12, bins=np.linspace(0, 12, 100)),
    }


def eval_pair_length_profile(profile):
    metrics = {}
    for k, gt in ipdiff_bond_cfg.PAIR_EMPIRICAL_DISTRIBUTIONS.items():
        name = f"JSD_{k}"
        if k not in profile:
            metrics[name] = None
        else:
            metrics[name] = float(sci_spatial.distance.jensenshannon(gt, profile[k]))
    return metrics


def eval_atom_type_distribution(pred_counter):
    total = sum(pred_counter.values())
    if total == 0:
        return None
    pred = np.array([pred_counter[k] / total for k in ATOM_TYPE_DISTRIBUTION.keys()])
    gt = np.array(list(ATOM_TYPE_DISTRIBUTION.values()))
    return float(sci_spatial.distance.jensenshannon(gt, pred))


def safe_get_chem(mol):
    # Local chem metrics to avoid hard dependency on rdkit.six-based SA scorer.
    try:
        from utils.sascorer import compute_sa_score  # optional
        sa_score = float(compute_sa_score(mol))
    except Exception:
        sa_score = np.nan
    try:
        lipinski_score = int(
            (Descriptors.ExactMolWt(mol) < 500)
            + (Lipinski.NumHDonors(mol) <= 5)
            + (Lipinski.NumHAcceptors(mol) <= 10)
            + (Crippen.MolLogP(mol) >= -2 and Crippen.MolLogP(mol) <= 5)
            + (Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10)
        )
    except Exception:
        lipinski_score = 0
    try:
        qed_score = float(qed(mol))
    except Exception:
        qed_score = np.nan
    try:
        logp_score = float(Crippen.MolLogP(mol))
    except Exception:
        logp_score = np.nan
    ring_info = mol.GetRingInfo()
    ring_size = Counter([len(r) for r in ring_info.AtomRings()])
    return {
        "qed": qed_score,
        "sa": sa_score,
        "logp": logp_score,
        "lipinski": lipinski_score,
        "ring_size": ring_size,
    }


def print_dict(d, logger):
    for k, v in d.items():
        if v is None:
            logger.info(f"{k}:\tNA")
        else:
            logger.info(f"{k}:\t{v:.4f}")


def print_ring_ratio(all_ring_sizes, logger):
    if len(all_ring_sizes) == 0:
        logger.info("No molecules available for ring size analysis")
        return
    for ring_size in range(3, 10):
        n_mol = sum(1 for c in all_ring_sizes if ring_size in c)
        logger.info(f"ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}")


def ensure_receptor_pdbqt(protein_path, rec_cache_dir):
    p = find_existing_receptor_pdbqt(protein_path, rec_cache_dir=rec_cache_dir)
    if p is not None:
        return p
    rec_stem = os.path.splitext(os.path.basename(protein_path))[0]
    return prepare_receptor_pdbqt(
        protein_path,
        os.path.join(rec_cache_dir, rec_stem + ".pdbqt"),
        work_dir=rec_cache_dir,
    )


def run_vina_modes(sdf_path, receptor_pdbqt, mode="vina_score", exhaustiveness=16, tmp_dir=None):
    import numpy as _np

    tmp_dir = tmp_dir or tempfile.mkdtemp(prefix="sdf_std_vina_")
    lig_name = os.path.splitext(os.path.basename(sdf_path))[0]
    lig_pdbqt = os.path.join(tmp_dir, lig_name + ".pdbqt")
    mol = sdf_to_pdbqt(sdf_path, lig_pdbqt)
    conf_pos = mol.GetConformer(0).GetPositions()
    center = ((conf_pos.max(0) + conf_pos.min(0)) / 2).tolist()
    box = ((conf_pos.max(0) - conf_pos.min(0)) + 5.0).tolist()

    v = Vina(sf_name="vina", seed=0, verbosity=0)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(lig_pdbqt)
    v.compute_vina_maps(center=center, box_size=box)

    out = {
        "score_only": {"affinity": _np.nan},
        "minimize": {"affinity": _np.nan},
        "dock": {"affinity": _np.nan},
    }
    if mode in ["vina_score", "vina_dock"]:
        out["score_only"]["affinity"] = float(v.score()[0])
        out["minimize"]["affinity"] = float(v.optimize()[0])
    if mode == "vina_dock":
        v.dock(exhaustiveness=exhaustiveness, n_poses=1)
        out["dock"]["affinity"] = float(v.energies(n_poses=1)[0][0])
    return out


def eval_one(sample):
    sdf_path, protein_path, gt_path, docking_mode, exhaustiveness, rec_cache, tmp_base = sample
    result = {
        "ok_recon": False,
        "ok_complete": False,
        "ok_eval": False,
        "no_clashes": None,
        "stereo": None,
        "bond_dist": [],
        "pair_dist": [],
        "atom_types": Counter(),
        "ring_size": Counter(),
        "record": None,
        "error": "",
    }
    try:
        mol_unsanitized = Chem.MolFromMolFile(sdf_path, sanitize=False)
        if mol_unsanitized is not None:
            protein = Chem.MolFromPDBFile(protein_path, sanitize=False, proximityBonding=False)
            if protein is not None:
                try:
                    from utils.buster_tools import check_intermolecular_distance
                    clash_results = check_intermolecular_distance(
                        mol_unsanitized,
                        protein,
                        ignore_types={"hydrogens", "organic_cofactors", "inorganic_cofactors", "waters"},
                        clash_cutoff=0.65 if sdf_path.endswith('.pdb') else 0.75,
                    )
                    result["no_clashes"] = clash_results['results']['no_clashes']
                except Exception:
                    pass
            
            if gt_path and os.path.exists(gt_path):
                try:
                    if gt_path.endswith('.sdf'):
                        mol_true = Chem.MolFromMolFile(gt_path, sanitize=False)
                    elif gt_path.endswith('.pdb'):
                        mol_true = Chem.MolFromPDBFile(gt_path, sanitize=False)
                    else:
                        mol_true = Chem.MolFromSmiles(gt_path)
                    if mol_true is not None:
                        from utils.buster_tools import check_identity
                        tet_results = check_identity(mol_unsanitized, mol_true, inchi_options="w")
                        result["stereo"] = tet_results['results']['stereo']
                except Exception:
                    pass

        mol = Chem.MolFromMolFile(sdf_path)
        if mol is None:
            raise ValueError("MolFromMolFile failed")
        result["ok_recon"] = True
        smiles = Chem.MolToSmiles(mol)
        if "." not in smiles:
            result["ok_complete"] = True

        atom_nums = [a.GetAtomicNum() for a in mol.GetAtoms()]
        result["atom_types"] = Counter(atom_nums)
        result["pair_dist"] = pair_distance_from_mol(mol)
        if result["ok_complete"]:
            result["bond_dist"] = bond_distance_from_mol(mol)

        chem = None
        try:
            chem = safe_get_chem(mol)
            result["ring_size"] = chem.get("ring_size", Counter())
        except Exception:
            chem = None

        vina = None
        if docking_mode != "none":
            rec_pdbqt = ensure_receptor_pdbqt(protein_path, rec_cache)
            vina = run_vina_modes(
                sdf_path=sdf_path,
                receptor_pdbqt=rec_pdbqt,
                mode=docking_mode,
                exhaustiveness=exhaustiveness,
                tmp_dir=tmp_base,
            )

        if chem is not None:
            result["ok_eval"] = True
            result["record"] = {
                "mol": mol,
                "smiles": smiles,
                "ligand_filename": os.path.basename(sdf_path),
                "pred_pos": None,
                "pred_v": None,
                "chem_results": chem,
                "vina": vina,
            }
        else:
            result["record"] = {
                "mol": mol,
                "smiles": smiles,
                "ligand_filename": os.path.basename(sdf_path),
                "pred_pos": None,
                "pred_v": None,
                "chem_results": None,
                "vina": vina,
            }

    except Exception as e:
        result["error"] = str(e)
    return result


def collect_sdf_triplets_from_split_by_name(
    sdf_dir, gen_info_csv, split_by_name_path, test_set_root
):
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

    triplets = []
    skipped = 0
    from evaluate.evaluate_vina_sdf import _protein_rel_to_rec_rel
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

            protein_rel, ligand_rel = split_test[order]
            protein_rel_rec = _protein_rel_to_rec_rel(protein_rel)
            protein_path = os.path.join(test_set_root, protein_rel_rec)
            gt_path = os.path.join(test_set_root, ligand_rel)
            triplets.append((os.path.join(root, f), protein_path, gt_path))

    if skipped:
        print(f"split_by_name mapping skipped {skipped} SDF files (no valid mapping)")
    return triplets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf_dir", type=str, required=True)
    parser.add_argument("--gen_info", type=str, required=True)
    parser.add_argument("--split_by_name_path", type=str, required=True)
    parser.add_argument("--test_set_root", type=str, required=True)
    parser.add_argument("--docking_mode", type=str, default="vina_score",
                        choices=["none", "vina_score", "vina_dock"])
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--max_mols", type=int, default=0)
    parser.add_argument("--eval_start_index", type=int, default=None)
    parser.add_argument("--eval_end_index", type=int, default=None)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--save", type=eval, default=True)
    parser.add_argument("--save_plot", action="store_true", help="Save pair-distance histogram (eval_split style)")
    parser.add_argument("--verbose", type=eval, default=False)
    args = parser.parse_args()

    result_path = args.result_path
    if result_path is None:
        result_path = os.path.join(args.sdf_dir, "eval_results_standard")
    os.makedirs(result_path, exist_ok=True)
    logger = get_logger("evaluate_sdf_standard", log_dir=result_path)

    pairs = collect_sdf_triplets_from_split_by_name(
        sdf_dir=args.sdf_dir,
        gen_info_csv=args.gen_info,
        split_by_name_path=args.split_by_name_path,
        test_set_root=args.test_set_root,
    )
    pairs = [(s, p, g) for s, p, g in pairs if os.path.exists(s) and os.path.exists(p)]
    pairs = sorted(pairs, key=lambda x: os.path.basename(x[0]))

    eval_start = 0 if args.eval_start_index is None else max(0, args.eval_start_index)
    eval_end = len(pairs) - 1 if args.eval_end_index is None else min(len(pairs) - 1, args.eval_end_index)
    if eval_end < eval_start:
        logger.error("Invalid eval range.")
        return 1
    pairs = pairs[eval_start:eval_end + 1]
    if args.max_mols > 0:
        pairs = pairs[:args.max_mols]
    eval_end = min(eval_end, eval_start + len(pairs) - 1)  # actual slice end
    logger.info(f"Loaded {len(pairs)} sdf-protein pairs for evaluation.")

    rec_cache = os.path.join(result_path, "receptor_cache")
    os.makedirs(rec_cache, exist_ok=True)
    tmp_base = tempfile.mkdtemp(prefix="sdf_std_eval_")

    # Pre-check receptors (prefer existing pdbqt).
    failed_proteins = set()
    for prot in tqdm(sorted({p for _, p, _ in pairs}), desc="Preparing receptors"):
        try:
            ensure_receptor_pdbqt(prot, rec_cache)
        except Exception as e:
            failed_proteins.add(prot)
            if args.verbose:
                logger.warning(f"Skip receptor {os.path.basename(prot)}: {e}")
    if failed_proteins:
        before = len(pairs)
        pairs = [(s, p, g) for s, p, g in pairs if p not in failed_proteins]
        logger.info(f"Filtered {before - len(pairs)} pairs due to receptor prep failure.")
    if len(pairs) == 0:
        logger.error("No pairs left after receptor filtering.")
        return 1

    samples = [
        (s, p, g, args.docking_mode, args.exhaustiveness, rec_cache, tmp_base)
        for s, p, g in pairs
    ]

    if args.n_workers > 1:
        with Pool(args.n_workers) as pool:
            out = list(tqdm(pool.imap_unordered(eval_one, samples), total=len(samples), desc="Eval"))
    else:
        out = [eval_one(x) for x in tqdm(samples, desc="Eval")]

    num_samples = len(out)
    n_recon_success = sum(1 for x in out if x["ok_recon"])
    n_complete = sum(1 for x in out if x["ok_complete"])
    n_eval_success = sum(1 for x in out if x["ok_eval"])

    all_bond_dist = []
    all_pair_dist = []
    success_atom_types = Counter()
    ring_sizes = []
    results = []
    for x in out:
        all_bond_dist.extend(x["bond_dist"])
        all_pair_dist.extend(x["pair_dist"])
        success_atom_types.update(x["atom_types"])
        ring_sizes.append(x["ring_size"])
        if x["record"] is not None:
            results.append(x["record"])

    n_has_clash_metric = sum(1 for x in out if x.get("no_clashes") is not None)
    n_no_clashes = sum(1 for x in out if x.get("no_clashes") is True)
    n_has_stereo_metric = sum(1 for x in out if x.get("stereo") is not None)
    n_stereo = sum(1 for x in out if x.get("stereo") is True)

    stability = {
        "mol_stable": None,   # not available from SDF-only mode
        "atm_stable": None,   # not available from SDF-only mode
        "recon_success": n_recon_success / num_samples if num_samples else None,
        "eval_success": n_eval_success / num_samples if num_samples else None,
        "complete": n_complete / num_samples if num_samples else None,
        "no_clashes": n_no_clashes / n_has_clash_metric if n_has_clash_metric else None,
        "stereo": n_stereo / n_has_stereo_metric if n_has_stereo_metric else None,
    }
    logger.info("Trajectory-based stability metrics unavailable in SDF mode -> set to NA.")
    print_dict(stability, logger)

    bond_profile = get_bond_length_profile(all_bond_dist)
    bond_js = eval_bond_length_profile(bond_profile)
    logger.info("JS bond distances:")
    print_dict(bond_js, logger)

    pair_profile = get_pair_length_profile(all_pair_dist)
    pair_js = eval_pair_length_profile(pair_profile)
    logger.info("JS pair distances:")
    print_dict(pair_js, logger)

    atom_js = eval_atom_type_distribution(success_atom_types)
    if atom_js is None:
        logger.info("Atom type JS: NA")
    else:
        logger.info(f"Atom type JS: {atom_js:.4f}")

    if len(results) > 0:
        qed = [r["chem_results"]["qed"] for r in results if r["chem_results"] is not None]
        sa = [r["chem_results"]["sa"] for r in results if r["chem_results"] is not None]
        if len(qed) > 0:
            logger.info(f"QED:   Mean: {np.mean(qed):.3f} Median: {np.median(qed):.3f}")
            logger.info(f"SA:    Mean: {np.mean(sa):.3f} Median: {np.median(sa):.3f}")

        if args.docking_mode in ["vina_score", "vina_dock"]:
            s_only = [r["vina"]["score_only"]["affinity"] for r in results if r["vina"] is not None]
            s_min = [r["vina"]["minimize"]["affinity"] for r in results if r["vina"] is not None]
            if len(s_only) > 0:
                logger.info(f"Vina Score: Mean: {np.nanmean(s_only):.3f} Median: {np.nanmedian(s_only):.3f}")
                logger.info(f"Vina Min:   Mean: {np.nanmean(s_min):.3f} Median: {np.nanmedian(s_min):.3f}")
        if args.docking_mode == "vina_dock":
            s_dock = [r["vina"]["dock"]["affinity"] for r in results if r["vina"] is not None]
            if len(s_dock) > 0:
                logger.info(f"Vina Dock:  Mean: {np.nanmean(s_dock):.3f} Median: {np.nanmedian(s_dock):.3f}")

    print_ring_ratio(ring_sizes, logger)

    if args.save and args.save_plot and len(all_pair_dist) > 0:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            pair_profile = get_pair_length_profile(all_pair_dist)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for idx, (k, gt) in enumerate(ipdiff_bond_cfg.PAIR_EMPIRICAL_DISTRIBUTIONS.items()):
                if k not in pair_profile:
                    continue
                x = ipdiff_bond_cfg.PAIR_EMPIRICAL_BINS[k]
                axes[idx].step(x, np.array(gt)[1:], label="Ref")
                axes[idx].step(x, pair_profile[k][1:], label="Pred")
                jsd = pair_js.get(f"JSD_{k}")
                axes[idx].set_title(f"{k} JSD: {jsd:.4f}" if jsd is not None else k)
                axes[idx].legend()
            plt.tight_layout()
            plot_path = os.path.join(result_path, f"pair_dist_hist_{eval_start}-to-{eval_end}.png")
            plt.savefig(plot_path)
            plt.close()
            logger.info(f"Saved pair-distance histogram to {plot_path}")
        except Exception as e:
            logger.warning(f"Could not save pair-distance plot: {e}")

    if args.save:
        save_name = f"metrics_sdf_{eval_start}-to-{eval_end}.pt"
        torch.save(
            {
                "stability": stability,
                "bond_length": all_bond_dist,
                "all_results": results,
                "bond_js": bond_js,
                "pair_js": pair_js,
                "atom_type_js": atom_js,
            },
            os.path.join(result_path, save_name),
        )
        pd.DataFrame(
            [
                {
                    "filename": os.path.basename(s),
                    "protein": os.path.basename(p),
                    "gt": os.path.basename(g),
                } for s, p, g in pairs
            ]
        ).to_csv(os.path.join(result_path, f"pairs_{eval_start}-to-{eval_end}.csv"), index=False)
        logger.info(f"Saved metrics to {os.path.join(result_path, save_name)}")

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

