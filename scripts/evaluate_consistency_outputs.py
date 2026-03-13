import argparse
import json
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm.auto import tqdm

import sys

sys.path.append(".")

from evaluate.evaluate_mols import evaluate_mol_dict, get_mols_dict_from_gen_path
from utils.docking_vina import VinaDockingTask
from utils.misc import get_logger


def _list_step_dirs(gen_root, sample_steps=None):
    step_dirs = []
    for name in os.listdir(gen_root):
        if not name.startswith("steps_"):
            continue
        try:
            step = int(name.split("_", 1)[1])
        except Exception:
            continue
        if (sample_steps is None) or (step in sample_steps):
            step_dirs.append((step, os.path.join(gen_root, name)))
    step_dirs.sort(key=lambda x: x[0])
    return step_dirs


def _summarize_from_files(step_dir):
    summary = {}
    info_path = os.path.join(step_dir, "gen_info.csv")
    if os.path.exists(info_path):
        df_info = pd.read_csv(info_path)
        tags = df_info["tag"].fillna("")
        total = max(len(tags), 1)
        bad = int((tags == "bad").sum())
        incomp = int((tags == "incomp").sum())
        summary["num_samples"] = int(len(df_info))
        summary["recon_success"] = float((total - bad) / total)
        summary["complete"] = float((total - bad - incomp) / total)
    else:
        summary["num_samples"] = 0
        summary["recon_success"] = float("nan")
        summary["complete"] = float("nan")

    meta_path = os.path.join(step_dir, "generation_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        summary["wall_time_sec"] = float(meta.get("wall_time_sec", float("nan")))
        summary["samples_per_sec"] = float(meta.get("samples_per_sec", float("nan")))
    return summary


def _collect_metric_summary(step_dir):
    out = {}
    metric_path = os.path.join(step_dir, "mol_metric.csv")
    if os.path.exists(metric_path):
        df_metric = pd.read_csv(metric_path, index_col=0)
        if "qed" in df_metric.columns:
            out["qed_mean"] = float(df_metric["qed"].mean())
        if "sa" in df_metric.columns:
            out["sa_mean"] = float(df_metric["sa"].mean())

    validity_path = os.path.join(step_dir, "validity.json")
    if os.path.exists(validity_path):
        with open(validity_path, "r") as f:
            validity = json.load(f)
        for k, v in validity.items():
            out[k] = float(v)
    return out


def _add_ref_protein_dict(mols_dict, gen_path):
    df_gen = pd.read_csv(os.path.join(gen_path, "gen_info.csv"))
    filename2data_id = df_gen[["filename", "data_id"]].copy()
    filename2data_id = filename2data_id.set_index("filename").to_dict()["data_id"]

    dock_inputs_list = []
    for gen_name, mol in mols_dict.items():
        data_id = filename2data_id[gen_name]
        protein_fn = data_id + "_pro.pdb"
        dock_inputs_list.append(
            {
                "filename": gen_name,
                "mol": mol,
                "protein_fn": protein_fn,
            }
        )
    return dock_inputs_list


def _dock_single(inputs):
    filename, mol, protein_fn, protein_root, exhaustiveness = inputs
    try:
        if mol is None:
            raise ValueError(f"{filename} is None")
        if mol.HasSubstructMatch(Chem.MolFromSmarts("[#5]")):
            raise ValueError(f"{filename} contains element B")

        vina_task = VinaDockingTask.from_generated_mol(mol, protein_fn, protein_root=protein_root)
        score_only = vina_task.run(mode="score_only", exhaustiveness=exhaustiveness)[0]
        minimize = vina_task.run(mode="minimize", exhaustiveness=exhaustiveness)[0]
        dock = vina_task.run(mode="dock", exhaustiveness=exhaustiveness)[0]
        vina_scores = {
            "vina_score": score_only["affinity"],
            "vina_min": minimize["affinity"],
            "vina_pose_min": minimize["pose"],
            "vina_dock": dock["affinity"],
            "vina_pose_dock": dock["pose"],
        }
    except Exception:
        vina_scores = {
            "vina_score": np.nan,
            "vina_min": np.nan,
            "vina_pose_min": "",
            "vina_dock": np.nan,
            "vina_pose_dock": "",
        }
    return {"filename": filename, **vina_scores}


def _calc_vina(inputs_list, gen_path, protein_root, exhaustiveness, n_workers):
    tuples = [
        (x["filename"], x["mol"], x["protein_fn"], protein_root, exhaustiveness)
        for x in inputs_list
    ]
    with Pool(n_workers) as p:
        results = list(tqdm(p.imap_unordered(_dock_single, tuples), total=len(tuples), desc="Vina"))
    df_vina = pd.DataFrame(results)
    df_vina.to_csv(os.path.join(gen_path, "vina.csv"), index=False)
    return df_vina


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_root", type=str, required=True, help="Generation root directory containing steps_*")
    parser.add_argument("--sample_steps", type=int, nargs="+", default=None)
    parser.add_argument("--metrics", type=str, nargs="+", default=["drug_chem", "count_prop", "validity"])
    parser.add_argument("--eval_vina", action="store_true")
    parser.add_argument("--protein_root", type=str, default="data/csd/files/proteins")
    parser.add_argument("--exhaustiveness", type=int, default=16)
    parser.add_argument("--vina_workers", type=int, default=16)
    args = parser.parse_args()

    logger = get_logger("consistency_eval_post", args.gen_root)
    step_dirs = _list_step_dirs(args.gen_root, sample_steps=args.sample_steps)
    if len(step_dirs) == 0:
        raise ValueError(f"No steps_* dirs found in {args.gen_root}")

    summary_rows = []
    for sample_steps, step_dir in step_dirs:
        logger.info("Evaluating generated outputs in %s", step_dir)
        mols_dict = get_mols_dict_from_gen_path(step_dir)
        evaluate_mol_dict(mols_dict, metrics_list=args.metrics, metric_path=step_dir)

        row = {"sample_steps": int(sample_steps)}
        row.update(_summarize_from_files(step_dir))
        row.update(_collect_metric_summary(step_dir))

        if args.eval_vina:
            dock_inputs = _add_ref_protein_dict(mols_dict, step_dir)
            df_vina = _calc_vina(
                dock_inputs,
                step_dir,
                protein_root=args.protein_root,
                exhaustiveness=args.exhaustiveness,
                n_workers=args.vina_workers,
            )
            if "vina_score" in df_vina.columns:
                row["vina_score_mean"] = float(df_vina["vina_score"].mean())
            if "vina_min" in df_vina.columns:
                row["vina_min_mean"] = float(df_vina["vina_min"].mean())

        summary_rows.append(row)
        logger.info("Summary@%d-step: %s", sample_steps, row)

    df_summary = pd.DataFrame(summary_rows).sort_values("sample_steps")
    out_path = os.path.join(args.gen_root, "summary.csv")
    df_summary.to_csv(out_path, index=False)
    logger.info("Saved summary to %s", out_path)


if __name__ == "__main__":
    main()

