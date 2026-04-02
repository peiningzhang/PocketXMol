import glob
import os
import subprocess
import sys

import numpy as np
import torch


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_standard_sdf_eval(
    step_dir,
    split_by_name_path,
    test_set_root,
    docking_mode="vina_score",
    exhaustiveness=16,
    n_workers=1,
    max_mols=0,
    result_path=None,
    save_plot=False,
):
    if result_path is None:
        result_path = os.path.join(step_dir, "SDF", "eval_results_standard")
    os.makedirs(result_path, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(_repo_root(), "evaluate", "evaluate_sdf_standard.py"),
        "--sdf_dir",
        os.path.join(step_dir, "SDF"),
        "--gen_info",
        os.path.join(step_dir, "gen_info.csv"),
        "--split_by_name_path",
        split_by_name_path,
        "--test_set_root",
        test_set_root,
        "--result_path",
        result_path,
        "--docking_mode",
        docking_mode,
        "--exhaustiveness",
        str(exhaustiveness),
        "--n_workers",
        str(n_workers),
    ]
    if max_mols and int(max_mols) > 0:
        cmd.extend(["--max_mols", str(int(max_mols))])
    if save_plot:
        cmd.append("--save_plot")

    subprocess.run(cmd, check=True)
    return result_path


def _latest_metrics_pt(result_path):
    metrics_paths = sorted(glob.glob(os.path.join(result_path, "metrics_sdf_*.pt")))
    if len(metrics_paths) == 0:
        return None
    return metrics_paths[-1]


def summarize_standard_sdf_eval(result_path, prefix="standard"):
    metrics_path = _latest_metrics_pt(result_path)
    if metrics_path is None:
        return {}

    metrics = torch.load(metrics_path, map_location="cpu", weights_only=False)
    summary = {}

    stability = metrics.get("stability", {}) or {}
    for key in ("recon_success", "eval_success", "complete", "no_clashes", "stereo"):
        if key in stability and stability[key] is not None:
            summary[f"{prefix}_{key}"] = float(stability[key])
        else:
            summary[f"{prefix}_{key}"] = float("nan")

    for group_name in ("bond_js", "pair_js"):
        group = metrics.get(group_name, {}) or {}
        for key, value in group.items():
            summary[f"{prefix}_{group_name}_{key}"] = float(value) if value is not None else float("nan")

    atom_js = metrics.get("atom_type_js", None)
    summary[f"{prefix}_atom_type_js"] = float(atom_js) if atom_js is not None else float("nan")

    all_results = metrics.get("all_results", []) or []
    chem_records = [r.get("chem_results") for r in all_results if isinstance(r, dict) and r.get("chem_results") is not None]
    if len(chem_records) > 0:
        for metric_name in ("qed", "sa", "logp"):
            values = [r.get(metric_name) for r in chem_records if r.get(metric_name) is not None and not np.isnan(r.get(metric_name))]
            if len(values) > 0:
                summary[f"{prefix}_{metric_name}_mean"] = float(np.mean(values))
                summary[f"{prefix}_{metric_name}_median"] = float(np.median(values))

    vina_records = [r.get("vina") for r in all_results if isinstance(r, dict) and r.get("vina") is not None]
    if len(vina_records) > 0:
        for mode_key, out_key in (
            ("score_only", "vina_score"),
            ("minimize", "vina_min"),
            ("dock", "vina_dock"),
        ):
            values = []
            for vina in vina_records:
                try:
                    value = vina[mode_key]["affinity"]
                except Exception:
                    value = None
                if value is not None and not np.isnan(value):
                    values.append(value)
            if len(values) > 0:
                summary[f"{prefix}_{out_key}_mean"] = float(np.mean(values))
                summary[f"{prefix}_{out_key}_median"] = float(np.median(values))

    return summary
