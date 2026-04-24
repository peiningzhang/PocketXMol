import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem


def _list_step_dirs(gen_root, sample_steps=None):
    step_dirs = []
    for name in os.listdir(gen_root):
        if not name.startswith("steps_"):
            continue
        try:
            step = int(name.split("_", 1)[1])
        except Exception:
            continue
        if sample_steps is None or step in sample_steps:
            step_dirs.append((step, os.path.join(gen_root, name)))
    step_dirs.sort(key=lambda x: x[0])
    return step_dirs


def _safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def _summarize_numeric(series):
    series = pd.Series(series).dropna()
    if len(series) == 0:
        return {}
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _summarize_counts(counter, top_k=None):
    items = counter.most_common(top_k)
    return [{"value": key, "count": int(value)} for key, value in items]


def _parse_smiles_stats(smiles):
    if not isinstance(smiles, str) or smiles == "":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    frag_sizes = sorted(int(frag.GetNumAtoms()) for frag in frags)
    frag_smiles = sorted(Chem.MolToSmiles(frag) for frag in frags)
    atom_degrees = [int(atom.GetDegree()) for atom in mol.GetAtoms()]
    return {
        "atom_count": int(mol.GetNumAtoms()),
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
        "bond_count": int(mol.GetNumBonds()),
        "ring_count": int(mol.GetRingInfo().NumRings()),
        "fragment_count": int(len(frags)),
        "fragment_size_signature": "-".join(str(x) for x in frag_sizes),
        "fragment_smiles": frag_smiles,
        "largest_fragment_atoms": int(max(frag_sizes)),
        "isolated_atom_fraction": float(sum(x == 0 for x in atom_degrees) / max(len(atom_degrees), 1)),
        "mean_atom_degree": float(np.mean(atom_degrees)) if len(atom_degrees) else 0.0,
    }


def _diagnose_step(step_dir):
    info_path = os.path.join(step_dir, "gen_info.csv")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Missing gen_info.csv in {step_dir}")

    df = pd.read_csv(info_path)
    if "tag" not in df.columns:
        df["tag"] = ""
    df["tag"] = df["tag"].fillna("")
    df["tag_label"] = df["tag"].replace({"": "complete"})

    total = len(df)
    tag_counts = df["tag_label"].value_counts().to_dict()
    bad = int(tag_counts.get("bad", 0))
    incomp = int(tag_counts.get("incomp", 0))
    complete = int(tag_counts.get("complete", 0))

    parsed = []
    for _, row in df.iterrows():
        stats = _parse_smiles_stats(row.get("smiles", ""))
        if stats is None:
            continue
        stats["tag_label"] = row["tag_label"]
        stats["cfd_pos"] = _safe_float(row.get("cfd_pos"))
        stats["cfd_node"] = _safe_float(row.get("cfd_node"))
        stats["cfd_edge"] = _safe_float(row.get("cfd_edge"))
        parsed.append(stats)
    df_parsed = pd.DataFrame(parsed)

    frag_count_dist = Counter()
    frag_size_patterns = Counter()
    frag_species = Counter()
    if len(df_parsed) > 0:
        df_incomp = df_parsed[df_parsed["tag_label"] == "incomp"]
        frag_count_dist.update(df_incomp["fragment_count"].astype(int).tolist())
        frag_size_patterns.update(df_incomp["fragment_size_signature"].tolist())
        for frag_list in df_incomp["fragment_smiles"].tolist():
            frag_species.update(frag_list)
    else:
        df_incomp = pd.DataFrame()

    confidence_by_tag = {}
    for tag_name in ["complete", "incomp", "bad"]:
        df_tag = df[df["tag_label"] == tag_name]
        confidence_by_tag[tag_name] = {
            "count": int(len(df_tag)),
            "cfd_pos_mean": _safe_float(df_tag["cfd_pos"].mean()) if "cfd_pos" in df_tag else None,
            "cfd_node_mean": _safe_float(df_tag["cfd_node"].mean()) if "cfd_node" in df_tag else None,
            "cfd_edge_mean": _safe_float(df_tag["cfd_edge"].mean()) if "cfd_edge" in df_tag else None,
        }

    summary = {
        "num_samples": int(total),
        "recon_success": float((total - bad) / max(total, 1)),
        "complete": float(complete / max(total, 1)),
        "tag_counts": {k: int(v) for k, v in tag_counts.items()},
        "parsed_smiles": int(len(df_parsed)),
        "atom_count": _summarize_numeric(df_parsed["atom_count"]) if len(df_parsed) else {},
        "heavy_atom_count": _summarize_numeric(df_parsed["heavy_atom_count"]) if len(df_parsed) else {},
        "bond_count": _summarize_numeric(df_parsed["bond_count"]) if len(df_parsed) else {},
        "ring_count": _summarize_numeric(df_parsed["ring_count"]) if len(df_parsed) else {},
        "largest_fragment_atoms": _summarize_numeric(df_parsed["largest_fragment_atoms"]) if len(df_parsed) else {},
        "isolated_atom_fraction": _summarize_numeric(df_parsed["isolated_atom_fraction"]) if len(df_parsed) else {},
        "mean_atom_degree": _summarize_numeric(df_parsed["mean_atom_degree"]) if len(df_parsed) else {},
        "incomp_fragment_count_distribution": _summarize_counts(frag_count_dist),
        "incomp_top_fragment_size_patterns": _summarize_counts(frag_size_patterns, top_k=10),
        "incomp_top_fragment_species": _summarize_counts(frag_species, top_k=10),
        "confidence_by_tag": confidence_by_tag,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_root", type=str, required=True)
    parser.add_argument("--sample_steps", type=int, nargs="+", default=None)
    parser.add_argument("--output_json", type=str, default="diagnostic_summary.json")
    parser.add_argument("--output_csv", type=str, default="diagnostic_summary.csv")
    args = parser.parse_args()

    step_dirs = _list_step_dirs(args.gen_root, sample_steps=args.sample_steps)
    if len(step_dirs) == 0:
        raise ValueError(f"No steps_* dirs found in {args.gen_root}")

    summary_by_step = {}
    rows = []
    for step, step_dir in step_dirs:
        summary = _diagnose_step(step_dir)
        summary_by_step[str(step)] = summary
        rows.append(
            {
                "sample_steps": int(step),
                "num_samples": summary["num_samples"],
                "recon_success": summary["recon_success"],
                "complete": summary["complete"],
                "count_complete": int(summary["tag_counts"].get("complete", 0)),
                "count_incomp": int(summary["tag_counts"].get("incomp", 0)),
                "count_bad": int(summary["tag_counts"].get("bad", 0)),
                "parsed_smiles": int(summary["parsed_smiles"]),
                "atom_count_mean": summary["atom_count"].get("mean"),
                "bond_count_mean": summary["bond_count"].get("mean"),
                "ring_count_mean": summary["ring_count"].get("mean"),
                "largest_fragment_atoms_mean": summary["largest_fragment_atoms"].get("mean"),
                "isolated_atom_fraction_mean": summary["isolated_atom_fraction"].get("mean"),
                "mean_atom_degree_mean": summary["mean_atom_degree"].get("mean"),
                "incomp_mode_fragment_count": (
                    summary["incomp_fragment_count_distribution"][0]["value"]
                    if summary["incomp_fragment_count_distribution"]
                    else None
                ),
                "incomp_mode_fragment_count_n": (
                    summary["incomp_fragment_count_distribution"][0]["count"]
                    if summary["incomp_fragment_count_distribution"]
                    else None
                ),
                "incomp_top_fragment_pattern": (
                    summary["incomp_top_fragment_size_patterns"][0]["value"]
                    if summary["incomp_top_fragment_size_patterns"]
                    else None
                ),
                "incomp_top_fragment_species": (
                    summary["incomp_top_fragment_species"][0]["value"]
                    if summary["incomp_top_fragment_species"]
                    else None
                ),
                "complete_cfd_node_mean": summary["confidence_by_tag"]["complete"]["cfd_node_mean"],
                "incomp_cfd_node_mean": summary["confidence_by_tag"]["incomp"]["cfd_node_mean"],
                "complete_cfd_edge_mean": summary["confidence_by_tag"]["complete"]["cfd_edge_mean"],
                "incomp_cfd_edge_mean": summary["confidence_by_tag"]["incomp"]["cfd_edge_mean"],
            }
        )

    json_path = os.path.join(args.gen_root, args.output_json)
    with open(json_path, "w") as f:
        json.dump(summary_by_step, f, indent=2, sort_keys=True)

    csv_path = os.path.join(args.gen_root, args.output_csv)
    pd.DataFrame(rows).sort_values("sample_steps").to_csv(csv_path, index=False)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
