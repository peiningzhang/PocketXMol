import argparse
import json
import os
import re
from collections import Counter, defaultdict

import pandas as pd
from rdkit import Chem


VALENCE_RE = re.compile(
    r"Explicit valence for atom # (?P<atom_idx>\d+) (?P<element>[A-Za-z]+), (?P<valence>\d+), is greater than permitted"
)


def _list_step_dirs(gen_root, sample_steps):
    step_dirs = []
    for step in sample_steps:
        step_dir = os.path.join(gen_root, f"steps_{step}")
        if not os.path.isdir(step_dir):
            raise FileNotFoundError(f"Missing step dir: {step_dir}")
        step_dirs.append((int(step), step_dir))
    return step_dirs


def _parse_sanitize_failure(path):
    mol = Chem.MolFromMolFile(path, sanitize=False, removeHs=False)
    if mol is None:
        return {
            "error_family": "parse_none",
            "error_message": "MolFromMolFile returned None",
            "valence_element": None,
            "valence_value": None,
            "atom_symbols": [],
            "bond_type_counts": {},
            "num_atoms": None,
            "num_bonds": None,
        }

    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    bond_type_counts = Counter(str(int(bond.GetBondTypeAsDouble())) for bond in mol.GetBonds())
    num_atoms = int(mol.GetNumAtoms())
    num_bonds = int(mol.GetNumBonds())

    try:
        Chem.SanitizeMol(mol)
        return {
            "error_family": "sanitize_ok",
            "error_message": "sanitize_ok",
            "valence_element": None,
            "valence_value": None,
            "atom_symbols": atom_symbols,
            "bond_type_counts": dict(bond_type_counts),
            "num_atoms": num_atoms,
            "num_bonds": num_bonds,
        }
    except Exception as exc:
        msg = str(exc).split("\n")[0]
        match = VALENCE_RE.search(msg)
        valence_element = None
        valence_value = None
        if match:
            error_family = "explicit_valence"
            valence_element = match.group("element")
            valence_value = int(match.group("valence"))
        elif "Can't kekulize mol" in msg:
            error_family = "kekulize"
        else:
            error_family = "other"
        return {
            "error_family": error_family,
            "error_message": msg,
            "valence_element": valence_element,
            "valence_value": valence_value,
            "atom_symbols": atom_symbols,
            "bond_type_counts": dict(bond_type_counts),
            "num_atoms": num_atoms,
            "num_bonds": num_bonds,
        }


def _safe_mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def _diagnose_step(step_dir):
    info_path = os.path.join(step_dir, "gen_info.csv")
    sdf_dir = os.path.join(step_dir, "SDF")
    df = pd.read_csv(info_path)
    df["tag"] = df["tag"].fillna("")
    df_bad = df[df["tag"] == "bad"].copy()

    rows = []
    family_counter = Counter()
    message_counter = Counter()
    valence_element_counter = Counter()
    valence_value_counter = Counter()
    bond_type_counter = Counter()
    atom_symbol_counter = Counter()

    for _, row in df_bad.iterrows():
        path = os.path.join(sdf_dir, row["filename"])
        parsed = _parse_sanitize_failure(path)
        family_counter.update([parsed["error_family"]])
        message_counter.update([parsed["error_message"]])
        if parsed["valence_element"] is not None:
            valence_element_counter.update([parsed["valence_element"]])
        if parsed["valence_value"] is not None:
            valence_value_counter.update([str(parsed["valence_value"])])
        bond_type_counter.update(parsed["bond_type_counts"])
        atom_symbol_counter.update(parsed["atom_symbols"])
        rows.append(
            {
                "filename": row["filename"],
                "data_id": row["data_id"],
                "cfd_node": row.get("cfd_node"),
                "cfd_edge": row.get("cfd_edge"),
                "error_family": parsed["error_family"],
                "error_message": parsed["error_message"],
                "valence_element": parsed["valence_element"],
                "valence_value": parsed["valence_value"],
                "num_atoms": parsed["num_atoms"],
                "num_bonds": parsed["num_bonds"],
            }
        )

    df_rows = pd.DataFrame(rows)
    summary = {
        "num_bad": int(len(df_bad)),
        "error_family_counts": dict(family_counter),
        "top_error_messages": [{"message": k, "count": int(v)} for k, v in message_counter.most_common(20)],
        "valence_element_counts": dict(valence_element_counter),
        "valence_value_counts": dict(valence_value_counter),
        "bond_type_counts_in_bad_sdf": dict(bond_type_counter),
        "atom_symbol_counts_in_bad_sdf": dict(atom_symbol_counter),
        "bad_cfd_node_mean": _safe_mean(df_rows["cfd_node"].tolist()) if len(df_rows) else None,
        "bad_cfd_edge_mean": _safe_mean(df_rows["cfd_edge"].tolist()) if len(df_rows) else None,
        "bad_num_atoms_mean": _safe_mean(df_rows["num_atoms"].tolist()) if len(df_rows) else None,
        "bad_num_bonds_mean": _safe_mean(df_rows["num_bonds"].tolist()) if len(df_rows) else None,
    }
    return summary, df_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_root", required=True)
    parser.add_argument("--sample_steps", type=int, nargs="+", required=True)
    parser.add_argument("--output_json", default="invalid_sdf_diagnostic.json")
    parser.add_argument("--output_csv", default="invalid_sdf_samples.csv")
    parser.add_argument("--output_summary_csv", default="invalid_sdf_summary.csv")
    args = parser.parse_args()

    step_dirs = _list_step_dirs(args.gen_root, args.sample_steps)
    by_step = {}
    sample_rows = []
    summary_rows = []
    for step, step_dir in step_dirs:
        summary, df_rows = _diagnose_step(step_dir)
        by_step[str(step)] = summary
        if len(df_rows):
            df_rows = df_rows.copy()
            df_rows.insert(0, "sample_steps", int(step))
            sample_rows.append(df_rows)
        summary_rows.append(
            {
                "sample_steps": int(step),
                "num_bad": summary["num_bad"],
                "explicit_valence_count": int(summary["error_family_counts"].get("explicit_valence", 0)),
                "kekulize_count": int(summary["error_family_counts"].get("kekulize", 0)),
                "other_count": int(summary["error_family_counts"].get("other", 0)),
                "parse_none_count": int(summary["error_family_counts"].get("parse_none", 0)),
                "bad_cfd_node_mean": summary["bad_cfd_node_mean"],
                "bad_cfd_edge_mean": summary["bad_cfd_edge_mean"],
                "bad_num_atoms_mean": summary["bad_num_atoms_mean"],
                "bad_num_bonds_mean": summary["bad_num_bonds_mean"],
                "top_valence_element": (
                    max(summary["valence_element_counts"], key=summary["valence_element_counts"].get)
                    if summary["valence_element_counts"]
                    else None
                ),
                "top_valence_element_count": (
                    max(summary["valence_element_counts"].values())
                    if summary["valence_element_counts"]
                    else None
                ),
            }
        )

    json_path = os.path.join(args.gen_root, args.output_json)
    with open(json_path, "w") as f:
        json.dump(by_step, f, indent=2, sort_keys=True)

    samples_path = os.path.join(args.gen_root, args.output_csv)
    if sample_rows:
        pd.concat(sample_rows, ignore_index=True).to_csv(samples_path, index=False)
    else:
        pd.DataFrame().to_csv(samples_path, index=False)

    summary_path = os.path.join(args.gen_root, args.output_summary_csv)
    pd.DataFrame(summary_rows).sort_values("sample_steps").to_csv(summary_path, index=False)

    print(f"Wrote {json_path}")
    print(f"Wrote {samples_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
