import argparse
import json
import os
from collections import Counter

import pandas as pd
import torch
from tqdm.auto import tqdm

import sys

sys.path.append(".")

from scripts.eval_consistency import (
    _build_test_loader,
    _consistency_sample_batch,
    _load_student,
)
from models.maskfill import PMAsymDenoiser
from utils.consistency import (
    add_log_sigma_to_model_config,
    build_transitions,
    get_sampling_update_modes,
    get_task_noise_cfg,
    infer_train_config_path_from_ckpt,
)
from utils.misc import get_logger, make_config, seed_all


def _accumulate(counter, values):
    values = values.detach().cpu().tolist()
    counter.update(int(x) for x in values)


def _counter_to_dict(counter):
    total = sum(counter.values())
    return {
        "counts": {str(k): int(v) for k, v in sorted(counter.items())},
        "fractions": {str(k): float(v / max(total, 1)) for k, v in sorted(counter.items())},
        "total": int(total),
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_distill", type=str, default="configs/distill/distill_pxm_euler.yaml")
    parser.add_argument("--consistency_ckpt", type=str, required=True)
    parser.add_argument("--config_task", type=str, default="configs/sample/test/sbdd_csd/simple.yml")
    parser.add_argument("--config_model", type=str, default="configs/sample/pxm.yml")
    parser.add_argument("--sample_steps", type=int, nargs="+", default=[1, 8, 100])
    parser.add_argument("--num_mols", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--outdir", type=str, default="outputs_edge_diagnostics")
    parser.add_argument("--use_ema", dest="use_ema", action="store_true")
    parser.add_argument("--no_use_ema", dest="use_ema", action="store_false")
    parser.set_defaults(use_ema=True)
    args = parser.parse_args()

    seed_all(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    logger = get_logger("inspect_edge_collapse", args.outdir)

    distill_cfg_all = make_config(args.config_distill)
    distill_cfg = distill_cfg_all.distill
    teacher_train_cfg_path = getattr(distill_cfg, "teacher_train_config", "")
    if not teacher_train_cfg_path:
        teacher_train_cfg_path = infer_train_config_path_from_ckpt(distill_cfg.teacher_checkpoint)
    teacher_train_cfg = make_config(teacher_train_cfg_path)

    task_cfg = make_config(args.config_task, args.config_model)
    batch_size = args.batch_size if args.batch_size > 0 else int(getattr(task_cfg.sample, "batch_size", 32))
    loader, _, in_dims = _build_test_loader(teacher_train_cfg, task_cfg, batch_size, args.num_workers)

    device = torch.device(args.device)
    model_cfg = add_log_sigma_to_model_config(teacher_train_cfg.model)
    model = PMAsymDenoiser(config=model_cfg, **in_dims).to(device)
    _load_student(model, args.consistency_ckpt, use_ema=args.use_ema, map_location=device)
    model.eval()

    task_noise_cfg = get_task_noise_cfg(teacher_train_cfg.noise, distill_cfg.task_name)
    coordinate_update, discrete_update = get_sampling_update_modes(distill_cfg)
    sampling_cfg = getattr(distill_cfg, "sampling", None)
    transitions = build_transitions(
        num_steps=int(distill_cfg.num_steps),
        sigma_min=float(distill_cfg.karras.sigma_min),
        sigma_max=float(distill_cfg.karras.sigma_max),
        rho=float(distill_cfg.karras.rho),
        num_node_types=in_dims["num_node_types"],
        num_edge_types=in_dims["num_edge_types"],
        node_prior_cfg=getattr(task_noise_cfg.prior, "node", None),
        edge_prior_cfg=getattr(task_noise_cfg.prior, "edge", None),
        device=device,
        schedule_type=str(getattr(distill_cfg, "schedule", "karras")),
    )

    logger.info(
        "Loaded model. coordinate_update=%s discrete_update=%s",
        coordinate_update,
        discrete_update,
    )

    rows = []
    by_step = {}
    for sample_steps in args.sample_steps:
        pred_counter = Counter()
        state_counter = Counter()
        n_saved = 0
        n_graphs = 0

        for batch in tqdm(loader, desc=f"Inspect {sample_steps}-step"):
            if n_saved >= args.num_mols:
                break
            batch = batch.to(device)
            batch, outputs = _consistency_sample_batch(
                batch=batch,
                model=model,
                transitions=transitions,
                total_steps=int(distill_cfg.num_steps),
                sample_steps=int(sample_steps),
                sigma_max=float(distill_cfg.karras.sigma_max),
                coordinate_update=coordinate_update,
                discrete_update=discrete_update,
                sampling_cfg=sampling_cfg,
            )

            pred_edge_type = outputs["pred_halfedge"].argmax(dim=-1)
            state_edge_type = batch["halfedge_type"]
            _accumulate(pred_counter, pred_edge_type)
            _accumulate(state_counter, state_edge_type)

            n_graphs += int(batch.num_graphs)
            n_saved += int(batch.num_graphs)

        pred_summary = _counter_to_dict(pred_counter)
        state_summary = _counter_to_dict(state_counter)
        pred_bond_frac = 1.0 - float(pred_summary["fractions"].get("0", 0.0))
        state_bond_frac = 1.0 - float(state_summary["fractions"].get("0", 0.0))

        step_summary = {
            "sample_steps": int(sample_steps),
            "num_graphs": int(n_graphs),
            "pred_edge_type": pred_summary,
            "final_edge_state": state_summary,
            "pred_positive_bond_fraction": float(pred_bond_frac),
            "final_positive_bond_fraction": float(state_bond_frac),
        }
        by_step[str(sample_steps)] = step_summary
        rows.append(
            {
                "sample_steps": int(sample_steps),
                "num_graphs": int(n_graphs),
                "pred_positive_bond_fraction": float(pred_bond_frac),
                "final_positive_bond_fraction": float(state_bond_frac),
                "pred_edge0_fraction": float(pred_summary["fractions"].get("0", 0.0)),
                "final_edge0_fraction": float(state_summary["fractions"].get("0", 0.0)),
            }
        )
        logger.info(
            "step=%d pred_bond_frac=%.6f final_bond_frac=%.6f pred=%s final=%s",
            int(sample_steps),
            float(pred_bond_frac),
            float(state_bond_frac),
            pred_summary["fractions"],
            state_summary["fractions"],
        )

    with open(os.path.join(args.outdir, "edge_diagnostic.json"), "w") as f:
        json.dump(by_step, f, indent=2, sort_keys=True)
    pd.DataFrame(rows).sort_values("sample_steps").to_csv(
        os.path.join(args.outdir, "edge_diagnostic.csv"), index=False
    )
    logger.info("Saved edge diagnostics to %s", args.outdir)


if __name__ == "__main__":
    main()
