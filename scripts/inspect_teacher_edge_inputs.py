import argparse
import copy
import json
import os
from collections import Counter

import pandas as pd
import torch
from tqdm.auto import tqdm

import sys

sys.path.append(".")

from models.maskfill import PMAsymDenoiser
from scripts.eval_consistency import _build_test_loader
from scripts.train_consistency import _load_model_checkpoint
from utils.consistency import (
    build_transitions,
    get_model_pos_input,
    get_task_noise_cfg,
    infer_train_config_path_from_ckpt,
    normalize_pred_x0,
)
from utils.misc import get_logger, make_config, seed_all


def _accumulate(counter, values):
    counter.update(int(x) for x in values.detach().cpu().tolist())


def _counter_to_dict(counter):
    total = sum(counter.values())
    return {
        "counts": {str(k): int(v) for k, v in sorted(counter.items())},
        "fractions": {str(k): float(v / max(total, 1)) for k, v in sorted(counter.items())},
        "total": int(total),
    }


def _safe_ratio(numer, denom):
    return float(numer / denom) if denom > 0 else float("nan")


@torch.no_grad()
def _collect_mode_stats(model, loader, transitions, timestep, num_mols, device, position_input_mode):
    pred_counter = Counter()
    gt_counter = Counter()
    pred_pos_total = 0
    gt_pos_total = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    n_graphs = 0

    for batch in tqdm(loader, desc=f"Teacher {position_input_mode} t={timestep}"):
        if n_graphs >= num_mols:
            break
        batch = batch.to(device)
        node_batch = batch["node_type_batch"]
        edge_batch = batch["halfedge_type_batch"]
        t_graph = torch.full(
            (batch.num_graphs,),
            int(timestep),
            dtype=torch.long,
            device=device,
        )

        node_high, _, _ = transitions["node"].add_noise(batch["node_type"], t_graph, batch=node_batch)
        edge_high, _, _ = transitions["edge"].add_noise(batch["halfedge_type"], t_graph, batch=edge_batch)
        pos_high = transitions["pos"].add_noise(batch["node_pos"], t_graph, batch=node_batch)
        pos_in = get_model_pos_input(
            pos_high,
            transitions,
            t_graph,
            node_batch,
            position_input_mode=position_input_mode,
        )

        teacher_batch = copy.copy(batch)
        teacher_batch["node_in"] = node_high
        teacher_batch["halfedge_in"] = edge_high
        teacher_batch["pos_in"] = pos_in
        outputs = normalize_pred_x0(model(teacher_batch))
        pred_edge = outputs["pred_halfedge_logits_x0"].argmax(dim=-1)
        gt_edge = batch["halfedge_type"]

        pred_is_pos = pred_edge > 0
        gt_is_pos = gt_edge > 0

        tp += int((pred_is_pos & gt_is_pos).sum().item())
        fp += int((pred_is_pos & ~gt_is_pos).sum().item())
        fn += int((~pred_is_pos & gt_is_pos).sum().item())
        tn += int((~pred_is_pos & ~gt_is_pos).sum().item())
        pred_pos_total += int(pred_is_pos.sum().item())
        gt_pos_total += int(gt_is_pos.sum().item())

        _accumulate(pred_counter, pred_edge)
        _accumulate(gt_counter, gt_edge)

        n_graphs += int(batch.num_graphs)

    return {
        "position_input_mode": position_input_mode,
        "timestep": int(timestep),
        "num_graphs": int(n_graphs),
        "pred_edge_type": _counter_to_dict(pred_counter),
        "gt_edge_type": _counter_to_dict(gt_counter),
        "pred_positive_fraction": _safe_ratio(pred_pos_total, tp + fp + fn + tn),
        "gt_positive_fraction": _safe_ratio(gt_pos_total, tp + fp + fn + tn),
        "gt_positive_recall": _safe_ratio(tp, tp + fn),
        "gt_positive_precision": _safe_ratio(tp, tp + fp),
        "gt_nonbond_false_positive_rate": _safe_ratio(fp, fp + tn),
        "confusion": {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },
    }


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_distill", type=str, default="configs/distill/distill_pxm_euler_edgepos20.yaml")
    parser.add_argument("--config_task", type=str, default="configs/sample/test/sbdd_csd/simple.yml")
    parser.add_argument("--config_model", type=str, default="configs/sample/pxm.yml")
    parser.add_argument("--timesteps", type=int, nargs="+", default=[0, 10, 25, 49])
    parser.add_argument("--num_mols", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--outdir", type=str, default="outputs_teacher_edge_input_diagnostics")
    args = parser.parse_args()

    seed_all(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    logger = get_logger("inspect_teacher_edge_inputs", args.outdir)

    distill_cfg_all = make_config(args.config_distill)
    distill_cfg = distill_cfg_all.distill
    teacher_ckpt = distill_cfg.teacher_checkpoint
    teacher_train_cfg_path = getattr(distill_cfg, "teacher_train_config", "")
    if not teacher_train_cfg_path:
        teacher_train_cfg_path = infer_train_config_path_from_ckpt(teacher_ckpt)
    teacher_train_cfg = make_config(teacher_train_cfg_path)

    task_cfg = make_config(args.config_task, args.config_model)
    batch_size = args.batch_size if args.batch_size > 0 else int(getattr(task_cfg.sample, "batch_size", 32))
    loader, _, in_dims = _build_test_loader(teacher_train_cfg, task_cfg, batch_size, args.num_workers)

    device = torch.device(args.device)
    teacher = PMAsymDenoiser(config=teacher_train_cfg.model, **in_dims).to(device)
    _load_model_checkpoint(teacher, teacher_ckpt, map_location=device)
    teacher.eval()

    task_noise_cfg = get_task_noise_cfg(teacher_train_cfg.noise, distill_cfg.task_name)
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
        "Loaded teacher from %s with %d timesteps: %s",
        teacher_ckpt,
        len(args.timesteps),
        args.timesteps,
    )

    rows = []
    by_timestep = {}
    for timestep in args.timesteps:
        per_mode = {}
        for position_input_mode in ("raw", "snr_scaled"):
            stats = _collect_mode_stats(
                model=teacher,
                loader=loader,
                transitions=transitions,
                timestep=int(timestep),
                num_mols=int(args.num_mols),
                device=device,
                position_input_mode=position_input_mode,
            )
            per_mode[position_input_mode] = stats
            rows.append(
                {
                    "timestep": int(timestep),
                    "position_input_mode": position_input_mode,
                    "num_graphs": stats["num_graphs"],
                    "pred_positive_fraction": stats["pred_positive_fraction"],
                    "gt_positive_fraction": stats["gt_positive_fraction"],
                    "gt_positive_recall": stats["gt_positive_recall"],
                    "gt_positive_precision": stats["gt_positive_precision"],
                    "gt_nonbond_false_positive_rate": stats["gt_nonbond_false_positive_rate"],
                }
            )
            logger.info(
                "t=%d mode=%s pred_pos=%.6f gt_pos=%.6f recall=%.6f precision=%.6f fp_rate=%.6f",
                int(timestep),
                position_input_mode,
                stats["pred_positive_fraction"],
                stats["gt_positive_fraction"],
                stats["gt_positive_recall"],
                stats["gt_positive_precision"],
                stats["gt_nonbond_false_positive_rate"],
            )
        by_timestep[str(timestep)] = per_mode

    with open(os.path.join(args.outdir, "teacher_edge_input_diagnostic.json"), "w") as f:
        json.dump(by_timestep, f, indent=2, sort_keys=True)
    pd.DataFrame(rows).sort_values(["timestep", "position_input_mode"]).to_csv(
        os.path.join(args.outdir, "teacher_edge_input_diagnostic.csv"),
        index=False,
    )
    logger.info("Saved teacher edge input diagnostics to %s", args.outdir)


if __name__ == "__main__":
    main()
