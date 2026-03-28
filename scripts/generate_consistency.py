import argparse
import copy
import json
import os
import time

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

import sys

sys.path.append(".")

from models.maskfill import PMAsymDenoiser
from models.sample import seperate_outputs2
from utils.consistency import (
    build_sampling_timestep_schedule,
    build_transitions,
    get_task_noise_cfg,
    infer_train_config_path_from_ckpt,
    get_pos_snr_scale,
    normalize_pred_x0,
    renoise_from_pred_x0,
    sample_prior_states,
)
from utils.dataset import TestTaskDataset
from utils.misc import get_logger, get_new_log_dir, make_config, save_config, seed_all
from utils.reconstruct import MolReconsError, create_sdf_string, reconstruct_from_generated_with_edges
from utils.transforms import Compose, FeaturizeMol, FeaturizePocket, get_transforms


def _build_featurizers(train_config):
    featurizers = [FeaturizeMol(train_config.transforms.featurizer)]
    if "featurizer_pocket" in train_config.transforms:
        featurizers = [FeaturizePocket(train_config.transforms.featurizer_pocket)] + featurizers
    return featurizers


def _build_in_dims(featurizers):
    in_dims = {
        "num_node_types": featurizers[-1].num_node_types,
        "num_edge_types": featurizers[-1].num_edge_types,
    }
    if len(featurizers) == 2:
        in_dims["pocket_in_dim"] = featurizers[0].feature_dim
    return in_dims


def _load_student(model, ckpt_path, use_ema=True, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if "ema_student" in ckpt or "student" in ckpt:
        state_dict = ckpt["ema_student"] if use_ema and "ema_student" in ckpt else ckpt["student"]
        model.load_state_dict(state_dict, strict=True)
    else:
        state_dict = ckpt.get("state_dict", ckpt)
        if any(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
        model.load_state_dict(state_dict, strict=True)
    return ckpt


def _build_test_loader(train_config, task_config, batch_size, num_workers):
    train_cfg_for_transform = copy.deepcopy(train_config)
    for samp_trans in task_config.get("transforms", {}).keys():
        if samp_trans in train_cfg_for_transform.transforms.keys():
            train_cfg_for_transform.transforms.get(samp_trans).update(task_config.transforms.get(samp_trans))

    featurizers = _build_featurizers(train_cfg_for_transform)
    in_dims = _build_in_dims(featurizers)
    task_trans = get_transforms(
        task_config.task.transform,
        mode="test",
        num_node_types=in_dims["num_node_types"],
    )
    if "variable_mol_size" in getattr(task_config, "transforms", {}):
        transform_list = featurizers + [get_transforms(task_config.transforms.variable_mol_size), task_trans]
    else:
        transform_list = featurizers + [task_trans]
    addition_transforms = [get_transforms(tr) for tr in task_config.data.get("transforms", [])]
    transforms = Compose(transform_list + addition_transforms)

    follow_batch = sum([getattr(t, "follow_batch", []) for t in transforms.transforms], [])
    exclude_keys = sum([getattr(t, "exclude_keys", []) for t in transforms.transforms], [])
    test_set = TestTaskDataset(
        task_config.data.dataset,
        task_config.task,
        mode="test",
        split=getattr(task_config.data, "split", None),
        transforms=transforms,
    )
    loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(getattr(train_config.train, "pin_memory", True)),
        follow_batch=follow_batch,
        exclude_keys=exclude_keys,
    )
    return loader, featurizers[-1], in_dims


@torch.no_grad()
def _consistency_sample_batch(batch, model, transitions, total_steps, sample_steps, sigma_max):
    node_state, pos_state, edge_state = sample_prior_states(batch, transitions, sigma_max=sigma_max)
    schedule = build_sampling_timestep_schedule(total_steps, sample_steps)

    outputs = None
    node_batch = batch["node_type_batch"]
    for i, t_cur in enumerate(schedule):
        t_cur_graph = torch.full(
            (batch.num_graphs,),
            int(t_cur),
            dtype=torch.long,
            device=batch["node_pos"].device,
        )
        input_batch = copy.copy(batch)
        input_batch["node_in"] = node_state
        input_batch["pos_in"] = pos_state * get_pos_snr_scale(transitions, t_cur_graph, node_batch)
        input_batch["halfedge_in"] = edge_state
        outputs = model(input_batch)
        pred_x0 = normalize_pred_x0(outputs)

        if i == len(schedule) - 1:
            node_state = pred_x0["pred_node_logits_x0"].argmax(dim=-1)
            edge_state = pred_x0["pred_halfedge_logits_x0"].argmax(dim=-1)
            pos_state = pred_x0["pred_pos_x0"]
        else:
            t_next = schedule[i + 1]
            t_next_graph = torch.full(
                (batch.num_graphs,),
                int(t_next),
                dtype=torch.long,
                device=batch["node_pos"].device,
            )
            node_state, pos_state, edge_state = renoise_from_pred_x0(pred_x0, t_next_graph, batch, transitions)

    batch["node_type"] = node_state
    batch["node_pos"] = pos_state
    batch["halfedge_type"] = edge_state
    return batch, outputs


def _post_process_batch(batch, outputs, featurizer, sdf_dir, i_saved):
    info_keys = ["data_id", "db", "task", "key"]
    data_list = [{k: batch[k][i] for k in info_keys} for i in range(len(batch))]
    generated_list, outputs_list, _ = seperate_outputs2(batch, outputs, trajs=None, off_tqdm=True)

    rows = []
    for i_mol in range(len(generated_list)):
        mol_info = featurizer.decode_output(**generated_list[i_mol])
        mol_info.update(data_list[i_mol])
        tag = ""
        smiles = ""
        try:
            rdmol = reconstruct_from_generated_with_edges(mol_info)
            smiles = Chem.MolToSmiles(rdmol)
            if "." in smiles:
                tag = "incomp"
        except MolReconsError:
            tag = "bad"
            rdmol = create_sdf_string(mol_info)

        filename = str(i_saved) + (f"-{tag}" if tag else "") + ".sdf"
        if tag != "bad":
            Chem.MolToMolFile(rdmol, os.path.join(sdf_dir, filename))
        else:
            with open(os.path.join(sdf_dir, filename), "w+") as f:
                f.write(rdmol)

        output = outputs_list[i_mol]
        cfd_pos = float(output["confidence_pos"].mean().cpu()) if "confidence_pos" in output else float("nan")
        cfd_node = float(output["confidence_node"].mean().cpu()) if "confidence_node" in output else float("nan")
        cfd_edge = float(output["confidence_halfedge"].mean().cpu()) if "confidence_halfedge" in output else float("nan")
        rows.append(
            {
                **{k: mol_info[k] for k in info_keys},
                "smiles": smiles,
                "tag": tag,
                "filename": filename,
                "cfd_traj": cfd_pos,
                "cfd_pos": cfd_pos,
                "cfd_node": cfd_node,
                "cfd_edge": cfd_edge,
            }
        )
        i_saved += 1
    return rows, i_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_distill", type=str, default="configs/distill/distill_pxm.yaml")
    parser.add_argument("--consistency_ckpt", type=str, default="")
    parser.add_argument("--config_task", type=str, default="configs/sample/test/sbdd_csd/simple.yml")
    parser.add_argument("--config_model", type=str, default="configs/sample/pxm.yml")
    parser.add_argument("--sample_steps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--num_mols", type=int, default=1000)
    parser.add_argument("--num_repeats", type=int, default=0, help="0 means auto from config/sample or unlimited until num_mols reached")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--outdir", type=str, default="outputs_consistency_gen")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--use_ema", dest="use_ema", action="store_true")
    parser.add_argument("--no_use_ema", dest="use_ema", action="store_false")
    parser.set_defaults(use_ema=True)
    args = parser.parse_args()

    seed_all(args.seed)
    distill_cfg_all = make_config(args.config_distill)
    distill_cfg = distill_cfg_all.distill
    ckpt_path = args.consistency_ckpt or getattr(distill_cfg, "eval_checkpoint", "")
    if not ckpt_path:
        raise ValueError("Please provide --consistency_ckpt or distill.eval_checkpoint")

    teacher_train_cfg_path = getattr(distill_cfg, "teacher_train_config", "")
    if not teacher_train_cfg_path:
        teacher_train_cfg_path = infer_train_config_path_from_ckpt(distill_cfg.teacher_checkpoint)
    teacher_train_cfg = make_config(teacher_train_cfg_path)

    task_cfg = make_config(args.config_task, args.config_model)
    batch_size = args.batch_size if args.batch_size > 0 else int(getattr(task_cfg.sample, "batch_size", 32))
    loader, featurizer, in_dims = _build_test_loader(teacher_train_cfg, task_cfg, batch_size, args.num_workers)
    configured_repeats = int(getattr(task_cfg.sample, "num_repeats", 1))
    if args.num_repeats > 0:
        max_repeats = args.num_repeats
    else:
        # If config has num_repeats, honor it; otherwise keep generating until num_mols.
        max_repeats = configured_repeats if configured_repeats > 0 else int(1e9)

    device = torch.device(args.device)
    model = PMAsymDenoiser(config=teacher_train_cfg.model, **in_dims).to(device)
    _load_student(model, ckpt_path, use_ema=args.use_ema, map_location=device)
    model.eval()

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

    os.makedirs(args.outdir, exist_ok=True)
    log_dir = get_new_log_dir(args.outdir, prefix="consistency_gen")
    logger = get_logger("consistency_gen", log_dir)
    logger.info(
        "Built %s transition grid with %d steps (sigma_min=%.6g, sigma_max=%.6g)",
        str(getattr(distill_cfg, "schedule", "karras")),
        int(distill_cfg.num_steps),
        float(distill_cfg.karras.sigma_min),
        float(distill_cfg.karras.sigma_max),
    )
    save_config(distill_cfg_all, os.path.join(log_dir, "distill_config.yml"))
    save_config(task_cfg, os.path.join(log_dir, "task_config.yml"))

    summary_rows = []
    for sample_steps in args.sample_steps:
        step_dir = os.path.join(log_dir, f"steps_{sample_steps}")
        sdf_dir = os.path.join(step_dir, "SDF")
        os.makedirs(sdf_dir, exist_ok=True)
        logger.info("Generating %d-step consistency samples...", sample_steps)

        i_saved = 0
        all_rows = []
        time_start = time.time()
        i_repeat = 0
        while i_saved < args.num_mols and i_repeat < max_repeats:
            for batch in tqdm(loader, desc=f"Sampling {sample_steps}-step (repeat {i_repeat})"):
                if i_saved >= args.num_mols:
                    break
                batch = batch.to(device)
                batch, outputs = _consistency_sample_batch(
                    batch=batch,
                    model=model,
                    transitions=transitions,
                    total_steps=int(distill_cfg.num_steps),
                    sample_steps=int(sample_steps),
                    sigma_max=float(distill_cfg.karras.sigma_max),
                )
                rows, i_saved = _post_process_batch(batch, outputs, featurizer, sdf_dir, i_saved)
                for row in rows:
                    row["i_repeat"] = i_repeat
                all_rows.extend(rows)
            i_repeat += 1
        wall_time = time.time() - time_start

        df_info = pd.DataFrame(all_rows)
        df_info.to_csv(os.path.join(step_dir, "gen_info.csv"), index=False)
        meta = {
            "sample_steps": int(sample_steps),
            "num_samples": int(len(df_info)),
            "num_repeats_run": int(i_repeat),
            "wall_time_sec": float(wall_time),
            "samples_per_sec": float(len(df_info) / max(wall_time, 1e-8)),
        }
        with open(os.path.join(step_dir, "generation_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        summary_rows.append(meta)
        if i_saved < args.num_mols:
            logger.warning(
                "Requested num_mols=%d but only generated %d after %d repeats.",
                args.num_mols, i_saved, i_repeat
            )
        logger.info("Saved %d molecules to %s", len(df_info), step_dir)

    df_summary = pd.DataFrame(summary_rows).sort_values("sample_steps")
    df_summary.to_csv(os.path.join(log_dir, "generation_summary.csv"), index=False)
    logger.info("Done. Generation summary saved to %s", os.path.join(log_dir, "generation_summary.csv"))


if __name__ == "__main__":
    main()
