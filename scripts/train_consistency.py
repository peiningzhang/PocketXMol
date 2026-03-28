import argparse
import copy
import gc
import os
import subprocess
from datetime import datetime
def cycle(iterable):
    while True:
        for x in iterable:
            yield x

import torch
import torch.nn.functional as F
from easydict import EasyDict
from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Batch as GeoBatch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

import sys

sys.path.append(".")

from models.consistency_loss import MixedStateConsistencyLoss
from models.diffusion import log_sample_categorical
from models.maskfill import PMAsymDenoiser
from utils.consistency import (
    build_transitions,
    ema_update_,
    get_task_noise_cfg,
    infer_train_config_path_from_ckpt,
    normalize_pred_x0,
    get_pos_snr_scale,
    to_plain_dict,
)
from utils.dataset import ForeverTaskDataset
from utils.misc import get_logger, get_new_log_dir, make_config, save_config, seed_all
from utils.transforms import Compose, FeaturizeMol, FeaturizePocket, get_transforms


def _load_model_checkpoint(model, ckpt_path, map_location="cpu"):
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(state_dict, strict=True)
    return ckpt


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


def _apply_data_override(train_config, distill_cfg, task_name, logger):
    data_cfg = copy.deepcopy(train_config.data)
    override = getattr(distill_cfg, "data_override", None)
    if override is None:
        return data_cfg

    if isinstance(override, dict):
        override = EasyDict(override)

    if "dataset" in override:
        for key, value in override.dataset.items():
            data_cfg.dataset[key] = value
    else:
        if "root" in override:
            data_cfg.dataset.root = override.root
        if "assembly_path" in override:
            data_cfg.dataset.assembly_path = override.assembly_path
        if "dbs" in override:
            data_cfg.dataset.dbs = override.dbs

    if "task_db_weights" in override:
        data_cfg.task_db_weights = override.task_db_weights
    elif "db_ratio" in override:
        if task_name not in data_cfg.task_db_weights:
            raise KeyError(f"task_name={task_name} not found in task_db_weights.")
        data_cfg.task_db_weights[task_name].db_ratio = override.db_ratio

    logger.info(
        "Applied data_override: root=%s assembly=%s",
        data_cfg.dataset.root,
        data_cfg.dataset.assembly_path,
    )
    return data_cfg


def _check_assembly_exists(dataset_cfg, mode="train"):
    root = dataset_cfg.root
    assembly = dataset_cfg.assembly_path
    base = os.path.join(root, assembly)
    if base.endswith(".lmdb"):
        # ForeverTaskDataset appends _train/_val/_test automatically.
        expect = base.replace(".lmdb", f"_{mode}.lmdb")
    else:
        expect = base
    if not os.path.exists(expect):
        raise FileNotFoundError(
            f"Assembly file not found: {expect}. "
            "Please set distill.data_override.root / assembly_path in config."
        )


def _normalize_single_task_weight(task_entry, task_name: str):
    """ForeverTaskDataset expects task-level probabilities to sum to 1."""
    task_entry = copy.deepcopy(task_entry)
    task_entry["weight"] = 1.0
    db_ratio = dict(task_entry.get("db_ratio", {}))
    if len(db_ratio) == 0:
        raise ValueError(f"Task {task_name} has empty db_ratio.")
    ratio_sum = float(sum(db_ratio.values()))
    if ratio_sum <= 0:
        raise ValueError(f"Task {task_name} has non-positive db_ratio sum: {ratio_sum}")
    task_entry["db_ratio"] = {k: float(v) / ratio_sum for k, v in db_ratio.items()}
    return task_entry


def _build_train_loader(train_config, distill_cfg, logger):
    task_name = distill_cfg.task_name
    data_cfg = _apply_data_override(train_config, distill_cfg, task_name, logger)
    _check_assembly_exists(data_cfg.dataset, mode="train")

    featurizers = _build_featurizers(train_config)
    in_dims = _build_in_dims(featurizers)
    task_cfg = getattr(distill_cfg, "task_transform", None)
    if task_cfg is None:
        task_cfg = EasyDict({"name": task_name})
    elif isinstance(task_cfg, dict) and not hasattr(task_cfg, "name"):
        task_cfg = EasyDict(task_cfg)
    task_trans = get_transforms(task_cfg, mode="train", num_node_types=in_dims["num_node_types"])

    transform_list = featurizers + [task_trans]
    if "cut_peptide" in train_config.transforms:
        transform_list = [get_transforms(train_config.transforms.cut_peptide)] + transform_list
    transforms = Compose(transform_list)

    follow_batch = sum([getattr(t, "follow_batch", []) for t in transforms.transforms], [])
    exclude_keys = sum([getattr(t, "exclude_keys", []) for t in transforms.transforms], [])

    if task_name not in data_cfg.task_db_weights:
        raise KeyError(f"task_name={task_name} not found in train config task_db_weights.")
    task_db_weights = {
        task_name: _normalize_single_task_weight(data_cfg.task_db_weights[task_name], task_name)
    }

    loader_num_workers = int(getattr(distill_cfg.train, "num_workers", 2))
    # ForeverTaskDataset internally uses num_workers to split sampler shards.
    # Keep at least one shard even when DataLoader workers are set to 0.
    sampler_num_workers = max(loader_num_workers, 1)
    if loader_num_workers == 0:
        logger.info("DataLoader num_workers=0, use sampler_num_workers=1 to avoid zero shard.")
    dataset = ForeverTaskDataset(
        data_cfg.dataset,
        task_db_weights,
        mode="train",
        transforms=transforms,
        shuffle=True,
        num_workers=sampler_num_workers,
        global_rank=0,
        world_size=1,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(distill_cfg.train.batch_size),
        num_workers=loader_num_workers,
        pin_memory=bool(getattr(distill_cfg.train, "pin_memory", True)),
        follow_batch=follow_batch,
        exclude_keys=exclude_keys,
        persistent_workers=(
            bool(getattr(distill_cfg.train, "persistent_workers", loader_num_workers > 0))
            if loader_num_workers > 0
            else False
        ),
    )
    logger.info("Built distill train loader for task=%s", task_name)
    return loader, in_dims


def _is_dataloader_shm_error(exc: RuntimeError) -> bool:
    msg = str(exc)
    patterns = [
        "unable to mmap",
        "Cannot allocate memory",
        "Pin memory thread exited unexpectedly",
        "DataLoader worker",
    ]
    return any(p in msg for p in patterns)


def _build_safe_loader(train_config, distill_cfg, logger):
    safe_cfg = copy.deepcopy(distill_cfg)
    safe_cfg.train.num_workers = 0
    safe_cfg.train.pin_memory = False
    safe_cfg.train.persistent_workers = False
    logger.warning(
        "Switching to safe DataLoader: num_workers=0, pin_memory=False, persistent_workers=False"
    )
    return _build_train_loader(train_config, safe_cfg, logger)


def _is_cuda_oom_error(exc: RuntimeError) -> bool:
    return "out of memory" in str(exc).lower()


def _reduce_batch_for_oom(batch):
    num_graphs = int(getattr(batch, "num_graphs", len(batch)))
    if num_graphs <= 1:
        return None

    follow_batch = [k.replace("_batch", "") for k in batch.keys() if k.endswith("_batch")]
    device = batch["node_pos"].device
    batch_cpu = batch.cpu()
    data_list = batch_cpu.to_data_list()
    new_bs = max(num_graphs // 2, 1)
    reduced = GeoBatch.from_data_list(data_list[:new_bs], follow_batch=follow_batch).to(device)
    return reduced


def _wandb_meta_from_run(wandb_run):
    if wandb_run is None:
        return None
    return {
        "run_id": wandb_run.id,
        "name": wandb_run.name,
        "url": wandb_run.url,
        "project": wandb_run.project,
        "entity": wandb_run.entity,
    }


def _infer_wandb_id_from_resume_path(resume_path: str, logger):
    if not resume_path:
        return None
    # resume_path: <exp_dir>/checkpoints/last.pt
    exp_dir = os.path.dirname(os.path.dirname(resume_path))
    wandb_root = os.path.join(exp_dir, "wandb")
    if not os.path.isdir(wandb_root):
        return None
    # Prefer `latest-run` symlink/file if present.
    latest_run = os.path.join(wandb_root, "latest-run")
    candidates = []
    if os.path.exists(latest_run):
        candidates.append(os.path.realpath(latest_run))
    for name in os.listdir(wandb_root):
        path = os.path.join(wandb_root, name)
        if os.path.isdir(path) and name.startswith("run-"):
            candidates.append(path)
    if len(candidates) == 0:
        return None
    # Most recently modified run dir.
    candidates = sorted(set(candidates), key=lambda p: os.path.getmtime(p), reverse=True)
    run_dir = os.path.basename(candidates[0])
    # Format: run-YYYYmmdd_HHMMSS-<run_id>
    if "-" in run_dir:
        run_id = run_dir.rsplit("-", 1)[-1]
        if run_id:
            logger.info("Inferred wandb run id from resume path: %s", run_id)
            return run_id
    return None


def _init_wandb_run(distill_cfg, config, logger, log_dir, resume_ckpt=None, resume_path=""):
    wandb_cfg = getattr(distill_cfg, "wandb", EasyDict())
    use_wandb = bool(getattr(wandb_cfg, "enabled", False))
    if not use_wandb:
        return None

    try:
        import wandb
    except Exception as exc:
        logger.warning("wandb init failed, fallback to no-wandb: %s", exc)
        return None

    run_id = getattr(wandb_cfg, "id", "")
    if (not run_id) and (resume_ckpt is not None):
        run_id = (resume_ckpt.get("wandb") or {}).get("run_id", "")
    if not run_id:
        run_id = _infer_wandb_id_from_resume_path(resume_path, logger)
    run_id = run_id if run_id else None

    init_kwargs = {
        "project": getattr(wandb_cfg, "project", "pocketxmol-consistency"),
        "name": getattr(wandb_cfg, "name", None),
        "config": to_plain_dict(config),
        "dir": log_dir,
    }
    entity = getattr(wandb_cfg, "entity", "")
    if entity:
        init_kwargs["entity"] = entity

    if run_id is not None:
        resume_mode = getattr(wandb_cfg, "resume", "allow")
        init_kwargs.update({"id": run_id, "resume": resume_mode})
        logger.info("Initializing wandb with resume: id=%s mode=%s", run_id, resume_mode)
    else:
        logger.info("Initializing new wandb run.")

    return wandb.init(**init_kwargs)


def _save_distill_ckpt(path, step, student, ema_student, optimizer, config, transitions, wandb_meta=None):
    ckpt = {
        "step": step,
        "student": student.state_dict(),
        "ema_student": ema_student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": to_plain_dict(config),
        "sigmas": transitions["sigmas"].detach().cpu(),
        "betas": transitions["betas"].detach().cpu(),
        "wandb": wandb_meta,
        "saved_at": datetime.now().isoformat(),
    }
    torch.save(ckpt, path)


def _maybe_run_online_eval(config_path, ckpt_path, distill_cfg, logger, wandb_run=None, step=None, device="cuda:0"):
    online_cfg = getattr(distill_cfg, "online_eval", None)
    if online_cfg is None or not getattr(online_cfg, "enabled", False):
        return None

    eval_outdir = getattr(online_cfg, "outdir", "outputs_consistency_online_eval")
    sample_steps = [int(s) for s in getattr(online_cfg, "sample_steps", [4])]
    num_mols = int(getattr(online_cfg, "num_mols", 100))
    config_task = online_cfg.config_task
    test_df_path = getattr(online_cfg, "test_df_path", "data/test/dfs/sbdd_csd.csv")
    protein_root = getattr(online_cfg, "protein_root", "data/csd/files/proteins")
    exhaustiveness = int(getattr(online_cfg, "exhaustiveness", 16))

    cmd = [
        "python3",
        "scripts/eval_consistency.py",
        "--config_distill",
        config_path,
        "--consistency_ckpt",
        ckpt_path,
        "--config_task",
        config_task,
        "--outdir",
        eval_outdir,
        "--num_mols",
        str(num_mols),
        "--device",
        device,
        "--eval_vina",
        "--test_df_path",
        test_df_path,
        "--protein_root",
        protein_root,
        "--exhaustiveness",
        str(exhaustiveness),
        "--sample_steps",
    ] + [str(s) for s in sample_steps]
    logger.info("Run online eval: %s", " ".join(cmd))

    try:
        import glob
        import pandas as pd

        subprocess.run(cmd, check=True)
        summary_csvs = sorted(glob.glob(os.path.join(eval_outdir, "*", "summary.csv")), key=os.path.getmtime)
        if summary_csvs and wandb_run is not None and step is not None:
            df_summary = pd.read_csv(summary_csvs[-1])
            for _, row in df_summary.iterrows():
                sample_step = int(row["sample_steps"])
                metrics = {f"eval_{sample_step}step/{k}": v for k, v in row.to_dict().items()}
                wandb_run.log(metrics, step=step)
            logger.info("Evaluation metrics logged to wandb from %s", summary_csvs[-1])
    except Exception as exc:
        logger.error("Periodic evaluation failed: %s", exc)
        return None
    return True





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/distill/distill_pxm.yaml")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--outdir", type=str, default="outputs_distill")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    config = make_config(args.config)
    distill_cfg = config.distill
    seed_all(int(getattr(distill_cfg.train, "seed", 2024)))

    os.makedirs(args.outdir, exist_ok=True)
    log_dir = get_new_log_dir(args.outdir, prefix="consistency_distill")
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = get_logger("consistency_train", log_dir)
    logger.info("Config: %s", args.config)
    logger.info("Output: %s", log_dir)
    save_config(config, os.path.join(log_dir, os.path.basename(args.config)))

    teacher_ckpt = distill_cfg.teacher_checkpoint
    teacher_train_cfg_path = getattr(distill_cfg, "teacher_train_config", "")
    if not teacher_train_cfg_path:
        teacher_train_cfg_path = infer_train_config_path_from_ckpt(teacher_ckpt)
    teacher_train_cfg = make_config(teacher_train_cfg_path)
    logger.info("Teacher train config: %s", teacher_train_cfg_path)

    train_loader, in_dims = _build_train_loader(teacher_train_cfg, distill_cfg, logger)
    loader_is_safe = False
    train_iter = cycle(train_loader)

    device = torch.device(args.device)
    teacher = PMAsymDenoiser(config=teacher_train_cfg.model, **in_dims).to(device)
    _load_model_checkpoint(teacher, teacher_ckpt, map_location=device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = PMAsymDenoiser(config=teacher_train_cfg.model, **in_dims).to(device)
    init_from = getattr(distill_cfg, "student_init", "teacher")
    if init_from == "teacher":
        _load_model_checkpoint(student, teacher_ckpt, map_location=device)
    elif init_from == "checkpoint":
        _load_model_checkpoint(student, distill_cfg.student_checkpoint, map_location=device)
    elif init_from == "random":
        pass
    else:
        raise ValueError(f"Unknown student_init: {init_from}")

    ema_student = copy.deepcopy(student).to(device)
    ema_student.eval()
    for p in ema_student.parameters():
        p.requires_grad_(False)

    task_noise_cfg = get_task_noise_cfg(teacher_train_cfg.noise, distill_cfg.task_name)
    schedule_type = getattr(distill_cfg, "schedule", "karras")
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
        schedule_type=schedule_type,
    )
    logger.info("Built %s grid with %d steps", schedule_type.capitalize(), int(distill_cfg.num_steps))

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=float(distill_cfg.optimization.lr),
        weight_decay=float(getattr(distill_cfg.optimization, "weight_decay", 0.0)),
        betas=(
            float(getattr(distill_cfg.optimization, "beta1", 0.9)),
            float(getattr(distill_cfg.optimization, "beta2", 0.999)),
        ),
    )

    loss_fn = MixedStateConsistencyLoss(
        pos_weight=float(distill_cfg.loss_weights.pos),
        node_weight=float(distill_cfg.loss_weights.node),
        edge_weight=float(distill_cfg.loss_weights.halfedge),
        physics_weight=float(getattr(distill_cfg.loss_weights, "physics", 0.0)),
    )

    max_steps = int(distill_cfg.train.max_steps)
    grad_clip = float(getattr(distill_cfg.train, "grad_clip", 0.0))
    log_interval = int(getattr(distill_cfg.train, "log_interval", 50))
    ckpt_interval = int(getattr(distill_cfg.train, "ckpt_interval", 1000))
    eval_interval = int(getattr(distill_cfg.train, "eval_interval", 10000))
    ema_decay = float(getattr(distill_cfg, "ema_decay", 0.999))

    start_step = 1
    ckpt = None
    resume_path = args.resume or getattr(distill_cfg, "resume", "")
    if resume_path:
        logger.info("Resuming from %s", resume_path)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["student"], strict=True)
        ema_student.load_state_dict(ckpt.get("ema_student", ckpt["student"]), strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0)) + 1

    wandb_run = _init_wandb_run(
        distill_cfg=distill_cfg,
        config=config,
        logger=logger,
        log_dir=log_dir,
        resume_ckpt=ckpt,
        resume_path=resume_path,
    )

    prog = tqdm(range(start_step, max_steps + 1), desc="Consistency Distill")
    for step in prog:
        student.train()
        batch = None
        while batch is None:
            try:
                batch = next(train_iter).to(device)
            except RuntimeError as exc:
                if _is_dataloader_shm_error(exc) and (not loader_is_safe):
                    logger.warning("DataLoader shared-memory error detected: %s", exc)
                    train_loader, _ = _build_safe_loader(teacher_train_cfg, distill_cfg, logger)
                    train_iter = cycle(train_loader)
                    loader_is_safe = True
                elif _is_cuda_oom_error(exc):
                    logger.warning("CUDA OOM during batch data transfer. Skipping batch.")
                    if wandb_run is not None:
                        wandb_run.log({"train/oom_skip_data": 1}, step=step)
                    optimizer.zero_grad(set_to_none=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raise
        oom_retries = int(getattr(distill_cfg.train, "oom_retries", 8))
        retry_idx = 0
        skip_step = False
        while True:
            try:
                node_batch = batch["node_type_batch"]
                edge_batch = batch["halfedge_type_batch"]
                n_graphs = batch.num_graphs

                t_high_graph = torch.randint(
                    low=1,
                    high=int(distill_cfg.num_steps),
                    size=(n_graphs,),
                    device=device,
                    dtype=torch.long,
                )
                t_low_graph = torch.clamp(t_high_graph - 1, min=0)

                node_high, log_node_high, _ = transitions["node"].add_noise(batch["node_type"], t_high_graph, batch=node_batch)
                edge_high, log_edge_high, _ = transitions["edge"].add_noise(batch["halfedge_type"], t_high_graph, batch=edge_batch)
                pos_high = transitions["pos"].add_noise(batch["node_pos"], t_high_graph, batch=node_batch)
                pos_high_in = pos_high * get_pos_snr_scale(transitions, t_high_graph, node_batch)

                teacher_batch = copy.copy(batch)
                teacher_batch["node_in"] = node_high
                teacher_batch["halfedge_in"] = edge_high
                teacher_batch["pos_in"] = pos_high_in

                with torch.no_grad():
                    teacher_pred = normalize_pred_x0(teacher(teacher_batch))
                    teacher_log_node_x0 = F.log_softmax(teacher_pred["pred_node_logits_x0"], dim=-1)
                    teacher_log_edge_x0 = F.log_softmax(teacher_pred["pred_halfedge_logits_x0"], dim=-1)

                    pos_low_hat = transitions["pos"].get_prev_from_recon(
                        x_t=pos_high,
                        x_recon=teacher_pred["pred_pos_x0"],
                        t=t_high_graph,
                        batch=node_batch,
                    )
                    log_node_low = transitions["node"].q_v_posterior(
                        log_v0=teacher_log_node_x0,
                        log_vt=log_node_high,
                        t=t_high_graph,
                        batch=node_batch,
                        v0_prob=True,
                    )
                    node_low_hat = log_sample_categorical(log_node_low)
                    log_edge_low = transitions["edge"].q_v_posterior(
                        log_v0=teacher_log_edge_x0,
                        log_vt=log_edge_high,
                        t=t_high_graph,
                        batch=edge_batch,
                        v0_prob=True,
                    )
                    edge_low_hat = log_sample_categorical(log_edge_low)

                    pos_low_in = pos_low_hat * get_pos_snr_scale(transitions, t_low_graph, node_batch)
                    ema_batch = copy.copy(batch)
                    ema_batch["node_in"] = node_low_hat
                    ema_batch["halfedge_in"] = edge_low_hat
                    ema_batch["pos_in"] = pos_low_in
                    ema_pred = normalize_pred_x0(ema_student(ema_batch))

                student_batch = copy.copy(batch)
                student_batch["node_in"] = node_high
                student_batch["halfedge_in"] = edge_high
                student_batch["pos_in"] = pos_high_in
                student_pred = normalize_pred_x0(student(student_batch))

                loss_dict = loss_fn(
                    student_pred_x0=student_pred,
                    ema_pred_x0=ema_pred,
                    mask_pos=(batch["fixed_pos"] == 0),
                    mask_node=(batch["fixed_node"] == 0),
                    mask_edge=(batch["fixed_halfedge"] == 0),
                )
                loss = loss_dict["total"]

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.tensor(0.0, device=device)
                if grad_clip > 0:
                    grad_norm = clip_grad_norm_(student.parameters(), grad_clip)
                optimizer.step()
                ema_update_(ema_student, student, decay=ema_decay)
                break
            except RuntimeError as exc:
                if not _is_cuda_oom_error(exc):
                    raise
                optimizer.zero_grad(set_to_none=True)
                try:
                    del teacher_pred, ema_pred, student_pred
                    del loss_dict, loss, pos_low_hat, node_low_hat, edge_low_hat, log_node_low, log_edge_low
                    del node_high, edge_high, pos_high
                except NameError:
                    pass
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                reduced_batch = _reduce_batch_for_oom(batch)
                if (reduced_batch is None) or (retry_idx >= oom_retries):
                    logger.warning(
                        "Step %d skipped after CUDA OOM (retry=%d/%d).",
                        step, retry_idx, oom_retries,
                    )
                    if wandb_run is not None:
                        wandb_run.log({"train/oom_skip": 1}, step=step)
                    skip_step = True
                    break
                retry_idx += 1
                batch = reduced_batch
                logger.warning(
                    "CUDA OOM at step %d, retry %d with reduced num_graphs=%d",
                    step, retry_idx, batch.num_graphs,
                )
        if skip_step:
            try:
                del batch
            except NameError:
                pass
            continue

        scalars = {
            "train/loss_total": float(loss_dict["total"].detach().cpu()),
            "train/loss_pos": float(loss_dict["pos"].detach().cpu()),
            "train/loss_node": float(loss_dict["node_type"].detach().cpu()),
            "train/loss_edge": float(loss_dict["edge_type"].detach().cpu()),
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/grad_norm": float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
        }
        if device.type == "cuda":
            scalars["train/mem_alloc_mb"] = float(torch.cuda.memory_allocated(device) / 1024**2)
            scalars["train/mem_reserv_mb"] = float(torch.cuda.memory_reserved(device) / 1024**2)
        else:
            scalars["train/mem_alloc_mb"] = 0.0
            scalars["train/mem_reserv_mb"] = 0.0
        if wandb_run is not None:
            wandb_run.log(scalars, step=step)
        if step % log_interval == 0:
            logger.info(
                "step=%d total=%.6f pos=%.6f node=%.6f edge=%.6f mem_alloc=%.1fMB",
                step,
                scalars["train/loss_total"],
                scalars["train/loss_pos"],
                scalars["train/loss_node"],
                scalars["train/loss_edge"],
                scalars["train/mem_alloc_mb"],
            )
        prog.set_postfix(
            loss=f"{scalars['train/loss_total']:.4f}",
            alloc_mb=f"{scalars['train/mem_alloc_mb']:.0f}",
            t=step,
        )

        try:
            del teacher_pred, ema_pred, student_pred
            del loss_dict, loss, batch
            del pos_low_hat, node_low_hat, edge_low_hat, log_node_low, log_edge_low
            del node_high, edge_high, pos_high
        except NameError:
            pass
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        if step % ckpt_interval == 0 or step == max_steps:
            step_ckpt = os.path.join(ckpt_dir, f"step_{step}.pt")
            wandb_meta = _wandb_meta_from_run(wandb_run)
            _save_distill_ckpt(
                step_ckpt, step, student, ema_student, optimizer, config, transitions, wandb_meta=wandb_meta
            )
            _save_distill_ckpt(
                os.path.join(ckpt_dir, "last.pt"),
                step,
                student,
                ema_student,
                optimizer,
                config,
                transitions,
                wandb_meta=wandb_meta,
            )
            logger.info("Saved checkpoint to %s", step_ckpt)

        if step % eval_interval == 0 and step > 0:
            eval_ckpt = os.path.join(ckpt_dir, "last.pt")
            _maybe_run_online_eval(
                args.config,
                eval_ckpt,
                distill_cfg,
                logger,
                wandb_run=wandb_run,
                step=step,
                device=args.device,
            )

    if wandb_run is not None:
        wandb_run.finish()
    logger.info("Consistency distillation completed.")


if __name__ == "__main__":
    main()
