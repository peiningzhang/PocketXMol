import copy
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.transition import ContigousTransition, GeneralCategoricalTransition


def to_plain_dict(obj):
    if isinstance(obj, dict):
        return {k: to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_plain_dict(v) for v in obj]
    return obj


def infer_train_config_path_from_ckpt(ckpt_path: str) -> str:
    cfg_dir = os.path.dirname(ckpt_path).replace("checkpoints", "train_config")
    if not os.path.isdir(cfg_dir):
        raise FileNotFoundError(f"Cannot find train_config directory: {cfg_dir}")
    cfg_files = [f for f in os.listdir(cfg_dir) if f.endswith((".yml", ".yaml"))]
    if not cfg_files:
        raise FileNotFoundError(f"No config file in {cfg_dir}")
    if len(cfg_files) > 1:
        cfg_files = sorted(cfg_files)
    return os.path.join(cfg_dir, cfg_files[0])


def get_task_noise_cfg(noise_cfg, task_name: str):
    if noise_cfg.name == "mixed":
        for item in noise_cfg.individual:
            if item.name == task_name:
                return item
        raise KeyError(f"Task noise config {task_name} not found in mixed noise config.")
    return noise_cfg


def _prior_probs_from_cfg(prior_cfg, num_classes: int) -> np.ndarray:
    if prior_cfg is None:
        return np.ones(num_classes, dtype=np.float64) / num_classes
    prior_type = getattr(prior_cfg, "prior_type", "uniform")
    if prior_type == "uniform":
        probs = np.ones(num_classes, dtype=np.float64) / num_classes
    elif prior_type == "tomask":
        probs = np.zeros(num_classes, dtype=np.float64)
        probs[-1] = 1.0
    elif prior_type == "tomask_half":
        probs = np.ones(num_classes, dtype=np.float64) * (0.5 / max(num_classes - 1, 1))
        probs[-1] = 0.5
    elif prior_type == "predefined":
        probs = np.asarray([float(x) for x in prior_cfg.prior_probs], dtype=np.float64)
        probs = probs / probs.sum()
    else:
        probs = np.ones(num_classes, dtype=np.float64) / num_classes
    return probs


def karras_sigmas(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device: Optional[torch.device] = None,
    ascending: bool = True,
) -> torch.Tensor:
    ramp = torch.linspace(0.0, 1.0, num_steps, device=device)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas_desc = (max_inv + ramp * (min_inv - max_inv)) ** rho
    if ascending:
        return torch.flip(sigmas_desc, dims=[0])
    return sigmas_desc


def log_uniform_sigmas(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    device: Optional[torch.device] = None,
    ascending: bool = True,
) -> torch.Tensor:
    sigmas_desc = torch.exp(
        torch.linspace(math.log(sigma_max), math.log(sigma_min), num_steps, device=device)
    )
    if ascending:
        return torch.flip(sigmas_desc, dims=[0])
    return sigmas_desc


def sigmas_to_betas(sigmas_ascending: torch.Tensor) -> torch.Tensor:
    # alpha_bar = 1 / (1 + sigma^2), with t=0 close to clean and t=T-1 noisy.
    alpha_bar = 1.0 / (1.0 + sigmas_ascending ** 2)
    alphas = torch.empty_like(alpha_bar)
    alphas[0] = alpha_bar[0].clamp(min=1e-6, max=1.0)
    alphas[1:] = (alpha_bar[1:] / alpha_bar[:-1]).clamp(min=1e-6, max=1.0)
    betas = (1.0 - alphas).clamp(min=1e-6, max=0.999)
    return betas


def build_transitions(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    num_node_types: int,
    num_edge_types: int,
    node_prior_cfg=None,
    edge_prior_cfg=None,
    device: Optional[torch.device] = None,
    schedule_type: str = "karras",
) -> Dict[str, object]:
    if schedule_type == "log_uniform":
        sigmas = log_uniform_sigmas(num_steps, sigma_min, sigma_max, device=device, ascending=True)
    else:
        sigmas = karras_sigmas(num_steps, sigma_min, sigma_max, rho=rho, device=device, ascending=True)
    betas = sigmas_to_betas(sigmas).detach().cpu().numpy()
    node_init_prob = _prior_probs_from_cfg(node_prior_cfg, num_node_types)
    edge_init_prob = _prior_probs_from_cfg(edge_prior_cfg, num_edge_types)

    transitions = {
        "sigmas": sigmas,
        "betas": torch.from_numpy(betas).to(sigmas.device),
        "pos": ContigousTransition(sigmas.detach().cpu().numpy()).to(sigmas.device),
        "node": GeneralCategoricalTransition(betas, num_node_types, init_prob=node_init_prob).to(sigmas.device),
        "edge": GeneralCategoricalTransition(betas, num_edge_types, init_prob=edge_init_prob).to(sigmas.device),
    }
    return transitions


def normalize_pred_x0(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "pred_node_logits_x0": outputs["pred_node"],
        "pred_halfedge_logits_x0": outputs["pred_halfedge"],
        "pred_pos_x0": outputs["pred_pos"],
    }


def get_pos_snr_scale(
    transitions: Dict[str, object],
    t_graph: torch.Tensor,
    node_batch: torch.Tensor,
) -> torch.Tensor:
    # VE state is x = x0 + sigma * eps. Multiplying by 1 / sqrt(1 + sigma^2)
    # maps it to the VP form sqrt(alpha_bar) * x0 + sqrt(1 - alpha_bar) * eps
    # with alpha_bar = 1 / (1 + sigma^2), which matches the pretrained x0-prediction teacher.
    sigmas = transitions["sigmas"].index_select(0, t_graph)
    sigmas = sigmas.index_select(0, node_batch).unsqueeze(-1)
    return torch.rsqrt(1.0 + sigmas * sigmas)


@torch.no_grad()
def ema_update_(ema_model: torch.nn.Module, model: torch.nn.Module, decay: float):
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
    for ema_b, b in zip(ema_model.buffers(), model.buffers()):
        ema_b.data.copy_(b.data)


def build_sampling_timestep_schedule(total_steps: int, sample_steps: int) -> List[int]:
    if sample_steps <= 1:
        return [total_steps - 1]
    schedule = np.linspace(total_steps - 1, 0, sample_steps)
    schedule = np.round(schedule).astype(np.int64).tolist()
    # Keep order while removing duplicates from aggressive rounding.
    dedup = []
    for t in schedule:
        if not dedup or dedup[-1] != t:
            dedup.append(int(t))
    if dedup[-1] != 0:
        dedup.append(0)
    return dedup


def sample_prior_states(
    batch,
    transitions: Dict[str, object],
    sigma_max: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_nodes = batch["node_type"].shape[0]
    n_edges = batch["halfedge_type"].shape[0]
    node_state, _, _ = transitions["node"].sample_init(n_nodes)
    edge_state, _, _ = transitions["edge"].sample_init(n_edges)
    pos_state = torch.randn_like(batch["node_pos"]) * sigma_max
    return node_state, pos_state, edge_state


def renoise_from_pred_x0(
    pred_x0: Dict[str, torch.Tensor],
    t_next_graph: torch.Tensor,
    batch,
    transitions: Dict[str, object],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    node_batch = batch["node_type_batch"]
    edge_batch = batch["halfedge_type_batch"]

    pos_next = transitions["pos"].add_noise(pred_x0["pred_pos_x0"], t_next_graph, batch=node_batch)

    log_node_x0 = F.log_softmax(pred_x0["pred_node_logits_x0"], dim=-1)
    node_next, _ = transitions["node"].q_vt_sample(log_node_x0, t_next_graph, batch=node_batch)

    log_edge_x0 = F.log_softmax(pred_x0["pred_halfedge_logits_x0"], dim=-1)
    edge_next, _ = transitions["edge"].q_vt_sample(log_edge_x0, t_next_graph, batch=edge_batch)
    return node_next, pos_next, edge_next
