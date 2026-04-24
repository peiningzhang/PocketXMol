import copy
import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.transition import ContigousTransition, GeneralCategoricalTransition, SigmaUniformCategoricalTransition


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
    elif schedule_type == "uniform":
        sigmas = torch.linspace(sigma_min, sigma_max, num_steps, device=device)
    else:
        sigmas = karras_sigmas(num_steps, sigma_min, sigma_max, rho=rho, device=device, ascending=True)

    # sigmas are in [sigma_min, sigma_max].  For categorical diffusion we directly
    # use sigma as the mask/corruption rate, clamped to [0, 1], so that it matches
    # the teacher's CategoricalPrior:  prob = (1-sigma)*one_hot(v0) + sigma*uniform.
    sigmas_np = sigmas.detach().cpu().numpy()
    node_init_prob = _prior_probs_from_cfg(node_prior_cfg, num_node_types)
    edge_init_prob = _prior_probs_from_cfg(edge_prior_cfg, num_edge_types)

    transitions = {
        "sigmas": sigmas,
        "pos": ContigousTransition(sigmas_np).to(sigmas.device),
        "node": SigmaUniformCategoricalTransition(
            sigmas_np.clip(0.0, 1.0), num_node_types, init_prob=node_init_prob
        ).to(sigmas.device),
        "edge": SigmaUniformCategoricalTransition(
            sigmas_np.clip(0.0, 1.0), num_edge_types, init_prob=edge_init_prob
        ).to(sigmas.device),
    }
    return transitions


def normalize_pred_x0(outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "pred_node_logits_x0": outputs["pred_node"],
        "pred_halfedge_logits_x0": outputs["pred_halfedge"],
        "pred_pos_x0": outputs["pred_pos"],
    }


def get_sampling_update_modes(distill_cfg) -> Tuple[str, str]:
    sampling_cfg = getattr(distill_cfg, "sampling", None)
    if sampling_cfg is None:
        return "snr_renoise", "categorical_transition"
    coordinate_update = str(getattr(sampling_cfg, "coordinate_update", "snr_renoise"))
    discrete_update = str(getattr(sampling_cfg, "discrete_update", "categorical_transition"))
    return coordinate_update, discrete_update


def _gather_sigma(transitions: Dict[str, object], t_graph: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    sigmas = transitions["sigmas"].index_select(0, t_graph)
    return sigmas.index_select(0, batch_index).unsqueeze(-1)


def inject_log_sigma(
    batch,
    transitions: Dict[str, object],
    t_graph: torch.Tensor,
    node_batch: torch.Tensor,
    eps: float = 1e-6,
):
    """Write per-atom log(sigma_t) into batch['log_sigma'] (in-place).

    The model reads this value from node_extra when 'log_sigma' is in
    model.config.addition_node_features, giving it explicit noise-level
    conditioning without modifying the core denoiser architecture.
    """
    sigma = transitions["sigmas"].index_select(0, t_graph).index_select(0, node_batch)
    batch["log_sigma"] = torch.log(sigma.clamp(min=eps))
    return batch


def add_log_sigma_to_model_config(model_cfg):
    """Return a shallow-copy of model_cfg that adds 'log_sigma' to addition_node_features.

    Used when building the student / EMA model so they receive noise-level info.
    The teacher remains unchanged (it uses the average embedding, not the last dim).
    """
    import copy
    cfg = copy.deepcopy(model_cfg)
    feats = list(getattr(cfg, "addition_node_features", []))
    if "log_sigma" not in feats:
        feats.append("log_sigma")
    cfg.addition_node_features = feats
    return cfg


def apply_cosine_pos_preconditioning(
    pred_x0_dict: dict,
    pos_in: torch.Tensor,
    transitions: Dict[str, object],
    t_graph: torch.Tensor,
    node_batch: torch.Tensor,
) -> dict:
    """Apply cosine preconditioning to the coordinate prediction only.

    Blends the network's raw position output with the noisy input:
        pred_pos_x0 = cos(σ·π/2) · pos_in + sin(σ·π/2) · net_out

    Properties:
        σ = 0 (clean)     → pred = pos_in   (identity / pure skip)
        σ = 1 (max noise) → pred = net_out  (trust the network)

    This is a cosine instance of the EDM c_skip/c_out preconditioning.
    Discrete node / edge features are NOT modified.

    Args:
        pred_x0_dict: dict returned by `normalize_pred_x0`, containing
            'pred_pos_x0', 'pred_node_logits_x0', 'pred_halfedge_logits_x0'.
        pos_in: noisy input coordinates used for this forward pass, shape (N, 3).
        transitions: dict produced by `build_transitions`.
        t_graph: integer timestep indices per graph, shape (B,).
        node_batch: per-atom graph indices, shape (N,).

    Returns:
        A new dict with 'pred_pos_x0' replaced by the blended prediction.
        All other keys are shallow-copied from pred_x0_dict.
    """
    sigma = transitions["sigmas"].index_select(0, t_graph).index_select(0, node_batch)
    sigma = sigma.unsqueeze(-1)  # (N, 1)

    angle = sigma * (torch.pi / 2.0)
    alpha = torch.cos(angle)    # c_skip:  1→0 as σ goes 0→1
    beta  = torch.sin(angle)    # c_out:   0→1 as σ goes 0→1

    net_pos  = pred_x0_dict["pred_pos_x0"]           # (N, 3) raw network output
    blended  = alpha * pos_in + beta * net_pos        # (N, 3)

    result = dict(pred_x0_dict)
    result["pred_pos_x0"] = blended
    return result


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


def update_coordinate_state(
    pred_pos_x0: torch.Tensor,
    pos_state: torch.Tensor,
    t_cur_graph: torch.Tensor,
    t_next_graph: torch.Tensor,
    batch,
    transitions: Dict[str, object],
    coordinate_update: str,
) -> torch.Tensor:
    node_batch = batch["node_type_batch"]
    if coordinate_update == "snr_renoise":
        return transitions["pos"].add_noise(pred_pos_x0, t_next_graph, batch=node_batch)
    if coordinate_update == "euler":
        sigma_cur = _gather_sigma(transitions, t_cur_graph, node_batch)
        sigma_next = _gather_sigma(transitions, t_next_graph, node_batch)
        sigma_cur = sigma_cur.clamp(min=1e-12)
        pred_eps = (pos_state - pred_pos_x0) / sigma_cur
        return pos_state + (sigma_next - sigma_cur) * pred_eps
    if coordinate_update in {"direct_x0", "x0"}:
        return pred_pos_x0
    raise ValueError(f"Unknown coordinate update mode: {coordinate_update}")


def campbell_dfm_step(
    current_v: torch.Tensor,
    pred_logits: torch.Tensor,
    sigma_i: torch.Tensor,
    sigma_next: torch.Tensor,
    eps: float = 1e-5,
    sigma_data: float = 1.0,
    stochasticity: float = 2.0,
) -> torch.Tensor:
    """
    Campbell-style discrete update for a categorical state.

    This follows the local heuristic in the user's reference:
    predict a clean categorical proposal from `pred_logits`, then replace a
    subset of current categories with the proposal, with an optional
    stochastic resampling term.
    """
    num_classes = pred_logits.shape[-1]
    device = pred_logits.device
    p_1_given_t = F.softmax(pred_logits, dim=-1)

    sigma_data_t = torch.as_tensor(sigma_data, device=device, dtype=sigma_i.dtype)
    stochasticity_t = torch.as_tensor(stochasticity, device=device, dtype=sigma_i.dtype)

    mask_rate = sigma_i / (sigma_i + sigma_data_t)
    mask_rate_next = sigma_next / (sigma_next + sigma_data_t)
    t = 1.0 - mask_rate
    t_next = 1.0 - mask_rate_next

    dt = (t_next - t).clamp(min=0.0)
    alpha_t = t.clamp(min=0.0, max=1.0 - eps)
    alpha_t_next = t_next.clamp(min=0.0, max=1.0 - eps)
    alpha_t_prime = (alpha_t_next - alpha_t) / dt.clamp(min=1e-12)

    mask_rate_derivative = sigma_data_t / (sigma_i + sigma_data_t).pow(2)
    stochasticity_term = stochasticity_t / mask_rate_derivative.clamp(min=1e-12)

    unmask_prob = dt * (alpha_t_prime + stochasticity_term * alpha_t) / (1.0 - alpha_t).clamp(min=eps)
    mask_prob = dt * stochasticity_term
    unmask_prob = unmask_prob.clamp(min=0.0, max=1.0)
    mask_prob = mask_prob.clamp(min=0.0, max=1.0)
    denom = (unmask_prob + mask_prob).clamp(min=eps)
    unmask_prob = unmask_prob / denom

    x1 = torch.distributions.Categorical(probs=p_1_given_t).sample()
    will_unmask = torch.rand(current_v.shape[0], device=device) < unmask_prob.squeeze(-1)

    v_next = current_v.clone()
    v_next[will_unmask] = x1[will_unmask]

    # Keep the local probability tensor around for possible future logging/debugging.
    _ = num_classes
    return v_next


def update_discrete_state(
    pred_x0: Dict[str, torch.Tensor],
    t_cur_graph: torch.Tensor,
    t_next_graph: torch.Tensor,
    batch,
    transitions: Dict[str, object],
    discrete_update: str,
    sampling_cfg=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    node_batch = batch["node_type_batch"]
    edge_batch = batch["halfedge_type_batch"]
    if discrete_update == "categorical_transition":
        log_node_x0 = F.log_softmax(pred_x0["pred_node_logits_x0"], dim=-1)
        node_next, _ = transitions["node"].q_vt_sample(log_node_x0, t_next_graph, batch=node_batch)

        log_edge_x0 = F.log_softmax(pred_x0["pred_halfedge_logits_x0"], dim=-1)
        edge_next, _ = transitions["edge"].q_vt_sample(log_edge_x0, t_next_graph, batch=edge_batch)
        return node_next, edge_next
    if discrete_update == "campbell_dfm":
        sigma_data = float(getattr(sampling_cfg, "campbell_sigma_data", 1.0)) if sampling_cfg is not None else 1.0
        stochasticity = float(getattr(sampling_cfg, "campbell_stochasticity", 2.0)) if sampling_cfg is not None else 2.0
        sigma_node_cur = _gather_sigma(transitions, t_cur_graph, node_batch)
        sigma_node_next = _gather_sigma(transitions, t_next_graph, node_batch)
        sigma_edge_cur = _gather_sigma(transitions, t_cur_graph, edge_batch)
        sigma_edge_next = _gather_sigma(transitions, t_next_graph, edge_batch)
        node_next = campbell_dfm_step(
            batch["node_type"],
            pred_x0["pred_node_logits_x0"],
            sigma_node_cur,
            sigma_node_next,
            sigma_data=sigma_data,
            stochasticity=stochasticity,
        )
        edge_next = campbell_dfm_step(
            batch["halfedge_type"],
            pred_x0["pred_halfedge_logits_x0"],
            sigma_edge_cur,
            sigma_edge_next,
            sigma_data=sigma_data,
            stochasticity=stochasticity,
        )
        return node_next, edge_next
    if discrete_update in {"argmax", "direct_x0"}:
        node_next = pred_x0["pred_node_logits_x0"].argmax(dim=-1)
        edge_next = pred_x0["pred_halfedge_logits_x0"].argmax(dim=-1)
        return node_next, edge_next
    raise ValueError(f"Unknown discrete update mode: {discrete_update}")


def renoise_from_pred_x0(
    pred_x0: Dict[str, torch.Tensor],
    pos_state: torch.Tensor,
    t_cur_graph: torch.Tensor,
    t_next_graph: torch.Tensor,
    batch,
    transitions: Dict[str, object],
    coordinate_update: str = "snr_renoise",
    discrete_update: str = "categorical_transition",
    sampling_cfg=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos_next = update_coordinate_state(
        pred_x0["pred_pos_x0"],
        pos_state,
        t_cur_graph,
        t_next_graph,
        batch,
        transitions,
        coordinate_update,
    )
    node_next, edge_next = update_discrete_state(
        pred_x0,
        t_cur_graph,
        t_next_graph,
        batch,
        transitions,
        discrete_update,
        sampling_cfg=sampling_cfg,
    )
    return node_next, pos_next, edge_next
