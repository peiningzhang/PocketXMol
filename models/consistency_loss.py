from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(values: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return values.mean()
    if values.ndim > 1:
        view_shape = [values.shape[0]] + [1] * (values.ndim - 1)
        mask = mask.view(*view_shape)
    denom = mask.float().sum().clamp(min=1.0)
    return (values * mask.float()).sum() / denom


def _weighted_masked_mean(
    values: torch.Tensor,
    mask: Optional[torch.Tensor],
    weights: Optional[torch.Tensor],
) -> torch.Tensor:
    if weights is None:
        return _masked_mean(values, mask)
    if weights.ndim > 1:
        weights = weights.reshape(weights.shape[0], -1).mean(dim=-1)
    if mask is not None:
        weights = weights * mask.float()
    denom = weights.sum().clamp(min=1.0)
    return (values * weights).sum() / denom


def compute_joint_physics_loss(
    pred_pos_x0: torch.Tensor,
    pred_node_logits_x0: torch.Tensor,
    vdw_radii_table: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # TODO: keep as a design placeholder during cloud-stage experiments.
    # If enabled in follow-up, implement differentiable VdW clash penalties.
    _ = pred_pos_x0, pred_node_logits_x0, vdw_radii_table
    return pred_pos_x0.new_zeros(())


class MixedStateConsistencyLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 1.0,
        node_weight: float = 1.0,
        edge_weight: float = 1.0,
        edge_positive_weight: float = 1.0,
        physics_weight: float = 0.0,
    ):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.node_weight = float(node_weight)
        self.edge_weight = float(edge_weight)
        self.edge_positive_weight = float(edge_positive_weight)
        self.physics_weight = float(physics_weight)

    def forward(
        self,
        student_pred_x0: Dict[str, torch.Tensor],
        ema_pred_x0: Dict[str, torch.Tensor],
        mask_pos: Optional[torch.Tensor] = None,
        mask_node: Optional[torch.Tensor] = None,
        mask_edge: Optional[torch.Tensor] = None,
        edge_target_type: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        pos_loss_all = F.mse_loss(
            student_pred_x0["pred_pos_x0"],
            ema_pred_x0["pred_pos_x0"].detach(),
            reduction="none",
        )
        pos_loss = _masked_mean(pos_loss_all, mask_pos)

        node_log_prob = F.log_softmax(student_pred_x0["pred_node_logits_x0"], dim=-1)
        node_target_prob = F.softmax(ema_pred_x0["pred_node_logits_x0"].detach(), dim=-1)
        node_kl_all = F.kl_div(node_log_prob, node_target_prob, reduction="none").sum(dim=-1)
        node_loss = _masked_mean(node_kl_all, mask_node)

        edge_log_prob = F.log_softmax(student_pred_x0["pred_halfedge_logits_x0"], dim=-1)
        edge_target_prob = F.softmax(ema_pred_x0["pred_halfedge_logits_x0"].detach(), dim=-1)
        edge_kl_all = F.kl_div(edge_log_prob, edge_target_prob, reduction="none").sum(dim=-1)
        edge_weights = None
        if edge_target_type is not None and self.edge_positive_weight != 1.0:
            edge_weights = torch.ones_like(edge_kl_all)
            edge_weights = torch.where(
                edge_target_type > 0,
                edge_weights.new_full(edge_weights.shape, self.edge_positive_weight),
                edge_weights,
            )
        edge_loss = _weighted_masked_mean(edge_kl_all, mask_edge, edge_weights)

        physics_loss = compute_joint_physics_loss(
            student_pred_x0["pred_pos_x0"],
            student_pred_x0["pred_node_logits_x0"],
        )
        total = (
            self.pos_weight * pos_loss
            + self.node_weight * node_loss
            + self.edge_weight * edge_loss
            + self.physics_weight * physics_loss
        )
        return {
            "pos": pos_loss,
            "node_type": node_loss,
            "edge_type": edge_loss,
            "physics": physics_loss,
            "total": total,
        }
