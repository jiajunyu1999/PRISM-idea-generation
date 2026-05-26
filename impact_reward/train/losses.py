from __future__ import annotations

import torch
import torch.nn.functional as F


def build_weight(abs_gap: torch.Tensor, rel_gap: torch.Tensor, w_min: float = 1.0, w_max: float = 5.0) -> torch.Tensor:
    w = 0.5 * torch.log1p(abs_gap) + 2.0 * rel_gap
    return torch.clamp(w, min=w_min, max=w_max)


def pairwise_logistic_loss(pos: torch.Tensor, neg: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    per = -F.logsigmoid(pos - neg)
    if weight is not None:
        per = per * weight
    return per.mean()


def margin_ranking_loss(
    pos: torch.Tensor,
    neg: torch.Tensor,
    margin: float = 1.0,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    per = F.relu(margin - (pos - neg))
    if weight is not None:
        per = per * weight
    return per.mean()


def score_regularization(pos: torch.Tensor, neg: torch.Tensor, coef: float = 1e-4) -> torch.Tensor:
    return coef * (pos.pow(2).mean() + neg.pow(2).mean())
