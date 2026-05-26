from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from models.heads import ConfidenceHead, ScalarRewardHead
from models.pooling import last_token_pool


class RewardModel(nn.Module):
    def __init__(self, base_model_name: str, use_confidence_head: bool = False):
        super().__init__()
        config = AutoConfig.from_pretrained(base_model_name)
        self.backbone = AutoModel.from_pretrained(base_model_name, config=config)
        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "d_model")
        if hidden_size is None:
            raise ValueError("Cannot infer hidden size from model config")
        self.reward_head = ScalarRewardHead(hidden_size)
        self.confidence_head = ConfidenceHead(hidden_size) if use_confidence_head else None

    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor] | torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = last_token_pool(out.last_hidden_state, attention_mask)
        score = self.reward_head(pooled)
        if self.confidence_head is None:
            return score
        conf = self.confidence_head(pooled)
        return {"score": score, "confidence": conf}

    def forward(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pos = self.score(chosen_input_ids, chosen_attention_mask)
        neg = self.score(rejected_input_ids, rejected_attention_mask)
        if isinstance(pos, dict):
            return {
                "pos_scores": pos["score"],
                "neg_scores": neg["score"],
                "pos_confidence": pos["confidence"],
                "neg_confidence": neg["confidence"],
            }
        return {"pos_scores": pos, "neg_scores": neg, "aux_outputs": None}

    @torch.no_grad()
    def pairwise_compare(
        self,
        a_input_ids: torch.Tensor,
        a_attention_mask: torch.Tensor,
        b_input_ids: torch.Tensor,
        b_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        s_a = self.score(a_input_ids, a_attention_mask)
        s_b = self.score(b_input_ids, b_attention_mask)
        if isinstance(s_a, dict):
            s_a = s_a["score"]
            s_b = s_b["score"]
        prob = torch.sigmoid(s_a - s_b)
        return {"score_a": s_a, "score_b": s_b, "p_a_beats_b": prob}
