from __future__ import annotations

import torch


def last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # hidden_states: [B, T, H], attention_mask: [B, T]
    lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
    batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
    return hidden_states[batch_indices, lengths]
