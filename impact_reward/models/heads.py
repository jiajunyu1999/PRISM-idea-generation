from __future__ import annotations

import torch.nn as nn


class ScalarRewardHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.proj(x).squeeze(-1)


class ConfidenceHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.proj(x).squeeze(-1)
