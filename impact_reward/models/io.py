from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer

from models.reward_model import RewardModel


def load_checkpoint_into_model(model, ckpt_dir: str | Path) -> None:
    ckpt = Path(ckpt_dir)
    backbone_dir = ckpt / "backbone"
    head_path = ckpt / "reward_head.pt"

    if not backbone_dir.exists() or not head_path.exists():
        raise FileNotFoundError(f"Invalid checkpoint directory: {ckpt}")

    model.backbone = type(model.backbone).from_pretrained(str(backbone_dir))
    head_state = torch.load(head_path, map_location="cpu", weights_only=True)
    model.reward_head.load_state_dict(head_state)

    confidence_path = ckpt / "confidence_head.pt"
    if model.confidence_head is not None and confidence_path.exists():
        conf_state = torch.load(confidence_path, map_location="cpu", weights_only=True)
        model.confidence_head.load_state_dict(conf_state)


def load_reward_model(ckpt_dir: str | Path, device: str = "cpu"):
    ckpt = Path(ckpt_dir)
    backbone_dir = ckpt / "backbone"
    tokenizer_dir = ckpt / "tokenizer"
    model = RewardModel(str(backbone_dir))
    head_state = torch.load(ckpt / "reward_head.pt", map_location="cpu", weights_only=True)
    model.reward_head.load_state_dict(head_state)
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir if tokenizer_dir.exists() else backbone_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return model, tokenizer
