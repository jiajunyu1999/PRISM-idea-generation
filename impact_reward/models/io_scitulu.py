from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

from models.reward_model_scitulu import SciTuluRewardModel


def _load_training_config(ckpt_dir: Path) -> dict:
    config_path = ckpt_dir.parent / "config_used.yaml"
    if config_path.exists():
        return yaml.safe_load(config_path.read_text(encoding="utf-8"))
    launch_meta_path = ckpt_dir.parent / "launch_meta.json"
    if launch_meta_path.exists():
        meta = json.loads(launch_meta_path.read_text(encoding="utf-8"))
        return {
            "model": {"base_model_name": meta["base_model_name"]},
            "train": {"gradient_checkpointing": False},
        }
    raise FileNotFoundError(f"Could not find config_used.yaml or launch_meta.json for checkpoint {ckpt_dir}")


def load_scitulu_reward_model(ckpt_dir: str | Path, device: str = "cpu"):
    ckpt = Path(ckpt_dir)
    backbone_dir = ckpt / "backbone"
    tokenizer_dir = ckpt / "tokenizer"
    cfg = _load_training_config(ckpt)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir if tokenizer_dir.exists() else cfg["model"]["base_model_name"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.bos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"

    model = SciTuluRewardModel(
        base_model_name=cfg["model"]["base_model_name"],
        lora_r=int(cfg["model"].get("lora_r", 16)),
        lora_alpha=int(cfg["model"].get("lora_alpha", 32)),
        lora_dropout=float(cfg["model"].get("lora_dropout", 0.05)),
        target_modules=cfg["model"].get("target_modules"),
        torch_dtype=cfg["model"].get("torch_dtype", "bfloat16"),
        gradient_checkpointing=bool(cfg.get("train", {}).get("gradient_checkpointing", False)),
        resume_adapter_dir=str(backbone_dir),
        resize_token_embeddings_to=len(tokenizer),
    )
    head_state = torch.load(ckpt / "reward_head.pt", map_location="cpu", weights_only=True)
    model.reward_head.load_state_dict(head_state)
    model.to(device)
    model.eval()
    return model, tokenizer
