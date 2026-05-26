from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoModel

from models.heads import ScalarRewardHead
from models.pooling import last_token_pool


def _resolve_dtype(name: str | None) -> torch.dtype | None:
    if not name:
        return None
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


def resolve_lora_target_modules(config, requested: Iterable[str] | None) -> list[str]:
    if requested is not None:
        return list(requested)

    model_type = str(getattr(config, "model_type", "") or "").lower()
    if model_type == "llama":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if model_type == "qwen2":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if model_type == "opt":
        return ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    raise ValueError(
        f"Unsupported model_type for automatic LoRA target resolution: {model_type}. "
        "Please set model.target_modules explicitly in the config."
    )


class SciTuluRewardModel(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Iterable[str] | None = None,
        torch_dtype: str | None = "bfloat16",
        gradient_checkpointing: bool = True,
        resume_adapter_dir: str | None = None,
        resize_token_embeddings_to: int | None = None,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(base_model_name)
        dtype = _resolve_dtype(torch_dtype)
        self.backbone = AutoModel.from_pretrained(
            base_model_name,
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if resize_token_embeddings_to is not None:
            self.backbone.resize_token_embeddings(int(resize_token_embeddings_to), mean_resizing=False)
        if gradient_checkpointing:
            try:
                self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.backbone.gradient_checkpointing_enable()
            if hasattr(self.backbone, "enable_input_require_grads"):
                self.backbone.enable_input_require_grads()
            self.backbone.config.use_cache = False
        else:
            self.backbone.config.use_cache = False

        lora_targets = resolve_lora_target_modules(config, target_modules)
        if resume_adapter_dir is not None:
            self.backbone = PeftModel.from_pretrained(self.backbone, resume_adapter_dir, is_trainable=True)
        else:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=lora_targets,
            )
            self.backbone = get_peft_model(self.backbone, peft_config)

        hidden_size = getattr(config, "hidden_size", None) or getattr(config, "d_model")
        if hidden_size is None:
            raise ValueError("Cannot infer hidden size from model config")
        self.reward_head = ScalarRewardHead(hidden_size)
        self.confidence_head = None

    def print_trainable_parameters(self) -> None:
        trainable = 0
        total = 0
        for param in self.parameters():
            total += param.numel()
            if param.requires_grad:
                trainable += param.numel()
        pct = 100.0 * trainable / max(total, 1)
        print(f"trainable params: {trainable} || all params: {total} || trainable%: {pct:.4f}")

    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = last_token_pool(out.last_hidden_state, attention_mask).float()
        return self.reward_head(pooled)

    def forward(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pos = self.score(chosen_input_ids, chosen_attention_mask)
        neg = self.score(rejected_input_ids, rejected_attention_mask)
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
        prob = torch.sigmoid(s_a - s_b)
        return {"score_a": s_a, "score_b": s_b, "p_a_beats_b": prob}
