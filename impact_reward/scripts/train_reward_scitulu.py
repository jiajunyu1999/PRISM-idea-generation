from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from data.dataset import TokenizedSciJudgeEvalPairDataset, TokenizedSciJudgePairDataset, eval_collate_fn, pair_collate_fn
from data.preprocess_scijudge import preprocess_split
from data.tokenize_cache import build_tokenized_pair_cache, tokenized_cache_path
from models.reward_model_scitulu import SciTuluRewardModel
from train.trainer import RewardTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SciTulu-7B reward model with LoRA")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--processed_dir", type=Path, default=Path("reward_scijudge/outputs/processed"))
    parser.add_argument("--output_dir", type=Path, default=Path("reward_scijudge/outputs/run_scitulu"))
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume_from", type=Path, default=None)
    parser.add_argument("--resume_weights_only", action="store_true")
    return parser.parse_args()


def ensure_preprocessed(dataset_dir: Path, processed_dir: Path, formatter: str, max_samples: int | None) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_pair = processed_dir / f"train.pairs.{formatter}.jsonl"
    test_pair = processed_dir / f"test.pairs.{formatter}.jsonl"
    if train_pair.exists() and test_pair.exists() and max_samples is None:
        return
    preprocess_split(dataset_dir, processed_dir, "train", formatter, max_samples=max_samples)
    preprocess_split(dataset_dir, processed_dir, "test", formatter, max_samples=max_samples)


def ensure_tokenized_cache(
    processed_dir: Path,
    split: str,
    formatter: str,
    model_name: str,
    tokenizer,
    max_length: int,
    mode: str,
    limit: int | None,
) -> Path:
    source_path = processed_dir / f"{split}.pairs.{formatter}.jsonl"
    cache_path = tokenized_cache_path(
        processed_dir=processed_dir,
        split=split,
        formatter=formatter,
        model_name=model_name,
        max_length=max_length,
        mode=mode,
        limit=limit,
    )
    if cache_path.exists():
        return cache_path
    count = build_tokenized_pair_cache(
        source_path=source_path,
        output_path=cache_path,
        tokenizer=tokenizer,
        max_length=max_length,
        mode=mode,
        limit=limit,
    )
    print(f"[tokenize_cache] {split}/{mode}: {count} rows -> {cache_path}")
    return cache_path


def is_distributed_launch() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def resolve_launch_context(device_arg: str) -> tuple[bool, int, int, int, torch.device]:
    distributed = is_distributed_launch()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if device_arg == "cpu":
        device = torch.device("cpu")
    elif device_arg == "cuda":
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    else:
        device = torch.device(f"cuda:{local_rank}" if distributed and torch.cuda.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    return distributed, rank, world_size, local_rank, device


def init_distributed(device: torch.device, local_rank: int) -> None:
    backend = "nccl" if device.type == "cuda" else "gloo"
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)


def wait_for_file(path: Path, timeout_seconds: int = 7200, poll_interval: float = 5.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for cache file: {path}")


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    distributed, rank, world_size, local_rank, device = resolve_launch_context(args.device)
    is_main_process = rank == 0
    set_seed(cfg["train"]["seed"] + rank)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model_name"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token or tokenizer.bos_token
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer must provide a valid pad_token_id after fallback assignment")
    tokenizer.padding_side = "right"
    pad_token_id = int(tokenizer.pad_token_id)

    train_cache_path = tokenized_cache_path(
        processed_dir=args.processed_dir,
        split="train",
        formatter=cfg["data"]["formatter"],
        model_name=cfg["model"]["base_model_name"],
        max_length=cfg["model"]["max_length"],
        mode="train",
        limit=args.max_train_samples,
    )
    test_cache_path = tokenized_cache_path(
        processed_dir=args.processed_dir,
        split="test",
        formatter=cfg["data"]["formatter"],
        model_name=cfg["model"]["base_model_name"],
        max_length=cfg["model"]["max_length"],
        mode="eval",
        limit=args.max_eval_samples,
    )

    if is_main_process:
        ensure_preprocessed(args.dataset_dir, args.processed_dir, cfg["data"]["formatter"], args.max_train_samples)
        train_path = ensure_tokenized_cache(
            processed_dir=args.processed_dir,
            split="train",
            formatter=cfg["data"]["formatter"],
            model_name=cfg["model"]["base_model_name"],
            tokenizer=tokenizer,
            max_length=cfg["model"]["max_length"],
            mode="train",
            limit=args.max_train_samples,
        )
        test_path = ensure_tokenized_cache(
            processed_dir=args.processed_dir,
            split="test",
            formatter=cfg["data"]["formatter"],
            model_name=cfg["model"]["base_model_name"],
            tokenizer=tokenizer,
            max_length=cfg["model"]["max_length"],
            mode="eval",
            limit=args.max_eval_samples,
        )
    else:
        wait_for_file(train_cache_path)
        wait_for_file(test_cache_path)

    if distributed:
        init_distributed(device=device, local_rank=local_rank)

    train_path = train_cache_path
    test_path = test_cache_path

    all_train = TokenizedSciJudgePairDataset(
        path=train_path,
        limit=args.max_train_samples,
    )
    n_total = len(all_train)
    n_val = max(1, int(n_total * cfg["data"]["val_ratio"]))
    indices = np.random.permutation(n_total)
    val_indices = indices[:n_val].tolist()
    train_indices = indices[n_val:].tolist()

    train_ds = Subset(all_train, train_indices)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None

    eval_ds = TokenizedSciJudgeEvalPairDataset(
        path=test_path,
        limit=args.max_eval_samples,
    )
    val_eval_ds = Subset(eval_ds, list(range(min(len(eval_ds), max(8, n_val)))))
    val_sampler = DistributedSampler(val_eval_ds, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["data"]["num_workers"] > 0,
        collate_fn=lambda b: pair_collate_fn(b, pad_token_id=pad_token_id),
    )
    val_loader = DataLoader(
        val_eval_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["data"]["num_workers"] > 0,
        collate_fn=lambda b: eval_collate_fn(b, pad_token_id=pad_token_id),
    )

    resume_adapter_dir = None
    if args.resume_from is not None:
        resume_adapter_dir = str(args.resume_from / "backbone")

    model = SciTuluRewardModel(
        base_model_name=cfg["model"]["base_model_name"],
        lora_r=int(cfg["model"]["lora_r"]),
        lora_alpha=int(cfg["model"]["lora_alpha"]),
        lora_dropout=float(cfg["model"]["lora_dropout"]),
        target_modules=cfg["model"].get("target_modules"),
        torch_dtype=cfg["model"].get("torch_dtype", "bfloat16"),
        gradient_checkpointing=bool(cfg["train"].get("gradient_checkpointing", True)),
        resume_adapter_dir=resume_adapter_dir,
        resize_token_embeddings_to=len(tokenizer),
    )
    if is_main_process:
        model.print_trainable_parameters()
    if args.resume_from is not None:
        head_state = torch.load(args.resume_from / "reward_head.pt", map_location="cpu", weights_only=True)
        model.reward_head.load_state_dict(head_state)
    model.to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=bool(cfg["train"].get("ddp_find_unused_parameters", False)),
        )
        if bool(cfg["train"].get("ddp_static_graph", True)) and hasattr(model, "_set_static_graph"):
            model._set_static_graph()
        if is_main_process:
            print(f"Using DDP on {world_size} processes")

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    trainer = RewardTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
        output_dir=args.output_dir,
        tokenizer=tokenizer,
        train_sampler=train_sampler,
        distributed=distributed,
        is_main_process=is_main_process,
    )
    if args.resume_from is not None and not args.resume_weights_only:
        trainer.load_training_state(args.resume_from)

    if is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "config_used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        launch_meta = {
            "distributed": distributed,
            "world_size": world_size,
            "device": str(device),
            "base_model_name": cfg["model"]["base_model_name"],
            "resume_from": str(args.resume_from) if args.resume_from is not None else None,
            "resume_weights_only": bool(args.resume_weights_only),
            "dataset_dir": str(args.dataset_dir),
            "processed_dir": str(args.processed_dir),
            "output_dir": str(args.output_dir),
            "branch": "scitulu_lora_reward",
        }
        (args.output_dir / "launch_meta.json").write_text(json.dumps(launch_meta, indent=2), encoding="utf-8")

    result = trainer.train()
    if is_main_process:
        (args.output_dir / "train_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
