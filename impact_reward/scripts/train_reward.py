from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from data.build_manifest import main as _unused  # keeps package import path stable
from data.dataset import SciJudgeEvalPairDataset, SciJudgePairDataset, eval_collate_fn, pair_collate_fn
from data.preprocess_scijudge import preprocess_split
from models.io import load_checkpoint_into_model
from models.reward_model import RewardModel
from train.trainer import RewardTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SciJudge reward model")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--processed_dir", type=Path, default=Path("reward_scijudge/outputs/processed"))
    parser.add_argument("--output_dir", type=Path, default=Path("reward_scijudge/outputs/run"))
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


def is_distributed_launch() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed(device_arg: str) -> tuple[bool, int, int, int, torch.device]:
    distributed = is_distributed_launch()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if device_arg == "cpu":
        device = torch.device("cpu")
    elif device_arg == "cuda":
        device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
    else:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{local_rank}" if distributed else "cuda")
        else:
            device = torch.device("cpu")

    if distributed:
        backend = "nccl" if device.type == "cuda" else "gloo"
        if device.type == "cuda":
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)

    return distributed, rank, world_size, local_rank, device


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    distributed, rank, world_size, local_rank, device = setup_distributed(args.device)
    is_main_process = rank == 0
    set_seed(cfg["train"]["seed"] + rank)

    if is_main_process:
        ensure_preprocessed(args.dataset_dir, args.processed_dir, cfg["data"]["formatter"], args.max_train_samples)
    if distributed:
        dist.barrier()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base_model_name"], use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    train_path = args.processed_dir / f"train.pairs.{cfg['data']['formatter']}.jsonl"
    test_path = args.processed_dir / f"test.pairs.{cfg['data']['formatter']}.jsonl"

    all_train = SciJudgePairDataset(
        path=train_path,
        tokenizer=tokenizer,
        max_length=cfg["model"]["max_length"],
        limit=args.max_train_samples,
    )
    n_total = len(all_train)
    n_val = max(1, int(n_total * cfg["data"]["val_ratio"]))
    indices = np.random.permutation(n_total)
    val_indices = indices[:n_val].tolist()
    train_indices = indices[n_val:].tolist()

    train_ds = Subset(all_train, train_indices)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None

    eval_ds = SciJudgeEvalPairDataset(
        path=test_path,
        tokenizer=tokenizer,
        max_length=cfg["model"]["max_length"],
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
        collate_fn=lambda b: pair_collate_fn(b, pad_token_id=tokenizer.pad_token_id),
    )
    val_loader = DataLoader(
        val_eval_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        sampler=val_sampler,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=device.type == "cuda",
        persistent_workers=cfg["data"]["num_workers"] > 0,
        collate_fn=lambda b: eval_collate_fn(b, pad_token_id=tokenizer.pad_token_id),
    )

    model = RewardModel(cfg["model"]["base_model_name"], use_confidence_head=False)
    if args.resume_from is not None:
        load_checkpoint_into_model(model, args.resume_from)
    model.to(device)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
        if is_main_process:
            print(f"Using DDP on {world_size} processes")
    elif device.type == "cuda" and torch.cuda.device_count() > 1 and cfg["train"].get("use_data_parallel", True):
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]), weight_decay=float(cfg["train"]["weight_decay"]))
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
        }
        (args.output_dir / "launch_meta.json").write_text(json.dumps(launch_meta, indent=2), encoding="utf-8")
    result = trainer.train()
    if is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "train_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
