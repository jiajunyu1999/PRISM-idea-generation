from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader, Subset

from data.dataset import TokenizedSciJudgeEvalPairDataset, eval_collate_fn
from data.preprocess_scijudge import preprocess_split
from data.tokenize_cache import build_tokenized_pair_cache, tokenized_cache_path
from models.io_scitulu import load_scitulu_reward_model
from train.metrics import calibration_curve, category_breakdown, dedup_pair_accuracy, domain_breakdown, mean_margin, pair_accuracy, pair_auc, swap_consistency

SPLITS = ["test", "test_ood_year", "test_ood_iclr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distributed evaluate reward model on SciJudge splits")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--processed_dir", type=Path, default=Path("reward_scijudge/outputs/processed_scitulu"))
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("reward_scijudge/outputs/eval_scitulu_ddp"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def get_launch_context(device_arg: str) -> tuple[bool, int, int, int, torch.device]:
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
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
    import time

    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if path.exists() and path.stat().st_size > 0:
            return
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for file: {path}")


def ensure_eval_cache(
    dataset_dir: Path,
    processed_dir: Path,
    split: str,
    formatter: str,
    model_name: str,
    tokenizer,
    max_length: int,
    limit: int | None,
) -> Path:
    pair_path = processed_dir / f"{split}.pairs.{formatter}.jsonl"
    if not pair_path.exists() or limit is not None:
        preprocess_split(dataset_dir, processed_dir, split, formatter, max_samples=limit)

    cache_path = tokenized_cache_path(
        processed_dir=processed_dir,
        split=split,
        formatter=formatter,
        model_name=model_name,
        max_length=max_length,
        mode="eval",
        limit=limit,
    )
    if not cache_path.exists():
        count = build_tokenized_pair_cache(
            source_path=pair_path,
            output_path=cache_path,
            tokenizer=tokenizer,
            max_length=max_length,
            mode="eval",
            limit=limit,
        )
        print(f"[tokenize_cache] {split}/eval: {count} rows -> {cache_path}")
    return cache_path


@torch.no_grad()
def evaluate_split(model, loader, device: torch.device) -> dict:
    score_a, score_b, labels, pair_keys, categories = [], [], [], [], []
    for batch in loader:
        out = model.pairwise_compare(
            batch["a_input_ids"].to(device),
            batch["a_attention_mask"].to(device),
            batch["b_input_ids"].to(device),
            batch["b_attention_mask"].to(device),
        )
        score_a.extend(out["score_a"].detach().float().cpu().tolist())
        score_b.extend(out["score_b"].detach().float().cpu().tolist())
        labels.extend(batch["correct_answer"])
        pair_keys.extend(batch["pair_key"])
        categories.extend([m.get("category", "unknown") for m in batch["meta"]])

    margins = [a - b for a, b in zip(score_a, score_b)]
    probs = torch.sigmoid(torch.tensor(margins)).tolist()
    label_binary = [1 if y == "A" else 0 for y in labels]
    return {
        "pair_accuracy": pair_accuracy(score_a, score_b, labels),
        "pair_auc": pair_auc(margins, labels),
        "swap_consistency": swap_consistency(score_a, score_b),
        "margin_stats": mean_margin(score_a, score_b),
        "dedup_pair_accuracy": dedup_pair_accuracy(score_a, score_b, labels, pair_keys),
        "category_breakdown": category_breakdown(score_a, score_b, labels, categories),
        "domain_breakdown": domain_breakdown(score_a, score_b, labels, categories),
        "calibration_curve": calibration_curve(probs, label_binary),
        "num_examples": len(labels),
        "raw": {
            "score_a": score_a,
            "score_b": score_b,
            "labels": labels,
            "pair_keys": pair_keys,
            "categories": categories,
            "probs": probs,
            "margins": margins,
        },
    }


def merge_parts(parts: list[dict]) -> dict:
    merged = {"score_a": [], "score_b": [], "labels": [], "pair_keys": [], "categories": [], "probs": [], "margins": []}
    for part in parts:
        raw = part["raw"]
        for k in merged:
            merged[k].extend(raw[k])
    return merged


def export_error_cases(result: dict, out_path: Path, top_k: int = 100) -> None:
    rows = []
    raw = result["raw"]
    for i, (sa, sb, y, p, m) in enumerate(zip(raw["score_a"], raw["score_b"], raw["labels"], raw["probs"], raw["margins"])):
        pred = "A" if sa > sb else "B"
        if pred != y:
            conf = p if pred == "A" else 1.0 - p
            rows.append({"idx": i, "pred": pred, "gold": y, "confidence": conf, "margin": m})
    rows.sort(key=lambda x: abs(x["confidence"]), reverse=True)
    out_path.write_text(json.dumps(rows[:top_k], indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.processed_dir.mkdir(parents=True, exist_ok=True)

    distributed, rank, world_size, local_rank, device = get_launch_context(args.device)
    is_main = rank == 0

    model, tokenizer = load_scitulu_reward_model(args.ckpt, device=str(device))
    tokenizer.padding_side = "right"

    cache_paths = [
        tokenized_cache_path(
            processed_dir=args.processed_dir,
            split=split,
            formatter=cfg["data"]["formatter"],
            model_name=cfg["model"]["base_model_name"],
            max_length=cfg["model"]["max_length"],
            mode="eval",
            limit=args.max_samples,
        )
        for split in SPLITS
    ]

    if is_main:
        for split, cache_path in zip(SPLITS, cache_paths):
            ensure_eval_cache(
                dataset_dir=args.dataset_dir,
                processed_dir=args.processed_dir,
                split=split,
                formatter=cfg["data"]["formatter"],
                model_name=cfg["model"]["base_model_name"],
                tokenizer=tokenizer,
                max_length=cfg["model"]["max_length"],
                limit=args.max_samples,
            )
    else:
        for cache_path in cache_paths:
            wait_for_file(cache_path)

    if distributed:
        init_distributed(device, local_rank)

    all_results = {}
    for split in SPLITS:
        pair_path = tokenized_cache_path(
            processed_dir=args.processed_dir,
            split=split,
            formatter=cfg["data"]["formatter"],
            model_name=cfg["model"]["base_model_name"],
            max_length=cfg["model"]["max_length"],
            mode="eval",
            limit=args.max_samples,
        )
        ds = TokenizedSciJudgeEvalPairDataset(pair_path, limit=args.max_samples)
        local_ds = Subset(ds, list(range(rank, len(ds), world_size))) if distributed else ds
        loader = DataLoader(
            local_ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            collate_fn=lambda b: eval_collate_fn(b, pad_token_id=tokenizer.pad_token_id),
        )
        part = evaluate_split(model, loader, device)
        gathered = [None for _ in range(world_size)] if distributed else [part]
        if distributed:
            dist.all_gather_object(gathered, part)
        if is_main:
            merged = merge_parts(gathered)
            result = {
                "pair_accuracy": pair_accuracy(merged["score_a"], merged["score_b"], merged["labels"]),
                "pair_auc": pair_auc(merged["margins"], merged["labels"]),
                "swap_consistency": swap_consistency(merged["score_a"], merged["score_b"]),
                "margin_stats": mean_margin(merged["score_a"], merged["score_b"]),
                "dedup_pair_accuracy": dedup_pair_accuracy(merged["score_a"], merged["score_b"], merged["labels"], merged["pair_keys"]),
                "category_breakdown": category_breakdown(merged["score_a"], merged["score_b"], merged["labels"], merged["categories"]),
                "domain_breakdown": domain_breakdown(merged["score_a"], merged["score_b"], merged["labels"], merged["categories"]),
                "calibration_curve": calibration_curve(merged["probs"], [1 if y == "A" else 0 for y in merged["labels"]]),
                "num_examples": len(merged["labels"]),
                "raw": merged,
            }
            all_results[split] = {k: v for k, v in result.items() if k != "raw"}
            (args.output_dir / f"{split}_raw_metrics.json").write_text(json.dumps({
                "pair_accuracy": result["pair_accuracy"],
                "pair_auc": result["pair_auc"],
                "swap_consistency": result["swap_consistency"],
                "margin_stats": result["margin_stats"],
                "category_breakdown": result["category_breakdown"],
                "domain_breakdown": result["domain_breakdown"],
                "num_examples": result["num_examples"],
            }, indent=2), encoding="utf-8")
            (args.output_dir / f"{split}_dedup_metrics.json").write_text(json.dumps({"dedup_pair_accuracy": result["dedup_pair_accuracy"]}, indent=2), encoding="utf-8")
            export_error_cases(result, args.output_dir / f"{split}_error_cases.json")

    if is_main:
        test_acc = all_results["test"]["pair_accuracy"]
        all_results["ood_gap"] = {
            "acc_test_minus_ood_year": test_acc - all_results["test_ood_year"]["pair_accuracy"],
            "acc_test_minus_ood_iclr": test_acc - all_results["test_ood_iclr"]["pair_accuracy"],
        }
        (args.output_dir / "summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(json.dumps(all_results, indent=2))

    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
