from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from data.dataset import SciJudgeEvalPairDataset, eval_collate_fn
from data.preprocess_scijudge import preprocess_split
from models.io import load_reward_model
from train.metrics import (
    calibration_curve,
    category_breakdown,
    domain_breakdown,
    dedup_pair_accuracy,
    mean_margin,
    pair_accuracy,
    pair_auc,
    swap_consistency,
)

SPLITS = ["test", "test_ood_year", "test_ood_iclr"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate reward model on SciJudge splits")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--processed_dir", type=Path, default=Path("reward_scijudge/outputs/processed"))
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("reward_scijudge/outputs/eval"))
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def ensure_split(path_dataset: Path, path_processed: Path, split: str, formatter: str, max_samples: int | None) -> Path:
    pair_path = path_processed / f"{split}.pairs.{formatter}.jsonl"
    if not pair_path.exists() or max_samples is not None:
        preprocess_split(path_dataset, path_processed, split, formatter, max_samples=max_samples)
    return pair_path


@torch.no_grad()
def evaluate_split(model, loader, device: torch.device) -> Dict:
    score_a, score_b, labels, pair_keys, categories = [], [], [], [], []

    for batch in loader:
        a_ids = batch["a_input_ids"].to(device)
        a_mask = batch["a_attention_mask"].to(device)
        b_ids = batch["b_input_ids"].to(device)
        b_mask = batch["b_attention_mask"].to(device)

        out = model.pairwise_compare(a_ids, a_mask, b_ids, b_mask)
        sa = out["score_a"].detach().float().cpu().tolist()
        sb = out["score_b"].detach().float().cpu().tolist()

        score_a.extend(sa)
        score_b.extend(sb)
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


def export_error_cases(result: Dict, out_path: Path, top_k: int = 100) -> None:
    rows = []
    raw = result["raw"]
    for i, (sa, sb, y, p, m) in enumerate(
        zip(raw["score_a"], raw["score_b"], raw["labels"], raw["probs"], raw["margins"])
    ):
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

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_reward_model(args.ckpt, device=str(device))

    all_results: Dict[str, Dict] = {}

    for split in SPLITS:
        pair_path = ensure_split(args.dataset_dir, args.processed_dir, split, cfg["data"]["formatter"], args.max_samples)
        ds = SciJudgeEvalPairDataset(pair_path, tokenizer=tokenizer, max_length=cfg["model"]["max_length"], limit=args.max_samples)
        loader = DataLoader(
            ds,
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"]["num_workers"],
            collate_fn=lambda b: eval_collate_fn(b, pad_token_id=tokenizer.pad_token_id),
        )
        result = evaluate_split(model, loader, device)
        all_results[split] = {k: v for k, v in result.items() if k != "raw"}

        raw_path = args.output_dir / f"{split}_raw_metrics.json"
        dedup_path = args.output_dir / f"{split}_dedup_metrics.json"
        err_path = args.output_dir / f"{split}_error_cases.json"

        raw_metrics = {
            "pair_accuracy": result["pair_accuracy"],
            "pair_auc": result["pair_auc"],
            "swap_consistency": result["swap_consistency"],
            "margin_stats": result["margin_stats"],
            "domain_breakdown": result["domain_breakdown"],
            "num_examples": result["num_examples"],
        }
        dedup_metrics = {"dedup_pair_accuracy": result["dedup_pair_accuracy"]}

        raw_path.write_text(json.dumps(raw_metrics, indent=2), encoding="utf-8")
        dedup_path.write_text(json.dumps(dedup_metrics, indent=2), encoding="utf-8")
        export_error_cases(result, err_path)

    test_acc = all_results["test"]["pair_accuracy"]
    all_results["ood_gap"] = {
        "acc_test_minus_ood_year": test_acc - all_results["test_ood_year"]["pair_accuracy"],
        "acc_test_minus_ood_iclr": test_acc - all_results["test_ood_iclr"]["pair_accuracy"],
    }

    (args.output_dir / "summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
