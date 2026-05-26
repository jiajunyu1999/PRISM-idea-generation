from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from tqdm import tqdm


def model_cache_tag(model_name: str, max_length: int) -> str:
    raw = f"{model_name}::{max_length}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10]
    stem = Path(model_name).name.replace("/", "_")
    return f"{stem}.ml{max_length}.{digest}"


def tokenized_cache_path(
    processed_dir: Path,
    split: str,
    formatter: str,
    model_name: str,
    max_length: int,
    mode: Literal["train", "eval"],
    limit: int | None = None,
) -> Path:
    tag = model_cache_tag(model_name, max_length)
    suffix = f".limit{limit}" if limit is not None else ""
    return processed_dir / f"{split}.tokenized.{formatter}.{tag}.{mode}{suffix}.jsonl"


def build_tokenized_pair_cache(
    source_path: Path,
    output_path: Path,
    tokenizer,
    max_length: int,
    mode: Literal["train", "eval"],
    limit: int | None = None,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    total = limit if limit is not None else sum(1 for _ in source_path.open("r", encoding="utf-8"))
    with source_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        pbar = tqdm(total=total, desc=f"tokenize {source_path.stem} [{mode}]", unit="rows")
        for i, line in enumerate(src):
            if limit is not None and i >= limit:
                break
            row = json.loads(line)
            if mode == "train":
                chosen = tokenizer(
                    row["chosen_text"],
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_attention_mask=True,
                )
                rejected = tokenizer(
                    row["rejected_text"],
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_attention_mask=True,
                )
                payload = {
                    "chosen_input_ids": chosen["input_ids"],
                    "chosen_attention_mask": chosen["attention_mask"],
                    "rejected_input_ids": rejected["input_ids"],
                    "rejected_attention_mask": rejected["attention_mask"],
                    "abs_gap": row.get("abs_gap", 0.0),
                    "rel_gap": row.get("rel_gap", 0.0),
                    "pair_key": row.get("pair_key", ""),
                    "correct_answer": row.get("correct_answer", "A"),
                    "meta": row.get("meta", {}),
                }
            else:
                tok_a = tokenizer(
                    row["text_a"],
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_attention_mask=True,
                )
                tok_b = tokenizer(
                    row["text_b"],
                    max_length=max_length,
                    truncation=True,
                    padding=False,
                    return_attention_mask=True,
                )
                payload = {
                    "a_input_ids": tok_a["input_ids"],
                    "a_attention_mask": tok_a["attention_mask"],
                    "b_input_ids": tok_b["input_ids"],
                    "b_attention_mask": tok_b["attention_mask"],
                    "abs_gap": row.get("abs_gap", 0.0),
                    "rel_gap": row.get("rel_gap", 0.0),
                    "pair_key": row.get("pair_key", ""),
                    "correct_answer": row.get("correct_answer", "A"),
                    "meta": row.get("meta", {}),
                }
            dst.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
            pbar.update(1)
        pbar.close()
    return count
