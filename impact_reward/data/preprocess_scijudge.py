from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.formatter import format_paper

SPLITS = ["train", "test", "test_ood_year", "test_ood_iclr"]


def read_jsonl(path: Path, max_samples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            if not line.strip():
                continue
            yield json.loads(line)


def extract_record(raw: Dict[str, Any], split: str) -> Dict[str, Any]:
    label = 1 if raw.get("correct_answer") == "A" else 0
    record = {
        "paper_a": {
            "title": raw.get("paper_a_title", ""),
            "abstract": raw.get("paper_a_abstract", ""),
            "date": raw.get("paper_a_date", ""),
            "category": raw.get("paper_a_category", ""),
            "subcategory": raw.get("paper_a_subcategory", ""),
            "citations": int(raw.get("paper_a_citations", 0) or 0),
            "arxiv_id": str(raw.get("paper_a_arxiv_id", "")),
        },
        "paper_b": {
            "title": raw.get("paper_b_title", ""),
            "abstract": raw.get("paper_b_abstract", ""),
            "date": raw.get("paper_b_date", ""),
            "category": raw.get("paper_b_category", ""),
            "subcategory": raw.get("paper_b_subcategory", ""),
            "citations": int(raw.get("paper_b_citations", 0) or 0),
            "arxiv_id": str(raw.get("paper_b_arxiv_id", "")),
        },
        "label": label,
        "correct_answer": "A" if label == 1 else "B",
        "split": split,
    }
    return record


def build_pair_key(arxiv_a: str, arxiv_b: str) -> str:
    a, b = sorted([arxiv_a or "", arxiv_b or ""])
    return f"{a}::{b}"


def paper_uid(paper: Dict[str, Any]) -> str:
    arxiv_id = str(paper.get("arxiv_id", "") or "").strip()
    if arxiv_id:
        return arxiv_id
    title = str(paper.get("title", "") or "").strip()
    abstract = str(paper.get("abstract", "") or "").strip()
    digest = hashlib.sha1(f"{title}||{abstract}".encode("utf-8")).hexdigest()[:20]
    return f"hash:{digest}"


def compute_gaps(cit_a: int, cit_b: int) -> Tuple[float, float]:
    abs_gap = float(abs(cit_a - cit_b))
    denom = float(max(cit_a, cit_b, 1))
    rel_gap = abs_gap / denom
    return abs_gap, rel_gap


def build_pair_example(standardized: Dict[str, Any], formatter_mode: str) -> Dict[str, Any]:
    paper_a = standardized["paper_a"]
    paper_b = standardized["paper_b"]

    chosen = paper_a if standardized["label"] == 1 else paper_b
    rejected = paper_b if standardized["label"] == 1 else paper_a

    abs_gap, rel_gap = compute_gaps(paper_a["citations"], paper_b["citations"])
    paper_a_uid = paper_uid(paper_a)
    paper_b_uid = paper_uid(paper_b)
    pair_key = build_pair_key(paper_a_uid, paper_b_uid)

    return {
        "chosen_text": format_paper(chosen, mode=formatter_mode),
        "rejected_text": format_paper(rejected, mode=formatter_mode),
        "text_a": format_paper(paper_a, mode=formatter_mode),
        "text_b": format_paper(paper_b, mode=formatter_mode),
        "correct_answer": standardized["correct_answer"],
        "abs_gap": abs_gap,
        "rel_gap": rel_gap,
        "pair_key": pair_key,
        "split": standardized["split"],
        "meta": {
            "chosen_id": paper_uid(chosen),
            "rejected_id": paper_uid(rejected),
            "paper_a_id": paper_a_uid,
            "paper_b_id": paper_b_uid,
            "citation_gap_abs": abs_gap,
            "citation_gap_rel": rel_gap,
            "category": chosen["category"] or rejected["category"],
            "subcategory_pair": [paper_a["subcategory"], paper_b["subcategory"]],
        },
    }


def preprocess_split(
    dataset_dir: Path,
    processed_dir: Path,
    split: str,
    formatter_mode: str,
    max_samples: Optional[int] = None,
) -> int:
    src = dataset_dir / f"{split}.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"Missing split file: {src}")

    standardized_path = processed_dir / f"{split}.standardized.jsonl"
    pairs_path = processed_dir / f"{split}.pairs.{formatter_mode}.jsonl"
    processed_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    with standardized_path.open("w", encoding="utf-8") as fs, pairs_path.open("w", encoding="utf-8") as fp:
        for raw in read_jsonl(src, max_samples=max_samples):
            std = extract_record(raw, split=split)
            pair = build_pair_example(std, formatter_mode=formatter_mode)
            fs.write(json.dumps(std, ensure_ascii=False) + "\n")
            fp.write(json.dumps(pair, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess SciJudgeBench into standardized and pair files")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--processed_dir", type=Path, default=Path("reward_scijudge/outputs/processed"))
    parser.add_argument("--formatter", type=str, default="paper_raw_v1")
    parser.add_argument("--splits", nargs="+", default=SPLITS)
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = {}
    for split in args.splits:
        count = preprocess_split(
            dataset_dir=args.dataset_dir,
            processed_dir=args.processed_dir,
            split=split,
            formatter_mode=args.formatter,
            max_samples=args.max_samples,
        )
        stats[split] = count
        print(f"[preprocess] {split}: {count} rows")
    print(json.dumps({"formatter": args.formatter, "stats": stats}, indent=2))


if __name__ == "__main__":
    main()
