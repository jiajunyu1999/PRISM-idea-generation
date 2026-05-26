from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Dict, Optional

from tqdm import tqdm

SPLITS = ["train", "test", "test_ood_year", "test_ood_iclr"]


def read_jsonl(path: Path, max_samples: Optional[int] = None):
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            if line.strip():
                yield json.loads(line)


def stable_uid(arxiv_id: str, title: str, abstract: str) -> str:
    if arxiv_id and str(arxiv_id).strip():
        return str(arxiv_id).strip()
    digest = hashlib.sha1(f"{title}||{abstract}".encode("utf-8")).hexdigest()[:20]
    return f"hash:{digest}"


def analyze_split(path: Path, split: str, max_samples: Optional[int] = None) -> Dict:
    cnt = 0
    ans = Counter()
    category = Counter()
    subcategory = Counter()
    missing_title = 0
    missing_abs = 0
    abs_gaps = []
    rel_gaps = []
    pair_keys = set()

    for row in tqdm(read_jsonl(path, max_samples=max_samples), desc=f"inspect:{split}"):
        cnt += 1
        ans[row.get("correct_answer", "")] += 1

        cat_a = row.get("paper_a_category", "") or "unknown"
        cat_b = row.get("paper_b_category", "") or "unknown"
        category[cat_a] += 1
        category[cat_b] += 1

        sub_a = row.get("paper_a_subcategory", "") or "unknown"
        sub_b = row.get("paper_b_subcategory", "") or "unknown"
        subcategory[sub_a] += 1
        subcategory[sub_b] += 1

        if not str(row.get("paper_a_title", "")).strip() or not str(row.get("paper_b_title", "")).strip():
            missing_title += 1
        if not str(row.get("paper_a_abstract", "")).strip() or not str(row.get("paper_b_abstract", "")).strip():
            missing_abs += 1

        cit_a = int(row.get("paper_a_citations", 0) or 0)
        cit_b = int(row.get("paper_b_citations", 0) or 0)
        abs_gap = abs(cit_a - cit_b)
        rel_gap = abs_gap / max(cit_a, cit_b, 1)
        abs_gaps.append(abs_gap)
        rel_gaps.append(rel_gap)

        aid = stable_uid(
            str(row.get("paper_a_arxiv_id", "")),
            str(row.get("paper_a_title", "")),
            str(row.get("paper_a_abstract", "")),
        )
        bid = stable_uid(
            str(row.get("paper_b_arxiv_id", "")),
            str(row.get("paper_b_title", "")),
            str(row.get("paper_b_abstract", "")),
        )
        key = tuple(sorted([aid, bid]))
        pair_keys.add(key)

    return {
        "split": split,
        "count": cnt,
        "correct_answer_dist": dict(ans),
        "top_categories": category.most_common(20),
        "top_subcategories": subcategory.most_common(20),
        "missing_title_ratio": missing_title / max(cnt, 1),
        "missing_abstract_ratio": missing_abs / max(cnt, 1),
        "citation_gap_abs_mean": mean(abs_gaps) if abs_gaps else 0.0,
        "citation_gap_rel_mean": mean(rel_gaps) if rel_gaps else 0.0,
        "unique_pair_count": len(pair_keys),
    }


def render_markdown(profile: Dict) -> str:
    lines = ["# SciJudgeBench Data Profile", ""]
    for split, info in profile["splits"].items():
        lines.append(f"## {split}")
        lines.append(f"- count: {info['count']}")
        lines.append(f"- unique_pair_count: {info['unique_pair_count']}")
        lines.append(f"- correct_answer_dist: {info['correct_answer_dist']}")
        lines.append(f"- missing_title_ratio: {info['missing_title_ratio']:.4f}")
        lines.append(f"- missing_abstract_ratio: {info['missing_abstract_ratio']:.4f}")
        lines.append(f"- citation_gap_abs_mean: {info['citation_gap_abs_mean']:.4f}")
        lines.append(f"- citation_gap_rel_mean: {info['citation_gap_rel_mean']:.4f}")
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect SciJudgeBench and export data profile reports")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--reports_dir", type=Path, default=Path("reward_scijudge/reports"))
    parser.add_argument("--max_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    result = {"splits": {}}
    for split in SPLITS:
        path = args.dataset_dir / f"{split}.jsonl"
        result["splits"][split] = analyze_split(path, split, max_samples=args.max_samples)

    json_path = args.reports_dir / "data_profile.json"
    md_path = args.reports_dir / "data_profile.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(result), encoding="utf-8")
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
