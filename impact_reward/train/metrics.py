from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


def pair_accuracy(score_a: Sequence[float], score_b: Sequence[float], labels: Sequence[str]) -> float:
    correct = 0
    total = len(labels)
    for sa, sb, y in zip(score_a, score_b, labels):
        pred = "A" if sa > sb else "B"
        correct += int(pred == y)
    return correct / max(total, 1)


def pair_auc(margins: Sequence[float], labels: Sequence[str]) -> float:
    y_true = np.array([1 if y == "A" else 0 for y in labels])
    try:
        return float(roc_auc_score(y_true, np.array(margins)))
    except ValueError:
        return float("nan")


def mean_margin(score_a: Sequence[float], score_b: Sequence[float]) -> Dict[str, float]:
    margins = np.array(score_a) - np.array(score_b)
    return {
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
        "std_margin": float(np.std(margins)),
    }


def swap_consistency(score_a: Sequence[float], score_b: Sequence[float]) -> float:
    # For pointwise scorer this should be 1.0; we keep it explicit for API-level checks.
    margins = np.array(score_a) - np.array(score_b)
    reverse = np.array(score_b) - np.array(score_a)
    consistent = np.sign(margins) == -np.sign(reverse)
    return float(np.mean(consistent.astype(np.float32)))


def dedup_pair_accuracy(
    score_a: Sequence[float],
    score_b: Sequence[float],
    labels: Sequence[str],
    pair_keys: Sequence[str],
) -> float:
    buckets = defaultdict(list)
    for sa, sb, y, k in zip(score_a, score_b, labels, pair_keys):
        buckets[k].append((sa, sb, y))

    correct = 0
    total = 0
    for _, entries in buckets.items():
        avg_sa = float(np.mean([e[0] for e in entries]))
        avg_sb = float(np.mean([e[1] for e in entries]))
        # All prompt variants should share same label in SciJudgeBench.
        label = entries[0][2]
        pred = "A" if avg_sa > avg_sb else "B"
        correct += int(pred == label)
        total += 1
    return correct / max(total, 1)


def category_breakdown(
    score_a: Sequence[float],
    score_b: Sequence[float],
    labels: Sequence[str],
    categories: Sequence[str],
) -> Dict[str, float]:
    by_cat = defaultdict(lambda: {"correct": 0, "total": 0})
    for sa, sb, y, c in zip(score_a, score_b, labels, categories):
        pred = "A" if sa > sb else "B"
        bucket = by_cat[c or "unknown"]
        bucket["correct"] += int(pred == y)
        bucket["total"] += 1
    return {k: v["correct"] / max(v["total"], 1) for k, v in by_cat.items()}


def _normalize_domain(category: str) -> str:
    c = (category or "").strip().lower()
    if "computer science" in c or c.startswith("cs") or "cs." in c:
        return "CS"
    if "mathemat" in c or c.startswith("math") or "math." in c:
        return "Math"
    if "physics" in c or c.startswith("physics") or "phys." in c:
        return "Physics"
    return "Others"


def domain_breakdown(
    score_a: Sequence[float],
    score_b: Sequence[float],
    labels: Sequence[str],
    categories: Sequence[str],
) -> Dict[str, float]:
    buckets = {"CS": {"correct": 0, "total": 0}, "Math": {"correct": 0, "total": 0}, "Physics": {"correct": 0, "total": 0}, "Others": {"correct": 0, "total": 0}}
    for sa, sb, y, c in zip(score_a, score_b, labels, categories):
        pred = "A" if sa > sb else "B"
        bucket = buckets[_normalize_domain(c)]
        bucket["correct"] += int(pred == y)
        bucket["total"] += 1
    return {k: v["correct"] / max(v["total"], 1) for k, v in buckets.items()}


def calibration_curve(scores: Sequence[float], labels: Sequence[int], n_bins: int = 10) -> Dict[str, List[float]]:
    s = np.array(scores)
    y = np.array(labels)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(s, bins) - 1
    bin_acc, bin_conf = [], []
    for b in range(n_bins):
        m = idx == b
        if np.any(m):
            bin_acc.append(float(np.mean(y[m])))
            bin_conf.append(float(np.mean(s[m])))
    return {"bin_acc": bin_acc, "bin_conf": bin_conf}
