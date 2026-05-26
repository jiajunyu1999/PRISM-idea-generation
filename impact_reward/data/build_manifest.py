from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.manifest import Manifest, SplitManifest, save_manifest

SPLITS = ["train", "test", "test_ood_year", "test_ood_iclr"]


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifest for preprocessed SciJudge data")
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--processed_dir", type=Path, required=True)
    parser.add_argument("--formatter", type=str, default="paper_raw_v1")
    parser.add_argument("--output", type=Path, default=Path("reward_scijudge/outputs/processed/manifest.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = {}
    for split in SPLITS:
        standardized = args.processed_dir / f"{split}.standardized.jsonl"
        pairs = args.processed_dir / f"{split}.pairs.{args.formatter}.jsonl"
        if not standardized.exists() or not pairs.exists():
            raise FileNotFoundError(f"Missing preprocessed files for split={split}")
        splits[split] = SplitManifest(
            split=split,
            standardized_path=str(standardized.resolve()),
            pairs_path=str(pairs.resolve()),
            count=count_lines(standardized),
        )
    manifest = Manifest(
        dataset_root=str(args.dataset_dir.resolve()),
        processed_root=str(args.processed_dir.resolve()),
        formatter=args.formatter,
        splits=splits,
    )
    save_manifest(manifest, args.output)
    print(f"Saved manifest: {args.output}")


if __name__ == "__main__":
    main()
