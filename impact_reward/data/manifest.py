from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict


@dataclass
class SplitManifest:
    split: str
    standardized_path: str
    pairs_path: str
    count: int


@dataclass
class Manifest:
    dataset_root: str
    processed_root: str
    formatter: str
    splits: Dict[str, SplitManifest]

    def to_dict(self) -> dict:
        return {
            "dataset_root": self.dataset_root,
            "processed_root": self.processed_root,
            "formatter": self.formatter,
            "splits": {k: asdict(v) for k, v in self.splits.items()},
        }


def save_manifest(manifest: Manifest, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")


def load_manifest(manifest_path: str | Path) -> Manifest:
    data = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    splits = {k: SplitManifest(**v) for k, v in data["splits"].items()}
    return Manifest(
        dataset_root=data["dataset_root"],
        processed_root=data["processed_root"],
        formatter=data["formatter"],
        splits=splits,
    )
