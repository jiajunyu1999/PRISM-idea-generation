from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained checkpoint")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        shutil.rmtree(args.output)
    shutil.copytree(args.ckpt, args.output)
    meta = {"exported_from": str(args.ckpt.resolve())}
    (args.output / "export_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Exported checkpoint to {args.output}")


if __name__ == "__main__":
    main()
