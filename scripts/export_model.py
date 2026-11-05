#!/usr/bin/env python3
"""Copy a trained adapter and its processor. Merging weights is not the default."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a Vaakku adapter")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if args.merge:
        print(
            "Merging the adapter into full Whisper weights is not the default. "
            "Publish mode stays peft_adapter.",
            file=sys.stderr,
        )
        return 2
    run_dir = Path(args.run_dir)
    adapter = run_dir / args.checkpoint / "adapter"
    if not adapter.is_dir():
        print(f"Adapter directory does not exist: {adapter}", file=sys.stderr)
        return 1
    output = Path(args.output)
    if output.exists():
        print(f"Output already exists: {output}", file=sys.stderr)
        return 1
    output.mkdir(parents=True)
    for path in adapter.iterdir():
        if path.is_file():
            shutil.copy2(path, output / path.name)
    processor = run_dir / "processor"
    if processor.is_dir():
        for path in processor.iterdir():
            if path.is_file():
                shutil.copy2(path, output / path.name)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
