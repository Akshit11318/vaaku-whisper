#!/usr/bin/env python3
"""Publish the evaluated adapter. This does not run from CI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from vaakku.config import load_publish_config
    from vaakku.errors import PublishError
    from vaakku.publishing import publish_model

    parser = argparse.ArgumentParser(description="Publish vaakku-whisper-small")
    parser.add_argument("--config", default="configs/publish.yaml")
    args = parser.parse_args()
    try:
        record = publish_model(load_publish_config(Path(args.config)))
    except PublishError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"SUCCESS {record['url']} revision={record['revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
