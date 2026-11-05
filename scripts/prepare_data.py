#!/usr/bin/env python3
"""Prepare local datasets into train, validation, and test manifests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from vaakku.config import load_data_config
    from vaakku.data import prepare_dataset

    parser = argparse.ArgumentParser(description="Prepare Vaakku training data")
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    config = load_data_config(Path(args.config))
    report = prepare_dataset(config)
    print(
        f"accepted={report.accepted} rejected={report.rejected} "
        f"method={report.split_method} output={config.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
