#!/usr/bin/env python3
"""Download the five FLEURS language packs onto the data disk."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> int:
    from huggingface_hub import snapshot_download

    data_root = Path(os.environ.get("VAAKKU_DATA_DIR", "/home/ubuntu/vaaku/data"))
    destination = data_root / "raw" / "fleurs"
    destination.mkdir(parents=True, exist_ok=True)
    languages = ["hi_in", "ta_in", "te_in", "kn_in", "ml_in"]
    patterns = [f"data/{language}/**" for language in languages] + ["README.md"]
    print("DOWNLOAD", patterns, flush=True)
    path = snapshot_download(
        repo_id="google/fleurs",
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(destination),
    )
    print("SAVED", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
