#!/usr/bin/env python3
"""Fine-tune Whisper with LoRA. Quick runs are pipeline checks, not benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from vaakku.config import load_train_config
    from vaakku.training.features import load_processor, rows_from_manifest
    from vaakku.training.loop import run_training

    parser = argparse.ArgumentParser(description="Train vaakku-whisper-small")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()
    config = load_train_config(Path(args.config))
    train_rows = None
    eval_rows = None
    if not config.synthetic and config.model_init != "tiny_random":
        processor = load_processor(config.base_model_id)
        train_rows = rows_from_manifest(config, "train", processor)
        eval_rows = rows_from_manifest(config, "validation", processor)
        processor.save_pretrained(Path(config.output_dir) / "_processor_tmp")
    output = run_training(
        config, resume_from=args.resume, train_rows=train_rows, eval_rows=eval_rows
    )
    if not config.synthetic and config.model_init != "tiny_random":
        saved = Path(config.output_dir) / "_processor_tmp"
        destination = output / "processor"
        if saved.exists():
            destination.mkdir(parents=True, exist_ok=True)
            for path in saved.iterdir():
                path.replace(destination / path.name)
            saved.rmdir()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
