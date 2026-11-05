#!/usr/bin/env python3
"""Score a held-out test manifest. This refuses a train split."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from vaakku.config import load_eval_config
    from vaakku.evaluation import load_test_rows, write_evaluation
    from vaakku.inference.engine import load_engine_from_env

    parser = argparse.ArgumentParser(description="Evaluate a Vaakku checkpoint")
    parser.add_argument("--config", default="configs/evaluation.yaml")
    args = parser.parse_args()
    config = load_eval_config(Path(args.config))
    if not config.model_path:
        print(
            "model_path is empty. Point evaluation.yaml at a trained checkpoint.", file=sys.stderr
        )
        return 2
    os.environ["VAAKKU_MODEL_PATH"] = config.model_path
    engine = load_engine_from_env()
    status = engine.status()
    if not status.ready:
        print(status.reason or "Model not configured", file=sys.stderr)
        return 2
    predictions = []
    for row in load_test_rows(config):
        payload = Path(row["audio_path"]).read_bytes()
        result = engine.transcribe(
            payload, suffix=Path(row["audio_path"]).suffix, language=row["language"]
        )
        predictions.append(
            {
                "reference": row["transcription"],
                "hypothesis": result.transcript,
                "language": row["language"],
                "latency_s": result.processing_time_s,
                "audio_duration_s": result.audio_duration_s,
            }
        )
    output = write_evaluation(config, predictions, gpu_memory_mb=None)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
