#!/usr/bin/env python3
"""Score a held-out test manifest. This refuses a train split."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "src"))
    from vaakku.config import load_eval_config
    from vaakku.data.audio import read_audio
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
    rows = load_test_rows(config)
    total = len(rows)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["language"]].append(row)
    scored = 0
    for language, group in grouped.items():
        for start in range(0, len(group), config.batch_size):
            chunk = group[start : start + config.batch_size]
            audios = []
            durations = []
            for row in chunk:
                audio, duration = read_audio(Path(row["audio_path"]))
                audios.append(audio)
                durations.append(duration)
            started = time.perf_counter()
            transcripts = engine.transcribe_batch(audios, language)
            per_clip = (time.perf_counter() - started) / len(chunk)
            for row, transcript, duration in zip(chunk, transcripts, durations, strict=True):
                predictions.append(
                    {
                        "reference": row["transcription"],
                        "hypothesis": transcript,
                        "language": row["language"],
                        "latency_s": round(per_clip, 4),
                        "audio_duration_s": round(duration, 4),
                    }
                )
            scored += len(chunk)
            if scored == len(chunk) or scored % 100 < config.batch_size or scored == total:
                print(f"scored {scored}/{total}", flush=True)
    output = write_evaluation(config, predictions, gpu_memory_mb=None)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
