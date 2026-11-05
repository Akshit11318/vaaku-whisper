"""Turn prepared manifests into model features before training."""

from __future__ import annotations

from pathlib import Path

from vaakku.config.models import TrainConfig
from vaakku.constants import WHISPER_LANGUAGE_NAMES
from vaakku.data.audio import read_audio
from vaakku.errors import TrainingError


def rows_from_manifest(config: TrainConfig, split: str, processor) -> list[dict]:
    path_text = config.train_manifest if split == "train" else config.validation_manifest
    if not path_text:
        raise TrainingError(f"Missing {split} manifest")
    path = Path(path_text)
    if not path.is_file():
        raise TrainingError(f"Manifest not found: {path}")
    import json

    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    limit = config.max_train_samples if split == "train" else config.max_eval_samples
    if limit is not None:
        rows = rows[:limit]
    prepared = []
    for row in rows:
        if row.get("split") not in (None, split):
            raise TrainingError(f"{path} contains split {row.get('split')!r}, expected {split}")
        language = row["language"]
        if language not in config.languages:
            continue
        audio, _duration = read_audio(Path(row["audio_path"]), config.sample_rate)
        features = extract_log_mel(processor.feature_extractor, audio, config.sample_rate)
        processor.tokenizer.set_prefix_tokens(
            language=WHISPER_LANGUAGE_NAMES[language],
            task="transcribe",
        )
        labels = processor.tokenizer(row["transcription"]).input_ids
        prepared.append(
            {
                "input_features": features,
                "labels": list(labels),
                "duration_s": float(row["duration_s"]),
                "language": language,
                "sample_id": row.get("sample_id"),
            }
        )
    if not prepared:
        raise TrainingError(f"No usable rows in {path}")
    return prepared


def extract_log_mel(feature_extractor, audio, sample_rate: int):
    kwargs = {
        "sampling_rate": sample_rate,
        "return_tensors": "np",
        "truncation": False,
        "return_attention_mask": False,
        "padding": False,
    }
    try:
        outputs = feature_extractor(audio, **kwargs)
    except (TypeError, ValueError):
        kwargs["padding"] = "do_not_pad"
        outputs = feature_extractor(audio, **kwargs)
    array = outputs["input_features"]
    if getattr(array, "ndim", 0) == 3:
        array = array[0]
    return array


def load_processor(model_id: str):
    from transformers import WhisperProcessor

    return WhisperProcessor.from_pretrained(model_id)
