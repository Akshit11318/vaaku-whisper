"""Audio loading, resampling, and content hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from vaakku.constants import SAMPLE_RATE
from vaakku.errors import DataError


def read_audio(path: Path, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, float]:
    """Return mono float32 audio at target_sr and its duration in seconds."""
    if not path.is_file():
        raise DataError(f"Audio file does not exist: {path}")
    try:
        audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    except Exception as exc:  # soundfile raises several types for corrupt files
        raise DataError(f"Unreadable audio: {path}") from exc
    array = np.asarray(audio, dtype=np.float32)
    if array.size == 0:
        raise DataError(f"Empty audio: {path}")
    if array.ndim == 2:
        array = array.mean(axis=1).astype(np.float32)
    if int(sample_rate) != target_sr:
        array = _resample(array, int(sample_rate), target_sr)
    duration = float(array.shape[0]) / float(target_sr)
    return array, duration


def audio_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    from math import gcd

    divisor = gcd(source_sr, target_sr)
    converted = resample_poly(audio, target_sr // divisor, source_sr // divisor)
    return np.asarray(converted, dtype=np.float32)
