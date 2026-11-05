"""Shared audio fixtures."""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path


def write_tone(
    path: Path, seconds: float = 1.0, sample_rate: int = 16000, frequency: float = 440.0
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(seconds * sample_rate)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(count):
            value = int(8000 * math.sin(2 * math.pi * frequency * index / sample_rate))
            frames.extend(struct.pack("<h", value))
        handle.writeframes(frames)
