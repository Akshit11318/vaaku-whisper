"""Names that stay stable across configs, logs, and the demo."""

from __future__ import annotations

PROJECT_NAME = "Vaakku — Open Indic Speech"
MODEL_FAMILY = "vaakku-whisper-small"
STUDIO_NAME = "Vaakku Studio"
GITHUB_REPOSITORY = "https://github.com/Akshit11318/vaaku-whisper"
HF_USERNAME = "rudra11"
HF_MODEL_ID = "rudra11/vaakku-whisper-small"
DEFAULT_BASE_MODEL_ID = "openai/whisper-small"

SUPPORTED_LANGUAGES: tuple[str, ...] = ("ta", "te", "hi", "kn", "ml")

LANGUAGE_NAMES: dict[str, str] = {
    "ta": "Tamil",
    "te": "Telugu",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
}

# Whisper's tokenizer expects these English names, not the ISO codes.
WHISPER_LANGUAGE_NAMES: dict[str, str] = {
    "ta": "tamil",
    "te": "telugu",
    "hi": "hindi",
    "kn": "kannada",
    "ml": "malayalam",
}

LANGUAGE_ALIASES: dict[str, str] = {
    "ta": "ta",
    "tam": "ta",
    "tamil": "ta",
    "te": "te",
    "tel": "te",
    "telugu": "te",
    "hi": "hi",
    "hin": "hi",
    "hindi": "hi",
    "kn": "kn",
    "kan": "kn",
    "kannada": "kn",
    "ml": "ml",
    "mal": "ml",
    "malayalam": "ml",
}

SPLIT_ORDER: tuple[str, ...] = ("train", "validation", "test")
SAMPLE_RATE = 16_000
