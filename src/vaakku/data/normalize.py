"""Transcript and language cleanup. Invalid text is reported, not dropped quietly."""

from __future__ import annotations

import re
import unicodedata

from vaakku.constants import LANGUAGE_ALIASES

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)
_SPACE = re.compile(r"\s+")


def normalize_transcript(text: str) -> str:
    """NFC, strip zero-width characters, and collapse whitespace."""
    cleaned = unicodedata.normalize("NFC", text or "")
    cleaned = cleaned.translate(_ZERO_WIDTH)
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = _SPACE.sub(" ", cleaned).strip()
    return cleaned


def canonicalize_language(value: str | None) -> str | None:
    if value is None:
        return None
    key = value.strip().lower().replace("_", "-")
    if key in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[key]
    # FLEURS folder names look like hi_in. The language is the first piece.
    head = key.split("-")[0]
    return LANGUAGE_ALIASES.get(head)
