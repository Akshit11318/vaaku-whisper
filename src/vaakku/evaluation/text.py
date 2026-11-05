"""Text normalization and error rates for evaluation.

Word and character error rates use Levenshtein distance, the same definition
as the Hugging Face Evaluate `wer` and `cer` metrics. Scoring stays local so
evaluation does not download a metric script.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACE = re.compile(r"\s+")
NORMALIZATION = "NFC, strip punctuation, collapse whitespace, casefold"


def normalize_for_eval(text: str) -> str:
    cleaned = unicodedata.normalize("NFC", text or "")
    cleaned = _PUNCT.sub(" ", cleaned)
    cleaned = cleaned.casefold()
    return _SPACE.sub(" ", cleaned).strip()


def word_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    return _distance(ref_words, hyp_words), len(ref_words)


def char_errors(reference: str, hypothesis: str) -> tuple[int, int]:
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    return _distance(ref_chars, hyp_chars), len(ref_chars)


def wer(reference: str, hypothesis: str) -> float:
    errors, total = word_errors(reference, hypothesis)
    if total == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / total


def cer(reference: str, hypothesis: str) -> float:
    errors, total = char_errors(reference, hypothesis)
    if total == 0:
        return 0.0 if errors == 0 else 1.0
    return errors / total


def _distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, left_token in enumerate(left, start=1):
        current = [i]
        for j, right_token in enumerate(right, start=1):
            substitute = previous[j - 1] + (left_token != right_token)
            current.append(min(previous[j] + 1, current[j - 1] + 1, substitute))
        previous = current
    return previous[-1]
