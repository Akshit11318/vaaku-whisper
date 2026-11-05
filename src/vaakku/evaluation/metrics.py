"""Aggregate WER and CER without hiding a language inside one average."""

from __future__ import annotations

import math
from collections import defaultdict

from vaakku.evaluation.text import NORMALIZATION, char_errors, normalize_for_eval, word_errors


def score_predictions(rows: list[dict]) -> dict:
    """Each row needs reference, hypothesis, and language.

    Optional keys: latency_s, audio_duration_s.
    Empty references are counted and excluded from the rates.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    empty_reference = 0
    for row in rows:
        reference = normalize_for_eval(str(row.get("reference", "")))
        hypothesis = normalize_for_eval(str(row.get("hypothesis", "")))
        if not reference:
            empty_reference += 1
            continue
        grouped[str(row["language"])].append(
            {
                "reference": reference,
                "hypothesis": hypothesis,
                "latency_s": row.get("latency_s"),
                "audio_duration_s": row.get("audio_duration_s"),
            }
        )
    per_language = []
    micro_words_ref = 0
    micro_words_err = 0.0
    micro_chars_ref = 0
    micro_chars_err = 0.0
    for language in sorted(grouped):
        items = grouped[language]
        word_err = 0
        word_ref = 0
        char_err = 0
        char_ref = 0
        for item in items:
            w_err, w_ref = word_errors(item["reference"], item["hypothesis"])
            c_err, c_ref = char_errors(item["reference"], item["hypothesis"])
            word_err += w_err
            word_ref += w_ref
            char_err += c_err
            char_ref += c_ref
        micro_words_ref += word_ref
        micro_words_err += word_err
        micro_chars_ref += char_ref
        micro_chars_err += char_err
        per_language.append(
            {
                "language": language,
                "samples": len(items),
                "wer": None if word_ref == 0 else word_err / word_ref,
                "cer": None if char_ref == 0 else char_err / char_ref,
            }
        )
    latencies = [float(row["latency_s"]) for row in rows if row.get("latency_s") is not None]
    audio_seconds = [
        float(row["audio_duration_s"]) for row in rows if row.get("audio_duration_s") is not None
    ]
    total_latency = sum(latencies)
    return {
        "available": bool(per_language),
        "normalization": NORMALIZATION,
        "sample_count": sum(item["samples"] for item in per_language),
        "empty_reference_count": empty_reference,
        "per_language": per_language,
        "micro_wer": None if micro_words_ref == 0 else micro_words_err / micro_words_ref,
        "micro_cer": None if micro_chars_ref == 0 else micro_chars_err / micro_chars_ref,
        "macro_wer": _mean([item["wer"] for item in per_language]),
        "macro_cer": _mean([item["cer"] for item in per_language]),
        "latency_seconds": _latency(latencies),
        "throughput_samples_per_second": (
            None if total_latency <= 0 else round(len(latencies) / total_latency, 4)
        ),
        "throughput_audio_seconds_per_second": (
            None
            if total_latency <= 0 or not audio_seconds
            else round(sum(audio_seconds) / total_latency, 4)
        ),
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _latency(values: list[float]) -> dict | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "mean": _mean(ordered),
        "p50": _percentile(ordered, 0.50),
        "p95": _percentile(ordered, 0.95),
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
