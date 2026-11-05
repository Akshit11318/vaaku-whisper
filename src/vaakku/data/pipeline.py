"""Turn configured sources into split manifests and a quality report."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from vaakku.config.models import DataConfig
from vaakku.data.adapters import iter_source_rows
from vaakku.data.audio import audio_sha256, read_audio
from vaakku.data.normalize import canonicalize_language, normalize_transcript
from vaakku.data.schema import NormalizedSample, QualityReport, Rejection
from vaakku.data.split import assign_splits, official_speaker_overlap
from vaakku.errors import DataError
from vaakku.logging import log_event, setup_logging


def prepare_dataset(config: DataConfig) -> QualityReport:
    logger = setup_logging("data", config.logging)
    accepted: list[NormalizedSample] = []
    rejected: list[Rejection] = []
    seen_hashes: dict[str, str] = {}
    enabled = [source for source in config.sources if source.enabled]
    if not enabled:
        report = QualityReport(
            accepted=0,
            rejected=0,
            split_method="none",
            notes=["No sources are enabled. Nothing was downloaded or discarded."],
        )
        _write_report(Path(config.output_dir), report, {})
        log_event(logger, "data prepare finished", enabled_sources=0, accepted=0)
        return report

    for source in enabled:
        log_event(logger, "reading source", source=source.name, kind=source.kind)
        for row in iter_source_rows(source):
            sample, rejection = _normalize_row(row, source.name, config)
            if rejection:
                rejected.append(rejection)
                continue
            assert sample is not None
            digest = audio_sha256(Path(sample.audio_path))
            previous = seen_hashes.get(digest)
            if previous:
                rejected.append(
                    Rejection(
                        sample_id=sample.sample_id,
                        dataset_name=sample.dataset_name,
                        reason="duplicate_audio",
                        detail=f"same audio bytes as {previous}",
                        language=sample.language,
                    )
                )
                continue
            seen_hashes[digest] = sample.sample_id
            accepted.append(sample)

    ratios = config.splits.as_dict()
    buckets, method = assign_splits(accepted, ratios, config.seed, config.speaker_disjoint)
    overlap = official_speaker_overlap(buckets)
    reasons = Counter(item.reason for item in rejected)
    per_language = Counter(item.language for item in accepted)
    report = QualityReport(
        accepted=len(accepted),
        rejected=len(rejected),
        rejection_reasons=dict(reasons),
        per_language=dict(per_language),
        duration_s_total=round(sum(item.duration_s for item in accepted), 3),
        split_method=method,
        split_counts={name: len(rows) for name, rows in buckets.items()},
        official_speaker_overlap=overlap,
        notes=_notes(overlap),
        rejections=rejected,
    )
    _write_report(Path(config.output_dir), report, buckets)
    log_event(
        logger,
        "data prepare finished",
        accepted=report.accepted,
        rejected=report.rejected,
        reasons=report.rejection_reasons,
        per_language=report.per_language,
        split_method=method,
    )
    return report


def _normalize_row(
    row: dict,
    dataset_name: str,
    config: DataConfig,
) -> tuple[NormalizedSample | None, Rejection | None]:
    name = str(row.get("dataset_name") or dataset_name)
    sample_id = _as_text(row.get("sample_id") or row.get("id"))
    if row.get("__parse_error__"):
        return None, Rejection(
            sample_id=sample_id,
            dataset_name=name,
            reason="parse_error",
            detail=str(row["__parse_error__"]),
        )
    language = canonicalize_language(
        _as_text(row.get("language") or row.get("lang") or row.get("locale"))
    )
    if language is None:
        return None, Rejection(
            sample_id=sample_id,
            dataset_name=name,
            reason="missing_language",
            detail="language is missing or unrecognized",
        )
    if language not in config.languages:
        return None, Rejection(
            sample_id=sample_id,
            dataset_name=name,
            language=language,
            reason="unsupported_language",
            detail=language,
        )
    transcription = normalize_transcript(
        _as_text(
            row.get("transcription")
            or row.get("text")
            or row.get("sentence")
            or row.get("transcript")
        )
        or ""
    )
    if not transcription:
        return None, Rejection(
            sample_id=sample_id,
            dataset_name=name,
            language=language,
            reason="empty_transcript",
        )
    audio_path = _as_text(row.get("audio_path") or row.get("path") or row.get("audio_filepath"))
    if not audio_path or row.get("__missing_audio__"):
        return None, Rejection(
            sample_id=sample_id,
            dataset_name=name,
            language=language,
            reason="missing_audio",
            detail=audio_path or "",
        )
    path = Path(audio_path)
    try:
        _audio, duration = read_audio(path, config.sample_rate)
    except DataError as exc:
        return None, Rejection(
            sample_id=sample_id,
            dataset_name=name,
            language=language,
            reason="invalid_audio",
            detail=str(exc),
        )
    if duration < config.min_duration_s:
        return None, Rejection(
            sample_id=sample_id,
            dataset_name=name,
            language=language,
            reason="too_short",
            detail=f"duration_s={duration:.3f}",
        )
    if duration > config.max_duration_s:
        return None, Rejection(
            sample_id=sample_id,
            dataset_name=name,
            language=language,
            reason="too_long",
            detail=f"duration_s={duration:.3f}",
        )
    resolved_id = sample_id or f"{name}:{path.name}"
    speaker = _as_text(row.get("speaker_id") or row.get("speaker") or row.get("client_id"))
    split = _as_text(row.get("split"))
    return (
        NormalizedSample(
            audio_path=str(path),
            transcription=transcription,
            language=language,
            speaker_id=speaker,
            dataset_name=name,
            sample_id=resolved_id,
            duration_s=round(duration, 6),
            split=split,
        ),
        None,
    )


def _notes(overlap: list[str]) -> list[str]:
    notes = [
        "Official splits are preserved. FLEURS test rows are not reassigned.",
        "Rejection rows omit transcript text.",
    ]
    if overlap:
        notes.append("Some official splits share speaker ids. Those rows were left in place.")
    return notes


def _write_report(
    output_dir: Path,
    report: QualityReport,
    buckets: dict[str, list[NormalizedSample]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in buckets.items():
        _write_jsonl(output_dir / f"{split}.jsonl", rows, split)
    payload = report.model_dump()
    (output_dir / "quality_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "quality_report.md").write_text(_markdown(report), encoding="utf-8")
    (output_dir / "split_manifest.json").write_text(
        json.dumps(
            {
                "method": report.split_method,
                "counts": report.split_counts,
                "official_speaker_overlap": report.official_speaker_overlap,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[NormalizedSample], split: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = row.model_dump()
            payload["split"] = split
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _markdown(report: QualityReport) -> str:
    lines = [
        "# Data quality report",
        "",
        f"- Accepted: {report.accepted}",
        f"- Rejected: {report.rejected}",
        f"- Split method: {report.split_method}",
        f"- Total accepted duration (seconds): {report.duration_s_total}",
        "",
        "## Per language",
        "",
    ]
    if report.per_language:
        for language, count in sorted(report.per_language.items()):
            lines.append(f"- {language}: {count}")
    else:
        lines.append("- No accepted samples.")
    lines.extend(["", "## Rejection reasons", ""])
    if report.rejection_reasons:
        for reason, count in sorted(report.rejection_reasons.items()):
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- No rejections.")
    lines.extend(["", "## Notes", ""])
    for note in report.notes:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def _as_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
