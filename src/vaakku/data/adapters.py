"""Source adapters. They read local files. Downloads stay behind allow_download."""

from __future__ import annotations

import csv
import json
import tarfile
from collections.abc import Iterator
from pathlib import Path

from vaakku.config.models import DataSourceConfig
from vaakku.constants import SAMPLE_RATE
from vaakku.data.normalize import canonicalize_language
from vaakku.errors import DataError

FLEURS_SPLIT_FILES = {
    "train": "train",
    "dev": "validation",
    "validation": "validation",
    "test": "test",
}


def iter_source_rows(source: DataSourceConfig) -> Iterator[dict]:
    if source.kind == "vistaar":
        raise DataError(
            "Vistaar is a benchmark collection, not a training download. "
            "Leave it disabled so FLEURS test is not mixed back into training."
        )
    if source.kind == "fleurs":
        yield from iter_fleurs(source)
        return
    if source.kind in {"manifest", "common_voice", "indicvoices", "kathbath"}:
        yield from iter_table(source)
        return
    raise DataError(f"Unknown source kind: {source.kind}")


def iter_table(source: DataSourceConfig) -> Iterator[dict]:
    if not source.path:
        raise DataError(f"{source.name} has no local path")
    path = Path(source.path)
    if not path.exists():
        raise DataError(f"{source.name} path does not exist: {path}")
    if path.suffix == ".jsonl":
        yield from _jsonl(path, source.name)
        return
    if path.suffix == ".tsv":
        yield from _tsv(path, source)
        return
    raise DataError(f"{source.name} expects a .jsonl or .tsv file, got {path}")


def iter_fleurs(source: DataSourceConfig) -> Iterator[dict]:
    if source.path:
        path = Path(source.path)
        if not path.exists():
            raise DataError(f"FLEURS path does not exist: {path}")
        if path.suffix == ".jsonl":
            yield from _jsonl(path, source.name)
            return
        yield from _fleurs_directory(path, source.name)
        return
    if not source.allow_download:
        raise DataError(f"{source.name} needs a local FLEURS directory")
    yield from _fleurs_hub(source)


def _jsonl(path: Path, dataset_name: str) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                yield {
                    "__parse_error__": str(exc),
                    "__line__": line_number,
                    "dataset_name": dataset_name,
                }
                continue
            if isinstance(row, dict):
                row.setdefault("dataset_name", dataset_name)
                yield row


def _tsv(path: Path, source: DataSourceConfig) -> Iterator[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            mapped = {
                "dataset_name": source.name,
                "audio_path": row.get("path") or row.get("audio_path") or "",
                "transcription": row.get("sentence")
                or row.get("text")
                or row.get("transcription")
                or "",
                "language": row.get("locale") or row.get("language") or row.get("lang"),
                "speaker_id": row.get("client_id") or row.get("speaker_id"),
                "sample_id": row.get("path") or row.get("id"),
            }
            yield {key: value for key, value in mapped.items() if value not in (None, "")}


def _fleurs_directory(root: Path, dataset_name: str) -> Iterator[dict]:
    tsv_files = sorted(root.rglob("*.tsv"))
    if not tsv_files:
        raise DataError(f"No FLEURS tsv files under {root}")
    for tsv in tsv_files:
        split = FLEURS_SPLIT_FILES.get(tsv.stem)
        if split is None:
            continue
        language = canonicalize_language(tsv.parent.name)
        audio_root = _extract_fleurs_audio(tsv.parent / "audio")
        with tsv.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    yield {
                        "__parse_error__": "short row",
                        "dataset_name": dataset_name,
                        "language": language,
                        "split": split,
                    }
                    continue
                filename = parts[1]
                transcription = parts[3] or parts[2]
                num_samples = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else None
                wav = _find_audio(audio_root / tsv.stem, filename) if audio_root else None
                yield {
                    "dataset_name": dataset_name,
                    "audio_path": str(wav) if wav else "",
                    "transcription": transcription,
                    "language": language,
                    "sample_id": f"{language}-{split}-{filename}",
                    "split": split,
                    "duration_s": (num_samples / SAMPLE_RATE) if num_samples else None,
                    "__missing_audio__": wav is None,
                }


def _extract_fleurs_audio(audio_dir: Path) -> Path | None:
    if not audio_dir.is_dir():
        return None
    extracted = audio_dir / "extracted"
    marker = extracted / ".complete"
    archives = sorted(audio_dir.glob("*.tar.gz"))
    if marker.is_file() or not archives:
        return extracted if extracted.is_dir() else audio_dir
    extracted.mkdir(parents=True, exist_ok=True)
    for archive in archives:
        destination = extracted / archive.name.split(".")[0]
        destination.mkdir(parents=True, exist_ok=True)
        _safe_extract(archive, destination)
    marker.write_text("ok\n", encoding="utf-8")
    return extracted


def _safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise DataError(f"Refusing to extract unsafe path {member.name} from {archive}")
        handle.extractall(destination)


def _find_audio(directory: Path, filename: str) -> Path | None:
    direct = directory / filename
    if direct.is_file():
        return direct
    matches = [path for path in directory.rglob(filename) if path.is_file()]
    return matches[0] if matches else None


def _fleurs_hub(source: DataSourceConfig) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise DataError("Install the datasets package before allow_download") from exc
    dataset = load_dataset(source.hf_dataset, source.hf_config, split=source.hf_split)
    for row in dataset:
        audio = row.get("audio") or {}
        yield {
            "dataset_name": source.name,
            "audio_path": audio.get("path", ""),
            "transcription": row.get("transcription") or row.get("raw_transcription") or "",
            "language": row.get("language") or source.hf_config,
            "sample_id": str(row.get("id")),
            "split": source.hf_split,
        }
