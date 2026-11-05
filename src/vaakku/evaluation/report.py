"""Write evaluation JSON, CSV, and Markdown from measured numbers only."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_reports(output_dir: Path, results: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "results.csv", results)
    (output_dir / "report.md").write_text(render_markdown(results), encoding="utf-8")


def render_markdown(results: dict) -> str:
    lines = [
        "# Evaluation report",
        "",
        f"- Split: {results.get('split', 'unknown')}",
        f"- Samples scored: {results.get('sample_count')}",
        f"- Empty references excluded: {results.get('empty_reference_count', 0)}",
        f"- Micro WER: {_fmt(results.get('micro_wer'))}",
        f"- Micro CER: {_fmt(results.get('micro_cer'))}",
        f"- Macro WER: {_fmt(results.get('macro_wer'))}",
        f"- Macro CER: {_fmt(results.get('macro_cer'))}",
        "",
        "Macro is the unweighted mean of the languages. A weak language stays visible below.",
        "",
        "## Per language",
        "",
        "| Language | Samples | WER | CER |",
        "|---|---:|---:|---:|",
    ]
    for row in results.get("per_language") or []:
        lines.append(
            f"| {row['language']} | {row['samples']} | {_fmt(row.get('wer'))} | {_fmt(row.get('cer'))} |"
        )
    if not results.get("per_language"):
        lines.append("| — | 0 | unavailable | unavailable |")
    comparison = results.get("base_comparison") or {}
    lines.extend(["", "## Base model", ""])
    if comparison.get("available"):
        lines.append(f"- Base micro WER: {_fmt(comparison.get('micro_wer'))}")
        lines.append(f"- Base macro WER: {_fmt(comparison.get('macro_wer'))}")
    else:
        reason = comparison.get("reason", "Base-model comparison is unavailable.")
        lines.append(f"- {reason}")
    memory = results.get("gpu_memory_mb")
    lines.extend(["", "## Runtime", ""])
    if memory is None:
        lines.append(f"- GPU memory: {results.get('gpu_memory_reason', 'unavailable')}")
    else:
        lines.append(f"- GPU memory (MB): {memory}")
    lines.append("")
    return "\n".join(lines)


def _write_csv(path: Path, results: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["language", "samples", "wer", "cer"])
        writer.writeheader()
        for row in results.get("per_language") or []:
            writer.writerow(
                {
                    "language": row["language"],
                    "samples": row["samples"],
                    "wer": row.get("wer"),
                    "cer": row.get("cer"),
                }
            )


def _fmt(value: object) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
