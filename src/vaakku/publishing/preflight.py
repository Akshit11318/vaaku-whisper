"""Checks that must pass before a model upload."""

from __future__ import annotations

import json
from pathlib import Path

from vaakku.config.models import PublishConfig

ADAPTER_FILES = ("adapter_config.json",)
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
PROCESSOR_MARKERS = ("preprocessor_config.json", "tokenizer_config.json")
PLACEHOLDERS = ("REPLACE", "UNVERIFIED", "TODO")


def preflight(config: PublishConfig) -> list[str]:
    problems: list[str] = []
    run_dir = Path(config.run_dir)
    if not run_dir.is_dir():
        return [f"Training run directory does not exist: {run_dir}"]
    run_meta = _read_json(run_dir / "run.json")
    if run_meta is None:
        problems.append("run.json is missing, so training metadata cannot be verified")
    elif run_meta.get("purpose") == "pipeline_validation":
        problems.append("Refusing to publish a pipeline-validation run")
    checkpoint = run_dir / config.checkpoint
    if not checkpoint.is_dir():
        problems.append(f"Selected checkpoint does not exist: {checkpoint}")
    else:
        adapter = checkpoint / "adapter"
        if not adapter.is_dir():
            problems.append(f"Adapter directory is missing: {adapter}")
        else:
            for name in ADAPTER_FILES:
                if not (adapter / name).is_file():
                    problems.append(f"Missing adapter file: {name}")
            if not any((adapter / name).is_file() for name in ADAPTER_WEIGHTS):
                problems.append("Adapter weights are missing")
        best_state = _read_json(checkpoint / "training_state.json")
        if config.checkpoint == "best" and best_state is None:
            problems.append(
                "best/training_state.json is missing, so the best checkpoint cannot be confirmed"
            )
    eval_path = Path(config.eval_results) if config.eval_results else _find_eval(run_dir)
    if eval_path is None:
        problems.append("Evaluation results are missing. Run evaluation before publishing.")
    else:
        evaluation = _read_json(eval_path)
        if not evaluation or not evaluation.get("per_language"):
            problems.append("Evaluation file has no per-language results")
    processor = run_dir / "processor"
    if not processor.is_dir() or not all(
        (processor / name).is_file() for name in PROCESSOR_MARKERS
    ):
        problems.append(
            "Processor files are missing. A loadable repo needs the tokenizer and feature extractor."
        )
    if _looks_unverified(config.license):
        problems.append(
            "license is empty or still a placeholder. Set the confirmed license in the publish config."
        )
    if not config.datasets:
        problems.append("datasets metadata is empty")
    for dataset in config.datasets:
        if _looks_unverified(dataset.name) or _looks_unverified(dataset.license):
            problems.append(f"Dataset {dataset.name!r} still has unverified metadata")
    card = run_dir / "model_card.md"
    if not card.is_file():
        problems.append("model card has not been generated")
    return problems


def _find_eval(run_dir: Path) -> Path | None:
    direct = run_dir / "eval" / "results.json"
    if direct.is_file():
        return direct
    matches = sorted(run_dir.glob("eval/**/results.json"))
    return matches[-1] if matches else None


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def _looks_unverified(value: str) -> bool:
    text = value.strip()
    if not text:
        return True
    upper = text.upper()
    return any(marker in upper for marker in PLACEHOLDERS)
