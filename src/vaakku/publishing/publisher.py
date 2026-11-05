"""Upload a checked adapter. Success is recorded only after the hub accepts it."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from vaakku.config.models import PublishConfig
from vaakku.errors import PublishError
from vaakku.logging import log_event, setup_logging
from vaakku.publishing.card import render_model_card
from vaakku.publishing.preflight import _find_eval, _read_json, preflight


def generate_card(config: PublishConfig) -> Path:
    run_dir = Path(config.run_dir)
    run_meta = _read_json(run_dir / "run.json") or {}
    evaluation = _read_json(_evaluation_file(config) or Path("missing-eval.json")) or {}
    split_meta = _read_json(run_dir / "split_manifest.json") or {}
    context = {
        "languages": run_meta.get("languages") or [item for item in ("ta", "te", "hi", "kn", "ml")],
        "license": config.license,
        "base_model_id": run_meta.get("base_model_id"),
        "lora": run_meta.get("lora"),
        "datasets": [item.model_dump() for item in config.datasets],
        "evaluation": evaluation,
        "run_id": run_meta.get("run_id"),
        "git_sha": run_meta.get("git_sha"),
        "precision": run_meta.get("precision"),
        "split_method": split_meta.get("method"),
        "repo_id": config.repo_id,
    }
    card = render_model_card(context)
    destination = run_dir / "model_card.md"
    destination.write_text(card, encoding="utf-8")
    return destination


def publish_model(config: PublishConfig) -> dict:
    logger = setup_logging("publish", config.logging)
    generate_card(config)
    problems = preflight(config)
    if problems:
        for problem in problems:
            log_event(logger, "publish preflight failed", detail=problem)
        raise PublishError("Publish preflight failed:\n- " + "\n- ".join(problems))
    token = os.environ.get(config.token_env, "").strip()
    if not token:
        raise PublishError(
            f"Environment variable {config.token_env} is empty. The token was not logged."
        )
    staging = _stage(config)
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        api.create_repo(config.repo_id, repo_type="model", private=config.private, exist_ok=True)
        commit = api.upload_folder(
            folder_path=str(staging),
            repo_id=config.repo_id,
            repo_type="model",
            commit_message="Publish vaakku-whisper-small adapter",
        )
    except Exception as exc:
        log_event(logger, "publish upload failed", error_type=type(exc).__name__)
        raise PublishError(f"Upload failed: {exc}") from exc
    revision = getattr(commit, "oid", None) or getattr(commit, "commit_hash", None)
    url = f"https://huggingface.co/{config.repo_id}"
    record = {"repo_id": config.repo_id, "url": url, "revision": revision, "mode": config.mode}
    (Path(config.run_dir) / "publish_record.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )
    log_event(logger, "publish succeeded", repo_id=config.repo_id, revision=revision)
    return record


def _evaluation_file(config: PublishConfig) -> Path | None:
    if config.eval_results:
        path = Path(config.eval_results)
        return path if path.is_file() else None
    return _find_eval(Path(config.run_dir))


def _stage(config: PublishConfig) -> Path:
    run_dir = Path(config.run_dir)
    staging = run_dir / "publish_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    adapter = run_dir / config.checkpoint / "adapter"
    for path in adapter.iterdir():
        if path.is_file():
            shutil.copy2(path, staging / path.name)
    processor = run_dir / "processor"
    for path in processor.iterdir():
        if path.is_file():
            shutil.copy2(path, staging / path.name)
    shutil.copy2(run_dir / "model_card.md", staging / "README.md")
    eval_path = _evaluation_file(config)
    if eval_path:
        shutil.copy2(eval_path, staging / "eval_results.json")
    run_json = _read_json(run_dir / "run.json") or {}
    safe_run = {
        key: run_json.get(key)
        for key in ("run_id", "git_sha", "base_model_id", "precision", "lora", "purpose")
    }
    (staging / "run.json").write_text(json.dumps(safe_run, indent=2) + "\n", encoding="utf-8")
    return staging
