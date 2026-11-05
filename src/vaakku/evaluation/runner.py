"""Score a held-out manifest. Training rows are refused."""

from __future__ import annotations

import json
from pathlib import Path

from vaakku.config.models import EvalConfig
from vaakku.errors import EvaluationError
from vaakku.evaluation.metrics import score_predictions
from vaakku.evaluation.report import write_reports
from vaakku.logging import log_event, setup_logging
from vaakku.runtime import git_sha, new_run_id


def load_test_rows(config: EvalConfig) -> list[dict]:
    path = Path(config.test_manifest)
    if not path.is_file():
        raise EvaluationError(f"Test manifest not found: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        split = row.get("split")
        if split != config.expect_split:
            raise EvaluationError(
                f"Refusing to evaluate split {split!r}. Expected {config.expect_split!r}. "
                "Training data cannot be the test set."
            )
        rows.append(row)
    if config.max_samples is not None:
        rows = rows[: config.max_samples]
    if not rows:
        raise EvaluationError(f"Test manifest is empty: {path}")
    return rows


def evaluate_rows(
    config: EvalConfig, predictions: list[dict], gpu_memory_mb: float | None = None
) -> dict:
    scored = score_predictions(predictions)
    scored.update(
        {
            "split": config.expect_split,
            "git_sha": git_sha(),
            "base_model_id": config.base_model_id,
            "model_path": config.model_path,
            "gpu_memory_mb": gpu_memory_mb,
            "gpu_memory_reason": None
            if gpu_memory_mb is not None
            else "GPU memory is unavailable because this score was not collected on CUDA.",
            "base_comparison": {
                "available": False,
                "reason": "Base-model comparison was not requested."
                if not config.compare_base
                else "Base-model comparison did not produce a score.",
            },
        }
    )
    if not scored["available"]:
        scored["available"] = False
    return scored


def write_evaluation(
    config: EvalConfig, predictions: list[dict], gpu_memory_mb: float | None = None
) -> Path:
    logger = setup_logging("eval", config.logging)
    results = evaluate_rows(config, predictions, gpu_memory_mb=gpu_memory_mb)
    run_id = new_run_id("eval")
    output = Path(config.output_dir) / run_id
    write_reports(output, results)
    log_event(
        logger,
        "evaluation finished",
        output=str(output),
        sample_count=results["sample_count"],
        macro_wer=results["macro_wer"],
        per_language=[row["language"] for row in results["per_language"]],
    )
    return output
