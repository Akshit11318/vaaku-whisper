"""Training loop for Whisper LoRA. YAML supplies the hyperparameters."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np

from vaakku.config.models import TrainConfig
from vaakku.errors import TrainingError
from vaakku.logging import log_event, setup_logging
from vaakku.runtime import git_sha, new_run_id, utc_now_iso
from vaakku.training.batching import DurationBucketBatchSampler, pad_label_ids, pad_mel_features
from vaakku.training.model import build_model, trainable_parameter_count
from vaakku.training.precision import select_precision


def run_training(
    config: TrainConfig,
    resume_from: str | None = None,
    train_rows: list[dict] | None = None,
    eval_rows: list[dict] | None = None,
) -> Path:
    import torch

    set_seed(config.seed)
    run_id = new_run_id(config.run_name)
    output_dir = Path(config.output_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging("train", config.logging, run_id=run_id)
    if config.purpose == "pipeline_validation":
        log_event(
            logger,
            "pipeline validation run",
            detail="This run checks the pipeline. It is not a quality benchmark.",
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = select_precision(
        config.precision,
        cuda=device.type == "cuda",
        bf16_supported=bool(device.type == "cuda" and torch.cuda.is_bf16_supported()),
    )
    model = build_model(config)
    model.to(device)
    trainable = trainable_parameter_count(model)
    log_event(
        logger,
        "model ready",
        base_model_id=config.base_model_id,
        model_init=config.model_init,
        precision=precision,
        trainable_parameters=trainable,
        device=device.type,
    )
    train_rows = train_rows if train_rows is not None else _load_rows(config, "train")
    eval_rows = eval_rows if eval_rows is not None else _load_rows(config, "validation")
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda") if precision == "fp16" else None
    start_step = 0
    if resume_from:
        start_step = _resume(model, optimizer, scaler, Path(resume_from))
        log_event(logger, "resumed checkpoint", step=start_step, checkpoint=resume_from)
    metadata = {
        "run_id": run_id,
        "purpose": config.purpose,
        "git_sha": git_sha(),
        "base_model_id": config.base_model_id,
        "model_init": config.model_init,
        "precision": precision,
        "trainable_parameters": trainable,
        "created_at": utc_now_iso(),
        "lora": config.lora.model_dump(),
        "selection_metric": config.selection_metric,
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output_dir / "train_config.json").write_text(
        json.dumps(config.model_dump(), indent=2) + "\n",
        encoding="utf-8",
    )
    best_loss = math_inf()
    patience_left = config.early_stopping_patience
    step = start_step
    started = time.perf_counter()
    examples = 0
    stop = False
    for _epoch in range(config.num_epochs):
        sampler = DurationBucketBatchSampler(
            [float(row["duration_s"]) for row in train_rows],
            config.per_device_train_batch_size,
            seed=config.seed + step,
        )
        optimizer.zero_grad(set_to_none=True)
        accum_index = 0
        for batch_ids in sampler:
            batch = _collate([train_rows[index] for index in batch_ids])
            loss = _forward(model, batch, device, precision)
            raw_loss = float(loss.detach().item())
            scaled = loss / config.gradient_accumulation_steps
            if scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()
            accum_index += 1
            examples += len(batch_ids)
            if accum_index % config.gradient_accumulation_steps != 0:
                continue
            _optimizer_step(model, optimizer, scaler, config.max_grad_norm)
            step += 1
            elapsed = max(time.perf_counter() - started, 1e-6)
            log_event(
                logger,
                "train step",
                step=step,
                loss=round(raw_loss, 6),
                examples_per_second=round(examples / elapsed, 3),
                gpu_memory_mb=_gpu_memory_mb(),
            )
            _append_metric(output_dir, {"step": step, "train_loss": raw_loss})
            if step % config.eval_steps == 0 or (config.max_steps and step >= config.max_steps):
                eval_loss = _evaluate_loss(model, eval_rows, device, precision, config)
                improved = eval_loss + config.early_stopping_min_delta < best_loss
                log_event(
                    logger,
                    "validation",
                    step=step,
                    eval_loss=round(eval_loss, 6),
                    improved=improved,
                )
                _append_metric(output_dir, {"step": step, "eval_loss": eval_loss})
                if improved:
                    best_loss = eval_loss
                    patience_left = config.early_stopping_patience
                    _save_checkpoint(model, optimizer, scaler, output_dir / "best", step, best_loss)
                    log_event(
                        logger, "saved best checkpoint", step=step, eval_loss=round(best_loss, 6)
                    )
                elif patience_left is not None:
                    patience_left -= 1
                    if patience_left <= 0:
                        log_event(logger, "early stopping", step=step)
                        stop = True
            if step % config.save_steps == 0:
                _save_checkpoint(
                    model, optimizer, scaler, output_dir / f"checkpoint-{step}", step, best_loss
                )
            if stop or (config.max_steps is not None and step >= config.max_steps):
                break
        if stop or (config.max_steps is not None and step >= config.max_steps):
            break
    if not (output_dir / "best").exists():
        _save_checkpoint(model, optimizer, scaler, output_dir / "best", step, best_loss)
    (output_dir / "LATEST").write_text(run_id + "\n", encoding="utf-8")
    log_event(logger, "training finished", run_id=run_id, step=step, output=str(output_dir))
    return output_dir


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_rows(config: TrainConfig, split: str) -> list[dict]:
    if config.synthetic or config.model_init == "tiny_random":
        count = (
            config.synthetic_train_samples if split == "train" else config.synthetic_eval_samples
        )
        return [_synthetic_row(index) for index in range(count)]
    path = config.train_manifest if split == "train" else config.validation_manifest
    if not path:
        raise TrainingError(f"Missing {split} manifest")
    rows = _read_jsonl(Path(path))
    limit = config.max_train_samples if split == "train" else config.max_eval_samples
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise TrainingError(f"{split} manifest is empty: {path}")
    return [_manifest_row(row) for row in rows]


def _synthetic_row(index: int) -> dict:
    import torch

    generator = torch.Generator().manual_seed(index)
    # Tiny Whisper downsamples the mel sequence by 2, so 256 frames match max_source_positions=128.
    features = torch.randn(80, 256, generator=generator).numpy()
    labels = torch.randint(3, 128, (8,), generator=generator).tolist()
    return {"input_features": features, "labels": labels, "duration_s": 1.0}


def _manifest_row(row: dict) -> dict:
    raise TrainingError(
        "Manifest training needs the Whisper feature extractor. "
        "Use scripts/train.py, which builds features before the loop. "
        f"Received sample {row.get('sample_id')}"
    )


def _collate(rows: list[dict]) -> dict:
    return {
        "input_features": pad_mel_features([row["input_features"] for row in rows]),
        "labels": pad_label_ids([row["labels"] for row in rows]),
    }


def _forward(model, batch: dict, device, precision: str):
    import torch

    features = torch.as_tensor(batch["input_features"], device=device)
    labels = torch.as_tensor(batch["labels"], device=device)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)
    enabled = dtype is not None
    # LoRA layers are already inside the base module. Calling the PEFT wrapper
    # forwards a default input_ids that current Whisper rejects.
    module = model.get_base_model() if hasattr(model, "get_base_model") else model
    with torch.autocast(device_type=device.type, dtype=dtype or torch.float32, enabled=enabled):
        outputs = module(input_features=features, labels=labels)
    if outputs.loss is None:
        raise TrainingError("The model did not return a loss")
    return outputs.loss


def _optimizer_step(model, optimizer, scaler, max_grad_norm: float) -> None:
    import torch

    params = [param for param in model.parameters() if param.requires_grad]
    if scaler is not None:
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _evaluate_loss(model, rows: list[dict], device, precision: str, config: TrainConfig) -> float:
    import torch

    was_training = model.training
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        sampler = DurationBucketBatchSampler(
            [float(row["duration_s"]) for row in rows],
            config.per_device_eval_batch_size,
            seed=config.seed,
            shuffle=False,
        )
        for batch_ids in sampler:
            batch = _collate([rows[index] for index in batch_ids])
            losses.append(float(_forward(model, batch, device, precision).item()))
    if was_training:
        model.train()
    if not losses:
        raise TrainingError("Validation produced no batches")
    return sum(losses) / len(losses)


def _save_checkpoint(model, optimizer, scaler, path: Path, step: int, best_loss: float) -> None:
    import torch

    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path / "adapter")
    state = {"step": step, "best_loss": best_loss, "optimizer": optimizer.state_dict()}
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path / "training_state.pt")
    (path / "training_state.json").write_text(
        json.dumps({"step": step, "best_loss": None if best_loss == math_inf() else best_loss})
        + "\n",
        encoding="utf-8",
    )


def _resume(model, optimizer, scaler, checkpoint: Path) -> int:
    import torch
    from peft import set_peft_model_state_dict

    adapter = checkpoint / "adapter"
    weights = adapter / "adapter_model.safetensors"
    if weights.is_file():
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(weights))
    elif (adapter / "adapter_model.bin").is_file():
        set_peft_model_state_dict(
            model, torch.load(adapter / "adapter_model.bin", map_location="cpu")
        )
    else:
        raise TrainingError(f"No adapter weights in {checkpoint}")
    state_path = checkpoint / "training_state.pt"
    if not state_path.is_file():
        raise TrainingError(f"No training state in {checkpoint}")
    state = torch.load(state_path, map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and "scaler" in state:
        scaler.load_state_dict(state["scaler"])
    return int(state["step"])


def _append_metric(output_dir: Path, payload: dict) -> None:
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise TrainingError(f"Manifest not found: {path}")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _gpu_memory_mb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)


def math_inf() -> float:
    return float("inf")
