"""Load one Whisper model and transcribe uploads. Requests are serialized."""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path

from vaakku.api.schemas import EngineStatus, TranscriptResult
from vaakku.constants import DEFAULT_BASE_MODEL_ID, SUPPORTED_LANGUAGES, WHISPER_LANGUAGE_NAMES
from vaakku.data.audio import read_audio
from vaakku.errors import InferenceError, ModelNotReady

MAX_AUDIO_SECONDS = 30.0


class InferenceEngine:
    """One loaded model. `transcribe` holds a lock because one GPU is shared."""

    def __init__(self, status: EngineStatus, model=None, processor=None) -> None:
        self._status = status
        self.model = model
        self.processor = processor
        self._lock = threading.Lock()

    def status(self) -> EngineStatus:
        return self._status

    def transcribe(self, payload: bytes, suffix: str, language: str | None) -> TranscriptResult:
        if not self._status.ready or self.model is None or self.processor is None:
            raise ModelNotReady(self._status.reason or "Model not configured")
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="vaakku-") as directory:
            audio_path = Path(directory) / f"upload{suffix}"
            audio_path.write_bytes(payload)
            try:
                audio, duration = read_audio(audio_path)
            except Exception as exc:
                raise InferenceError("The upload is not readable audio.") from exc
        if duration > MAX_AUDIO_SECONDS:
            raise InferenceError(f"Audio is longer than {MAX_AUDIO_SECONDS:.0f} seconds.")
        with self._lock:
            transcript, detected = _generate(
                self.model,
                self.processor,
                audio,
                language,
                self._status.device or "cpu",
            )
        mode = "selected" if language else ("detected" if detected else "unavailable")
        return TranscriptResult(
            transcript=transcript,
            language=language or detected,
            language_mode=mode,
            model_id=self._status.model_id,
            revision=self._status.revision,
            processing_time_s=round(time.perf_counter() - started, 4),
            audio_duration_s=round(duration, 4),
            device=self._status.device,
        )


def load_engine_from_env() -> InferenceEngine:
    model_path = os.environ.get("VAAKKU_MODEL_PATH", "").strip()
    model_id = os.environ.get("VAAKKU_MODEL_ID", "").strip()
    base_model_id = os.environ.get("VAAKKU_BASE_MODEL_ID", DEFAULT_BASE_MODEL_ID).strip()
    device_name = os.environ.get("VAAKKU_DEVICE", "auto").strip() or "auto"
    if not model_path and not model_id:
        return InferenceEngine(
            EngineStatus(configured=False, ready=False, reason="Model not configured")
        )
    try:
        import torch
        from peft import PeftConfig, PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as exc:
        return InferenceEngine(
            EngineStatus(
                configured=True, ready=False, reason=f"Model dependencies are not installed: {exc}"
            )
        )
    device = _select_device(device_name, torch.cuda.is_available())
    source = model_path or model_id
    peft_config = _peft_config(PeftConfig, source)
    if peft_config is not None:
        resolved_base = peft_config.base_model_name_or_path or base_model_id
        base = WhisperForConditionalGeneration.from_pretrained(resolved_base)
        model = PeftModel.from_pretrained(base, source)
    else:
        resolved_base = source
        model = WhisperForConditionalGeneration.from_pretrained(source)
    processor = _load_processor(WhisperProcessor, source, resolved_base)
    model.to(device)
    model.eval()
    revision = _revision(source)
    return InferenceEngine(
        EngineStatus(
            configured=True,
            ready=True,
            model_id=source,
            base_model_id=resolved_base,
            revision=revision,
            device=device.type,
            reason=None,
        ),
        model=model,
        processor=processor,
    )


def _generate(model, processor, audio, language: str | None, device: str) -> tuple[str, str | None]:
    import torch

    inputs = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    features = inputs.input_features.to(device)
    generate_kwargs = {"task": "transcribe"}
    detected = None
    if language:
        generate_kwargs["language"] = WHISPER_LANGUAGE_NAMES[language]
    else:
        detect = getattr(model, "detect_language", None)
        if detect is not None:
            try:
                detected_ids = detect(features)
                detected = _language_from_detection(processor, detected_ids)
            except Exception:
                detected = None
            if detected:
                generate_kwargs["language"] = WHISPER_LANGUAGE_NAMES.get(detected, detected)
    # This PEFT build rejects a positional feature tensor. The LoRA layers
    # already live on the base module, so generation uses that module.
    generator = model.get_base_model() if hasattr(model, "get_base_model") else model
    with torch.no_grad():
        predicted = generator.generate(features, **generate_kwargs)
    text = processor.tokenizer.batch_decode(predicted, skip_special_tokens=True)[0].strip()
    return text, detected if detected in SUPPORTED_LANGUAGES else None


def _peft_config(peft_config_cls, source: str):
    try:
        return peft_config_cls.from_pretrained(source)
    except Exception:
        return None


def _load_processor(processor_cls, source: str, base_model_id: str):
    candidates = [source]
    path = Path(source)
    if path.exists():
        candidates.append(str(path.parent.parent / "processor"))
        candidates.append(str(path.parent / "processor"))
    candidates.append(base_model_id)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return processor_cls.from_pretrained(candidate)
        except Exception as exc:
            last_error = exc
    raise InferenceError(f"Could not load the Whisper processor for {source}") from last_error


def _language_from_detection(processor, detected_ids) -> str | None:
    tokenizer = processor.tokenizer
    token_id = int(detected_ids[0][0] if getattr(detected_ids, "ndim", 1) > 1 else detected_ids[0])
    token = tokenizer.convert_ids_to_tokens(token_id)
    if not isinstance(token, str):
        return None
    name = token.strip("<|>")
    for code, whisper_name in WHISPER_LANGUAGE_NAMES.items():
        if name == whisper_name or name == code:
            return code
    return None


def _select_device(requested: str, cuda: bool):
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not cuda:
            raise InferenceError("VAAKKU_DEVICE=cuda but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if cuda else "cpu")


def _revision(source: str) -> str | None:
    if Path(source).exists():
        return None
    try:
        from huggingface_hub import model_info

        info = model_info(source)
    except Exception:
        return None
    return getattr(info, "sha", None)
