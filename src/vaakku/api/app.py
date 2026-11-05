"""FastAPI app. The model is loaded once and inference is serialized."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from vaakku.api.metrics import RequestMetrics
from vaakku.api.schemas import (
    EngineStatus,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    ReadyResponse,
    TranscriptResponse,
)
from vaakku.constants import SUPPORTED_LANGUAGES
from vaakku.errors import InferenceError, ModelNotReady
from vaakku.logging import log_event, setup_logging

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def create_app(engine=None) -> FastAPI:
    if engine is None:
        from vaakku.inference.engine import load_engine_from_env

        engine = load_engine_from_env()
    logger = setup_logging("api")
    metrics = RequestMetrics()
    app = FastAPI(title="Vaakku — Open Indic Speech", version="0.1.0")

    @app.middleware("http")
    async def request_id(request, call_next):
        started = time.perf_counter()
        request_id_value = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        try:
            response = await call_next(request)
        except Exception:
            log_event(
                logger,
                "request failed",
                request_id=request_id_value,
                path=request.url.path,
                method=request.method,
            )
            raise
        response.headers["x-request-id"] = request_id_value
        log_event(
            logger,
            "request",
            request_id=request_id_value,
            path=request.url.path,
            method=request.method,
            status=response.status_code,
            latency_s=round(time.perf_counter() - started, 4),
        )
        return response

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/ready", response_model=ReadyResponse)
    def ready():
        status = _status(engine)
        body = ReadyResponse(ready=status.ready, reason=None if status.ready else status.reason)
        if not status.ready:
            return JSONResponse(status_code=503, content=body.model_dump())
        return body

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        status = _status(engine)
        return ModelInfoResponse(
            configured=status.configured,
            model_id=status.model_id,
            base_model_id=status.base_model_id,
            revision=status.revision,
            device=status.device,
            reason=status.reason,
        )

    @app.get("/metrics", response_model=MetricsResponse)
    def metrics_endpoint() -> MetricsResponse:
        return MetricsResponse(**metrics.as_dict())

    @app.post("/transcribe", response_model=TranscriptResponse)
    async def transcribe(
        file: UploadFile = File(...),
        language: str | None = Form(default=None),
    ) -> TranscriptResponse:
        started = time.perf_counter()
        selected = _clean_language(language)
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            raise HTTPException(
                status_code=400, detail="Upload a WAV, FLAC, OGG, MP3, or M4A file."
            )
        payload = await file.read()
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Audio upload exceeds 25 MB.")
        try:
            result = engine.transcribe(payload, suffix=suffix, language=selected)
        except ModelNotReady as exc:
            metrics.observe(time.perf_counter() - started, failed=True)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except InferenceError as exc:
            metrics.observe(time.perf_counter() - started, failed=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        metrics.observe(time.perf_counter() - started, failed=False)
        log_event(
            logger,
            "transcribed",
            language=result.language,
            language_mode=result.language_mode,
            audio_duration_s=result.audio_duration_s,
            processing_time_s=result.processing_time_s,
        )
        return TranscriptResponse(
            transcript=result.transcript,
            language=result.language,
            language_mode=result.language_mode,
            model_id=result.model_id,
            revision=result.revision,
            processing_time_s=result.processing_time_s,
            audio_duration_s=result.audio_duration_s,
        )

    return app


def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _status(engine) -> EngineStatus:
    if hasattr(engine, "status"):
        return engine.status()
    return EngineStatus(ready=False, configured=False, reason="Model not configured")


def _clean_language(language: str | None) -> str | None:
    if language is None or language.strip() == "" or language.strip().lower() == "auto":
        return None
    code = language.strip().lower()
    if code not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Language must be one of {', '.join(SUPPORTED_LANGUAGES)}, or auto.",
        )
    return code
