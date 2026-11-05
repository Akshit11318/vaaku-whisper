"""API response models. Confidence is omitted unless a model actually returns it."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    reason: str | None = None


class ModelInfoResponse(BaseModel):
    configured: bool
    model_id: str | None = None
    base_model_id: str | None = None
    revision: str | None = None
    device: str | None = None
    reason: str | None = None


class TranscriptResponse(BaseModel):
    transcript: str
    language: str | None = None
    language_mode: Literal["selected", "detected", "unavailable"]
    model_id: str | None = None
    revision: str | None = None
    processing_time_s: float
    audio_duration_s: float | None = None


class MetricsResponse(BaseModel):
    requests: int = 0
    errors: int = 0
    latency_seconds_sum: float = 0.0


class TranscriptResult(BaseModel):
    transcript: str
    language: str | None = None
    language_mode: Literal["selected", "detected", "unavailable"] = "unavailable"
    model_id: str | None = None
    revision: str | None = None
    processing_time_s: float
    audio_duration_s: float | None = None
    device: str | None = None


class EngineStatus(BaseModel):
    configured: bool = False
    ready: bool = False
    model_id: str | None = None
    base_model_id: str | None = None
    revision: str | None = None
    device: str | None = None
    reason: str | None = Field(default="Model not configured")
