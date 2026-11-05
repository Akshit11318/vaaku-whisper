"""Normalized speech records and rejection rows."""

from __future__ import annotations

from pydantic import BaseModel, Field


class NormalizedSample(BaseModel):
    audio_path: str
    transcription: str
    language: str
    speaker_id: str | None = None
    dataset_name: str
    sample_id: str
    duration_s: float
    split: str | None = None


class Rejection(BaseModel):
    sample_id: str | None = None
    dataset_name: str
    reason: str
    detail: str = ""
    language: str | None = None


class QualityReport(BaseModel):
    accepted: int
    rejected: int
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    per_language: dict[str, int] = Field(default_factory=dict)
    duration_s_total: float = 0.0
    split_method: str
    split_counts: dict[str, int] = Field(default_factory=dict)
    official_speaker_overlap: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    rejections: list[Rejection] = Field(default_factory=list)
