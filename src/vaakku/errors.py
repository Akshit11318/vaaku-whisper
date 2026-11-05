"""Project errors. Callers turn these into logs and non-zero exits."""

from __future__ import annotations


class VaakkuError(Exception):
    """Base error for expected failures."""


class ConfigError(VaakkuError):
    """A YAML config or environment value is invalid."""


class DataError(VaakkuError):
    """The data pipeline cannot continue."""


class TrainingError(VaakkuError):
    """Training cannot start or resume."""


class EvaluationError(VaakkuError):
    """Evaluation refused to score the given split."""


class PublishError(VaakkuError):
    """Publishing preflight failed, or the upload did not succeed."""


class InferenceError(VaakkuError):
    """A single transcription request failed."""


class ModelNotReady(InferenceError):
    """The process is up, but no model is loaded."""
