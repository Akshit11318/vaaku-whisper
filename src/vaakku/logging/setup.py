"""JSON logs with a redaction list. Audio and transcripts stay out of the log line."""

from __future__ import annotations

import json
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from vaakku.config.models import LoggingConfig
from vaakku.runtime import git_sha, utc_now_iso

REDACTED_FIELDS = {
    "audio",
    "waveform",
    "transcript",
    "transcription",
    "text",
    "sentence",
    "token",
    "hf_token",
    "authorization",
    "password",
    "secret",
    "access_token",
}

_CONFIGURED: set[str] = set()


class JsonFormatter(logging.Formatter):
    def __init__(self, component: str, run_id: str | None) -> None:
        super().__init__()
        self.component = component
        self.run_id = run_id
        self.git_sha = git_sha()

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "level": record.levelname,
            "component": getattr(record, "component", self.component),
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": getattr(record, "run_id", self.run_id),
            "git_sha": self.git_sha,
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            payload.update(_redact(fields))
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def _redact(fields: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in fields.items():
        if key.lower() in REDACTED_FIELDS:
            cleaned[key] = "[redacted]"
        else:
            cleaned[key] = value
    return cleaned


def setup_logging(
    component: str,
    config: LoggingConfig | None = None,
    run_id: str | None = None,
) -> logging.Logger:
    """Configure one component logger. Safe to call again for the same component."""
    config = config or LoggingConfig()
    logger = logging.getLogger(f"vaakku.{component}")
    logger.setLevel(config.level)
    logger.propagate = False
    if component in _CONFIGURED:
        return logger

    formatter = JsonFormatter(component=component, run_id=run_id)
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    log_dir = Path(config.dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / f"{component}.log",
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    _CONFIGURED.add(component)
    return logger


def log_event(logger: logging.Logger, message: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, message, extra={"fields": fields})


def reset_logging_state() -> None:
    """Test helper. Closes handlers so later tests can reconfigure log paths."""
    for name in list(_CONFIGURED):
        logger = logging.getLogger(f"vaakku.{name}")
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)
    _CONFIGURED.clear()
