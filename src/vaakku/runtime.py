"""Run identifiers and git metadata. No network calls."""

from __future__ import annotations

import os
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path


def find_root(start: Path | None = None) -> Path:
    """Walk upwards until pyproject.toml is found."""
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def git_sha(root: Path | None = None) -> str:
    override = os.environ.get("VAAKKU_GIT_SHA", "").strip()
    if override:
        return override
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root or find_root(),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return output.strip() or "unknown"


def new_run_id(name: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(char if char.isalnum() or char in "-_" else "-" for char in name).strip("-")
    slug = slug or "run"
    return f"{stamp}-{slug}-{uuid.uuid4().hex[:8]}"


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
