"""Values the Studio UI can render before a model exists."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from vaakku.constants import HF_MODEL_ID, LANGUAGE_NAMES, SUPPORTED_LANGUAGES

LANGUAGES = [
    {"code": code, "name": LANGUAGE_NAMES[code], "script": script}
    for code, script in (
        ("ta", "தமிழ்"),
        ("te", "తెలుగు"),
        ("hi", "हिन्दी"),
        ("kn", "ಕನ್ನಡ"),
        ("ml", "മലയാളം"),
    )
]


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else None


def evaluation_view(path: Path) -> dict:
    payload = load_json(path)
    if not payload or not payload.get("per_language"):
        return {"available": False, "reason": "Evaluation metrics unavailable."}
    payload["available"] = True
    return payload


def language_chart(results: dict) -> dict:
    if not results.get("available"):
        return {
            "available": False,
            "reason": results.get("reason", "Evaluation metrics unavailable."),
        }
    rows = [row for row in results.get("per_language", []) if row.get("wer") is not None]
    if not rows:
        return {"available": False, "reason": "Evaluation metrics unavailable."}
    return {
        "available": True,
        "languages": [row["language"] for row in rows],
        "wer": [row["wer"] for row in rows],
        "cer": [row.get("cer") for row in rows],
        "samples": [row.get("samples") for row in rows],
    }


def api_status(base_url: str | None, timeout: float = 2.0) -> dict:
    if not base_url:
        return {"reachable": False, "reason": "Model not configured"}
    try:
        with urlopen(f"{base_url.rstrip('/')}/health", timeout=timeout) as response:
            health = json.loads(response.read().decode("utf-8"))
        try:
            with urlopen(f"{base_url.rstrip('/')}/ready", timeout=timeout) as response:
                ready = json.loads(response.read().decode("utf-8"))
                ready_ok = response.status == 200 and bool(ready.get("ready"))
        except HTTPError as exc:
            ready = json.loads(exc.read().decode("utf-8"))
            ready_ok = False
    except URLError as exc:
        return {"reachable": False, "reason": f"API is unreachable: {exc.reason}"}
    return {
        "reachable": True,
        "health": health,
        "ready": ready_ok,
        "reason": ready.get("reason"),
    }


def target_model_id() -> str:
    return HF_MODEL_ID


def supported_language_codes() -> list[str]:
    return list(SUPPORTED_LANGUAGES)
