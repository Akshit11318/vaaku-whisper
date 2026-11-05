import json
from pathlib import Path

from fastapi.testclient import TestClient

from vaakku.api.app import create_app
from vaakku.api.schemas import EngineStatus, TranscriptResult
from vaakku.config import load_publish_config
from vaakku.publishing.card import render_model_card
from vaakku.publishing.preflight import preflight
from vaakku.studio.loaders import evaluation_view, language_chart


class ReadyEngine:
    def status(self) -> EngineStatus:
        return EngineStatus(
            configured=True, ready=True, model_id="local", device="cpu", reason=None
        )

    def transcribe(self, payload: bytes, suffix: str, language: str | None) -> TranscriptResult:
        return TranscriptResult(
            transcript="வணக்கம்",
            language=language or "ta",
            language_mode="selected" if language else "unavailable",
            model_id="local",
            revision=None,
            processing_time_s=0.2,
            audio_duration_s=1.0,
            device="cpu",
        )


class IdleEngine:
    def status(self) -> EngineStatus:
        return EngineStatus(configured=False, ready=False, reason="Model not configured")

    def transcribe(self, payload: bytes, suffix: str, language: str | None) -> TranscriptResult:
        raise AssertionError("idle engine should not transcribe")


def test_health_ready_and_transcribe() -> None:
    client = TestClient(create_app(ReadyEngine()))
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").status_code == 200
    response = client.post(
        "/transcribe",
        files={"file": ("clip.wav", b"RIFFdemo", "audio/wav")},
        data={"language": "ta"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["transcript"] == "வணக்கம்"
    assert body["language"] == "ta"
    assert "confidence" not in body


def test_unconfigured_model_is_not_ready() -> None:
    client = TestClient(create_app(IdleEngine()))
    ready = client.get("/ready")
    assert ready.status_code == 503
    assert ready.json()["reason"] == "Model not configured"


def test_model_card_does_not_invent_metrics() -> None:
    card = render_model_card(
        {
            "languages": ["ta", "hi"],
            "license": "apache-2.0",
            "base_model_id": "openai/whisper-small",
            "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
            "datasets": [
                {
                    "name": "google/fleurs",
                    "license": "CC-BY-4.0",
                    "url": "https://example.test",
                    "split_method": "official",
                }
            ],
            "evaluation": {},
            "repo_id": "rudra11/vaakku-whisper-small",
        }
    )
    assert "Not measured." in card
    assert "google/fleurs" in card


def test_publish_preflight_blocks_placeholders(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "best" / "adapter").mkdir(parents=True)
    (run / "best" / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (run / "best" / "adapter" / "adapter_model.safetensors").write_text("x", encoding="utf-8")
    (run / "best" / "training_state.json").write_text('{"step": 1}', encoding="utf-8")
    (run / "processor").mkdir()
    (run / "processor" / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (run / "processor" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (run / "run.json").write_text(json.dumps({"purpose": "experiment"}), encoding="utf-8")
    (run / "model_card.md").write_text("# card", encoding="utf-8")
    eval_file = tmp_path / "results.json"
    eval_file.write_text(json.dumps({"per_language": []}), encoding="utf-8")
    config_path = tmp_path / "publish.yaml"
    config_path.write_text(
        f"""
repo_id: rudra11/vaakku-whisper-small
token_env: HF_TOKEN
run_dir: {run}
license: REPLACE_WITH_CONFIRMED_LICENSE
eval_results: {eval_file}
datasets:
  - name: google/fleurs
    license: CC-BY-4.0
    url: https://huggingface.co/datasets/google/fleurs
    split_method: official
""",
        encoding="utf-8",
    )
    problems = preflight(load_publish_config(config_path))
    assert any("license" in problem for problem in problems)
    assert any("per-language" in problem for problem in problems)


def test_evaluation_dashboard_has_an_empty_state(tmp_path: Path) -> None:
    view = evaluation_view(tmp_path / "missing.json")
    assert view["available"] is False
    assert language_chart(view)["reason"] == "Evaluation metrics unavailable."


def test_repo_publish_config_is_not_ready() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_publish_config(root / "configs" / "publish.yaml")
    assert config.repo_id == "rudra11/vaakku-whisper-small"
    assert config.mode == "peft_adapter"
    problems = preflight(config)
    assert problems
