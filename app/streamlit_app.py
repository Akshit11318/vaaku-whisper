"""Vaakku Studio. Empty states stay empty until a real run exists."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from vaakku.constants import GITHUB_REPOSITORY, HF_MODEL_ID, PROJECT_NAME, STUDIO_NAME
from vaakku.studio.loaders import LANGUAGES, api_status, evaluation_view, language_chart

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,650&family=Source+Sans+3:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: "Source Sans 3", sans-serif; color: #13242A; }
.stApp { background: #F4F7F5; }
h1, .studio-title { font-family: Newsreader, serif; letter-spacing: -0.03em; }
.script-rail { display: flex; gap: 0.9rem; flex-wrap: wrap; padding-bottom: 0.4rem; border-bottom: 1px solid #D7E2DE; }
.script-rail span { font-size: 1.05rem; color: #0F6E6B; }
.script-rail span:first-child { box-shadow: inset 0 -2px 0 #E4B15A; }
div[data-testid="stSidebar"] { background: #13242A; }
div[data-testid="stSidebar"] * { color: #F4F7F5; }
.transcript-block { font-family: Newsreader, serif; font-size: 1.7rem; line-height: 1.45; }
"""


def main() -> None:
    st.set_page_config(page_title=STUDIO_NAME, layout="wide")
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    page = st.sidebar.radio("Studio", ["Overview", "Transcribe", "Evaluation", "Model"])
    st.sidebar.markdown(_rail(), unsafe_allow_html=True)
    if page == "Overview":
        _overview()
    elif page == "Transcribe":
        _transcribe()
    elif page == "Evaluation":
        _evaluation()
    else:
        _model()


def _rail() -> str:
    parts = "".join(f"<span>{item['script']}</span>" for item in LANGUAGES)
    return f"<div class='script-rail'>{parts}</div>"


def _overview() -> None:
    st.title(STUDIO_NAME)
    st.write(PROJECT_NAME)
    st.write(
        "Speech-to-text for Tamil, Telugu, Hindi, Kannada, and Malayalam. "
        "Scores appear here only after an evaluation file exists."
    )
    columns = st.columns(5)
    for column, item in zip(columns, LANGUAGES, strict=True):
        column.metric(item["name"], item["code"])
    status = api_status(os.environ.get("VAAKKU_API_URL"))
    if not status["reachable"]:
        st.info(status["reason"])
    elif status["ready"]:
        st.success("API is ready.")
    else:
        st.warning(status.get("reason") or "Model not configured")
    st.subheader("Architecture")
    st.markdown(
        "Audio is prepared on the data disk, fine-tuned with LoRA on Whisper, "
        "and scored on the FLEURS test split. The API and this studio read that checkpoint. "
        "GitHub does not train or publish on push."
    )
    st.write(f"Code: {GITHUB_REPOSITORY}")
    st.write(f"Target model repo: https://huggingface.co/{HF_MODEL_ID}")


def _transcribe() -> None:
    st.title("Transcribe")
    base_url = os.environ.get("VAAKKU_API_URL", "").strip()
    if not base_url:
        st.info("Model not configured. Set VAAKKU_API_URL and start the API with `make api`.")
        return
    language = st.selectbox("Language", ["auto", *[item["code"] for item in LANGUAGES]])
    upload = st.file_uploader("Upload audio", type=["wav", "flac", "ogg", "mp3", "m4a"])
    recorded = None
    if hasattr(st, "audio_input"):
        recorded = st.audio_input("Record")
    else:
        st.caption("This Streamlit build has no browser recorder. Upload a file instead.")
    chosen = recorded or upload
    if chosen is None:
        st.info("Upload or record audio to transcribe it.")
        return
    st.audio(chosen.getvalue())
    if not st.button("Transcribe"):
        return
    with st.spinner("Transcribing"):
        result = _post_audio(base_url, chosen.getvalue(), chosen.name, language)
    if "error" in result:
        st.error(result["error"])
        return
    st.markdown(
        f"<div class='transcript-block'>{result.get('transcript') or ''}</div>",
        unsafe_allow_html=True,
    )
    st.write(
        f"Language: {result.get('language') or 'unavailable'} ({result.get('language_mode')}). "
        f"Duration: {result.get('audio_duration_s')} s. "
        f"Latency: {result.get('processing_time_s')} s."
    )
    transcript = result.get("transcript") or ""
    st.download_button("Download transcript", data=transcript, file_name="transcript.txt")


def _evaluation() -> None:
    st.title("Evaluation")
    path = Path(os.environ.get("VAAKKU_EVAL_PATH", "artifacts/eval/latest/results.json"))
    results = evaluation_view(path)
    chart = language_chart(results)
    if not chart["available"]:
        st.info(chart["reason"])
        return
    left, right = st.columns(2)
    left.metric("Macro WER", _number(results.get("macro_wer")))
    right.metric("Macro CER", _number(results.get("macro_cer")))
    st.bar_chart({"WER": chart["wer"], "CER": chart["cer"]}, x=None)
    st.dataframe(results["per_language"], hide_index=True)
    comparison = results.get("base_comparison") or {}
    if comparison.get("available"):
        st.write(f"Base macro WER: {_number(comparison.get('macro_wer'))}")
    else:
        st.info(comparison.get("reason") or "Base-model comparison is unavailable.")


def _model() -> None:
    st.title("Model")
    st.write("Base model: openai/whisper-small")
    st.write("Method: LoRA. The base weights stay frozen.")
    run = Path(os.environ.get("VAAKKU_RUN_DIR", "artifacts/runs/latest"))
    meta_path = run / "run.json" if run.is_dir() else Path("missing")
    meta = {}
    if meta_path.is_file():
        import json

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    st.write(f"Run id: {meta.get('run_id') or 'Not recorded.'}")
    st.write(f"Git commit: {meta.get('git_sha') or 'Not recorded.'}")
    st.write(f"LoRA: {meta.get('lora') or 'Not recorded.'}")
    record = Path("artifacts/publish/publish_record.json")
    if record.is_file():
        import json

        published = json.loads(record.read_text(encoding="utf-8"))
        st.write(f"Published: {published.get('url')} @ {published.get('revision')}")
    else:
        st.info("This model has not been published from this machine.")
    st.write(f"Target repository: https://huggingface.co/{HF_MODEL_ID}")
    st.caption(
        "Limitations: five Indic languages, LoRA only, and no score is shown until evaluation writes a file."
    )


def _post_audio(base_url: str, payload: bytes, name: str, language: str) -> dict:
    import json
    from urllib.error import HTTPError, URLError
    from urllib.request import Request, urlopen

    boundary = "vaakkuBoundary"
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="language"\r\n\r\n{language}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{Path(name).name}"\r\n'
            "Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + payload
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = Request(
        f"{base_url.rstrip('/')}/transcribe",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8")
        return {"error": detail or f"API returned {exc.code}"}
    except URLError as exc:
        return {"error": f"API is unreachable: {exc.reason}"}


def _number(value) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
