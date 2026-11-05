"""Model card text filled only from config and measured files."""

from __future__ import annotations


def render_model_card(context: dict) -> str:
    languages = ", ".join(context.get("languages") or []) or "Not recorded."
    lora = context.get("lora") or {}
    datasets = context.get("datasets") or []
    evaluation = context.get("evaluation") or {}
    per_language = evaluation.get("per_language") or []
    lines = [
        "---",
        "language:",
        *[f"- {code}" for code in (context.get("languages") or [])],
        f"license: {context.get('license', 'other')}",
        "tags:",
        "- whisper",
        "- automatic-speech-recognition",
        "- peft",
        "- lora",
        f"base_model: {context.get('base_model_id', 'openai/whisper-small')}",
        "---",
        "",
        "# vaakku-whisper-small",
        "",
        "Vaakku — Open Indic Speech. This repository is a LoRA adapter for multilingual",
        "speech recognition in Tamil, Telugu, Hindi, Kannada, and Malayalam.",
        "",
        "## Supported languages",
        "",
        languages,
        "",
        "## Fine-tuning",
        "",
        f"- Base model: {context.get('base_model_id', 'Not recorded.')}",
        "- Method: LoRA through Hugging Face PEFT. The base model is not fully fine-tuned.",
        f"- LoRA rank: {_show(lora.get('r'))}",
        f"- LoRA alpha: {_show(lora.get('alpha'))}",
        f"- LoRA dropout: {_show(lora.get('dropout'))}",
        f"- Target modules: {_show(lora.get('target_modules'))}",
        f"- Training run: {_show(context.get('run_id'))}",
        f"- Git commit: {_show(context.get('git_sha'))}",
        f"- Precision: {_show(context.get('precision'))}",
        "",
        "## Datasets",
        "",
    ]
    if not datasets:
        lines.append("Dataset list was not provided.")
    for dataset in datasets:
        lines.append(
            f"- {dataset.get('name')}: license {dataset.get('license')}. "
            f"Split method: {_show(dataset.get('split_method'))}. "
            f"Link: {_show(dataset.get('url'))}."
        )
    lines.extend(
        [
            "",
            "Raw audio is not part of this repository.",
            "",
            f"Split methodology: {_show(context.get('split_method'))}",
            "",
            "## Evaluation",
            "",
            "Scores below are copied from the evaluation file. Missing values stay missing.",
            "",
            f"- Micro WER: {_metric(evaluation.get('micro_wer'))}",
            f"- Micro CER: {_metric(evaluation.get('micro_cer'))}",
            f"- Macro WER: {_metric(evaluation.get('macro_wer'))}",
            f"- Macro CER: {_metric(evaluation.get('macro_cer'))}",
            "",
        ]
    )
    if per_language:
        lines.extend(["| Language | Samples | WER | CER |", "|---|---:|---:|---:|"])
        for row in per_language:
            lines.append(
                f"| {row.get('language')} | {row.get('samples')} | "
                f"{_metric(row.get('wer'))} | {_metric(row.get('cer'))} |"
            )
    else:
        lines.append("Per-language results: Not measured.")
    lines.extend(
        [
            "",
            "## Usage",
            "",
            "```python",
            "from peft import PeftModel",
            "from transformers import WhisperForConditionalGeneration, WhisperProcessor",
            "",
            f'base = WhisperForConditionalGeneration.from_pretrained("{context.get("base_model_id", "openai/whisper-small")}")',
            f'model = PeftModel.from_pretrained(base, "{context.get("repo_id", "rudra11/vaakku-whisper-small")}")',
            f'processor = WhisperProcessor.from_pretrained("{context.get("repo_id", "rudra11/vaakku-whisper-small")}")',
            "```",
            "",
            "The default published artifact is the PEFT adapter plus the processor,",
            "not a merged copy of the full Whisper weights.",
            "",
            "## Intended use",
            "",
            "Research and demonstration of multilingual Indic speech-to-text.",
            "",
            "## Limitations",
            "",
            "- This card does not claim a production or state-of-the-art system.",
            "- Quality depends on the datasets listed above and on the measured table.",
            "- Languages outside Tamil, Telugu, Hindi, Kannada, and Malayalam are outside this fine-tune.",
            "- A pipeline-validation run is not a publishable model.",
            "",
            "## License",
            "",
            str(context.get("license", "Not recorded.")),
            "",
            "Dataset licenses are listed separately and are not replaced by the adapter license.",
            "",
            "## Acknowledgements",
            "",
            "- OpenAI Whisper",
            "- Hugging Face Transformers and PEFT",
        ]
    )
    for dataset in datasets:
        lines.append(f"- {dataset.get('name')}")
    lines.append("")
    return "\n".join(lines)


def _show(value: object) -> str:
    if value is None or value == "":
        return "Not recorded."
    return str(value)


def _metric(value: object) -> str:
    if value is None:
        return "Not measured."
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
