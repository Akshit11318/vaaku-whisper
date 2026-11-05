"""Build a LoRA Whisper model and check which parameters train."""

from __future__ import annotations

from vaakku.config.models import TrainConfig
from vaakku.errors import TrainingError


def build_model(config: TrainConfig):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import WhisperForConditionalGeneration

    if config.model_init == "tiny_random":
        base = WhisperForConditionalGeneration(tiny_whisper_config())
    else:
        base = WhisperForConditionalGeneration.from_pretrained(config.base_model_id)
    base.config.use_cache = False
    missing = missing_target_modules(base, config.lora.target_modules)
    if missing:
        raise TrainingError(
            f"LoRA target modules are not present on this Whisper checkpoint: {missing}"
        )
    lora = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias=config.lora.bias,
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(base, lora)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    assert_lora_only(model)
    return model


def tiny_whisper_config():
    from transformers import WhisperConfig

    return WhisperConfig(
        vocab_size=128,
        num_mel_bins=80,
        d_model=64,
        encoder_layers=1,
        encoder_attention_heads=4,
        encoder_ffn_dim=128,
        decoder_layers=1,
        decoder_attention_heads=4,
        decoder_ffn_dim=128,
        max_source_positions=128,
        max_target_positions=32,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )


def missing_target_modules(model, target_modules: list[str]) -> list[str]:
    import torch

    names = {
        module_name.split(".")[-1]
        for module_name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    return [target for target in target_modules if target not in names]


def assert_lora_only(model) -> None:
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise TrainingError("LoRA setup produced no trainable parameters")
    unexpected = [name for name in trainable if "lora_" not in name]
    if unexpected:
        preview = ", ".join(unexpected[:8])
        raise TrainingError(f"Non-LoRA parameters are trainable: {preview}")


def trainable_parameter_count(model) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)
