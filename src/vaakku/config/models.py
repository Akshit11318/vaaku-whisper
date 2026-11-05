"""YAML configuration models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from vaakku.constants import SAMPLE_RATE, SPLIT_ORDER, SUPPORTED_LANGUAGES


class LoggingConfig(BaseModel):
    level: str = "INFO"
    dir: str = "logs"
    json_logs: bool = True
    max_bytes: int = 10_485_760
    backup_count: int = 5

    @field_validator("level")
    @classmethod
    def known_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log level must be one of {sorted(allowed)}")
        return upper


class DataSourceConfig(BaseModel):
    name: str
    kind: Literal["manifest", "fleurs", "common_voice", "indicvoices", "kathbath", "vistaar"]
    enabled: bool = False
    path: str | None = None
    allow_download: bool = False
    hf_dataset: str | None = None
    hf_config: str | None = None
    hf_split: str | None = None

    @model_validator(mode="after")
    def enabled_sources_have_a_location(self) -> DataSourceConfig:
        if not self.enabled:
            return self
        if self.kind == "manifest" and not self.path:
            raise ValueError(f"source {self.name} is enabled but has no path")
        if self.kind != "manifest" and not self.path and not self.allow_download:
            raise ValueError(
                f"source {self.name} needs a local path, or allow_download: true "
                "after you have confirmed access and license"
            )
        if self.allow_download and not self.hf_dataset:
            raise ValueError(f"source {self.name} sets allow_download without hf_dataset")
        return self


class SplitConfig(BaseModel):
    train: float = 0.9
    validation: float = 0.05
    test: float = 0.05

    @model_validator(mode="after")
    def ratios_are_usable(self) -> SplitConfig:
        values = [self.train, self.validation, self.test]
        if any(value <= 0 for value in values):
            raise ValueError("train, validation, and test ratios must all be > 0")
        if abs(sum(values) - 1.0) > 1e-6:
            raise ValueError("split ratios must sum to 1")
        return self

    def as_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in SPLIT_ORDER}


class DataConfig(BaseModel):
    seed: int = 42
    output_dir: str = "data/prepared"
    sample_rate: int = SAMPLE_RATE
    min_duration_s: float = 0.5
    max_duration_s: float = 30.0
    languages: list[str] = Field(default_factory=lambda: list(SUPPORTED_LANGUAGES))
    sources: list[DataSourceConfig] = Field(default_factory=list)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    speaker_disjoint: bool = True
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("languages")
    @classmethod
    def known_languages(cls, value: list[str]) -> list[str]:
        return _validate_languages(value)

    @field_validator("sample_rate")
    @classmethod
    def whisper_rate(cls, value: int) -> int:
        if value != SAMPLE_RATE:
            raise ValueError(f"sample_rate must be {SAMPLE_RATE} for Whisper")
        return value

    @model_validator(mode="after")
    def duration_window(self) -> DataConfig:
        if self.max_duration_s <= self.min_duration_s:
            raise ValueError("max_duration_s must be greater than min_duration_s")
        return self


class LoraSettings(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: Literal["none", "lora_only", "all"] = "none"

    @model_validator(mode="after")
    def ranks_are_positive(self) -> LoraSettings:
        if self.r < 1 or self.alpha < 1:
            raise ValueError("LoRA r and alpha must be >= 1")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.target_modules:
            raise ValueError("LoRA target_modules cannot be empty")
        return self


class TrainConfig(BaseModel):
    purpose: Literal["pipeline_validation", "experiment"]
    run_name: str
    seed: int = 42
    output_dir: str = "artifacts/runs"
    base_model_id: str = "openai/whisper-small"
    model_init: Literal["pretrained", "tiny_random"] = "pretrained"
    languages: list[str] = Field(default_factory=lambda: list(SUPPORTED_LANGUAGES))
    train_manifest: str | None = None
    validation_manifest: str | None = None
    synthetic: bool = False
    synthetic_train_samples: int = 4
    synthetic_eval_samples: int = 2
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    num_epochs: int = 3
    max_steps: int | None = None
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 50
    max_grad_norm: float = 1.0
    precision: Literal["auto", "fp32", "fp16", "bf16"] = "auto"
    gradient_checkpointing: bool = True
    num_workers: int = 0
    max_audio_seconds: float = 30.0
    sample_rate: int = SAMPLE_RATE
    lora: LoraSettings = Field(default_factory=LoraSettings)
    eval_steps: int = 200
    save_steps: int = 200
    early_stopping_patience: int | None = 3
    early_stopping_min_delta: float = 0.0
    selection_metric: Literal["eval_loss"] = "eval_loss"
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("languages")
    @classmethod
    def known_languages(cls, value: list[str]) -> list[str]:
        return _validate_languages(value)

    @field_validator("sample_rate")
    @classmethod
    def whisper_rate(cls, value: int) -> int:
        if value != SAMPLE_RATE:
            raise ValueError(f"sample_rate must be {SAMPLE_RATE} for Whisper")
        return value

    @model_validator(mode="after")
    def training_choices_are_consistent(self) -> TrainConfig:
        if self.num_epochs < 1:
            raise ValueError("num_epochs must be >= 1")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be >= 1 when set")
        if self.per_device_train_batch_size < 1 or self.per_device_eval_batch_size < 1:
            raise ValueError("batch sizes must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.eval_steps < 1 or self.save_steps < 1:
            raise ValueError("eval_steps and save_steps must be >= 1")
        if self.synthetic and self.purpose != "pipeline_validation":
            raise ValueError("synthetic data is only allowed when purpose is pipeline_validation")
        if self.model_init == "tiny_random" and self.purpose != "pipeline_validation":
            raise ValueError("tiny_random init is only allowed for pipeline_validation")
        needs_manifests = not self.synthetic and self.model_init != "tiny_random"
        if needs_manifests and (not self.train_manifest or not self.validation_manifest):
            raise ValueError("train_manifest and validation_manifest are required")
        return self


class EvalConfig(BaseModel):
    test_manifest: str
    expect_split: Literal["test"] = "test"
    output_dir: str = "artifacts/eval"
    model_path: str | None = None
    base_model_id: str = "openai/whisper-small"
    compare_base: bool = False
    languages: list[str] = Field(default_factory=lambda: list(SUPPORTED_LANGUAGES))
    max_samples: int | None = None
    max_new_tokens: int = 225
    num_beams: int = 1
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = 1
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("languages")
    @classmethod
    def known_languages(cls, value: list[str]) -> list[str]:
        return _validate_languages(value)

    @field_validator("num_beams", "max_new_tokens", "batch_size")
    @classmethod
    def positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("decoding and batch settings must be >= 1")
        return value


class DatasetRecord(BaseModel):
    name: str
    url: str = ""
    license: str
    split_method: str = ""


class PublishConfig(BaseModel):
    repo_id: str = "rudra11/vaakku-whisper-small"
    token_env: str = "HF_TOKEN"
    mode: Literal["peft_adapter", "merged"] = "peft_adapter"
    run_dir: str
    checkpoint: str = "best"
    eval_results: str | None = None
    license: str
    datasets: list[DatasetRecord]
    private: bool = False
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("token_env")
    @classmethod
    def token_env_is_a_name(cls, value: str) -> str:
        if not value.isidentifier():
            raise ValueError("token_env must be an environment variable name, not a token")
        lowered = value.lower()
        if lowered.startswith("hf_") and value != "HF_TOKEN":
            raise ValueError("token_env looks like a token. Put the variable name here.")
        return value

    @field_validator("repo_id")
    @classmethod
    def repo_shape(cls, value: str) -> str:
        if value.count("/") != 1 or not all(value.split("/")):
            raise ValueError("repo_id must look like username/model-name")
        return value


def _validate_languages(value: list[str]) -> list[str]:
    if not value:
        raise ValueError("languages cannot be empty")
    unknown = [lang for lang in value if lang not in SUPPORTED_LANGUAGES]
    if unknown:
        raise ValueError(
            f"Unsupported languages: {unknown}. Supported: {list(SUPPORTED_LANGUAGES)}"
        )
    if len(set(value)) != len(value):
        raise ValueError("languages contains duplicates")
    return value
