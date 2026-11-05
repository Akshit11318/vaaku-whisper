"""Load YAML configs and expand path variables. Tokens are not expanded here."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel, ValidationError

from vaakku.config.models import DataConfig, EvalConfig, PublishConfig, TrainConfig
from vaakku.errors import ConfigError

T = TypeVar("T", bound=BaseModel)
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def expand_env(value: object) -> object:
    """Expand ${VAR:-default} in strings. Missing variables use the default or empty string."""
    if isinstance(value, str):

        def replace(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), match.group(2) or "")

        return _ENV.sub(replace, value)
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    return value


def read_yaml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    expanded = expand_env(loaded)
    if not isinstance(expanded, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return expanded


def load_model(path: Path, model_type: type[T]) -> T:
    try:
        return model_type.model_validate(read_yaml(path))
    except ValidationError as exc:
        raise ConfigError(f"Invalid config {path}:\n{exc}") from exc


def load_data_config(path: Path) -> DataConfig:
    return load_model(path, DataConfig)


def load_train_config(path: Path) -> TrainConfig:
    return load_model(path, TrainConfig)


def load_eval_config(path: Path) -> EvalConfig:
    return load_model(path, EvalConfig)


def load_publish_config(path: Path) -> PublishConfig:
    return load_model(path, PublishConfig)
