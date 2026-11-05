"""Precision selection from the requested mode and the actual device."""

from __future__ import annotations

from vaakku.errors import TrainingError


def select_precision(requested: str, cuda: bool, bf16_supported: bool) -> str:
    if requested == "fp32":
        return "fp32"
    if requested == "fp16":
        if not cuda:
            raise TrainingError("fp16 was requested, but CUDA is not available")
        return "fp16"
    if requested == "bf16":
        if not cuda or not bf16_supported:
            raise TrainingError("bf16 was requested, but this GPU does not support it")
        return "bf16"
    if requested == "auto":
        if not cuda:
            return "fp32"
        if bf16_supported:
            return "bf16"
        return "fp16"
    raise TrainingError(f"Unknown precision: {requested}")
