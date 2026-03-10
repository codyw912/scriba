"""Cerebras model metadata helpers."""

from __future__ import annotations

import os


_CEREBRAS_CONTEXT_LENGTHS: dict[str, dict[str, int]] = {
    "gpt-oss-120b": {
        "free": 65_000,
        "paygo": 131_000,
    },
    "llama3.1-8b": {
        "free": 8_000,
        "paygo": 32_000,
    },
    "qwen-3-235b-a22b-instruct-2507": {
        "free": 65_000,
        "paygo": 131_000,
    },
    "zai-glm-4.7": {
        "free": 64_000,
        "paygo": 131_000,
    },
}

_CEREBRAS_MAX_OUTPUT_TOKENS: dict[str, dict[str, int]] = {
    "gpt-oss-120b": {
        "free": 32_000,
        "paygo": 40_000,
    },
    "llama3.1-8b": {
        "free": 8_000,
        "paygo": 8_000,
    },
    "qwen-3-235b-a22b-instruct-2507": {
        "free": 32_000,
        "paygo": 40_000,
    },
    "zai-glm-4.7": {
        "free": 40_000,
        "paygo": 40_000,
    },
}


def lookup_context_length_from_cerebras(*, model: str, provider: str) -> int | None:
    """Return context length using built-in Cerebras model catalog."""
    if provider.strip().lower() != "cerebras":
        return None

    normalized_model = model.strip().lower()
    tiers = _CEREBRAS_CONTEXT_LENGTHS.get(normalized_model)
    if tiers is None:
        return None

    tier = _resolved_cerebras_tier()
    return tiers.get(tier) or tiers.get("free")


def lookup_max_output_tokens_from_cerebras(*, model: str, provider: str) -> int | None:
    """Return max output token limit from Cerebras model catalog."""
    if provider.strip().lower() != "cerebras":
        return None

    normalized_model = model.strip().lower()
    tiers = _CEREBRAS_MAX_OUTPUT_TOKENS.get(normalized_model)
    if tiers is None:
        return None

    tier = _resolved_cerebras_tier()
    return tiers.get(tier) or tiers.get("free")


def _resolved_cerebras_tier() -> str:
    raw = os.getenv("SCRIBAI_CEREBRAS_TIER", "free").strip().lower()
    if raw in {"paygo", "paid", "developer"}:
        return "paygo"
    return "free"
