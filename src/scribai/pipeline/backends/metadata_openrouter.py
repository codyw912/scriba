"""OpenRouter model metadata helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_OPENROUTER_MODELS_PATH = (
    Path(__file__).resolve().parents[4] / "samples" / "openrouter_models.json"
)
_OPENROUTER_MODELS_BY_MODEL: dict[str, int] | None = None
_OPENROUTER_VERSION_SUFFIX_RE = re.compile(r"-(?:\d{8})(?:-[0-9a-z]+)?$")


def lookup_context_length_from_openrouter(*, model: str, provider: str) -> int | None:
    """Return context length from local OpenRouter catalog for matching model."""
    if provider.lower() != "openrouter":
        return None

    data = _load_openrouter_model_contexts()
    if not data:
        return None

    model_key = _normalize_model_id_for_lookup(model)
    if model_key in data:
        return data[model_key]

    for known_key, context_length in data.items():
        if _model_keys_match(model_key, known_key):
            return context_length
    return None


def _load_openrouter_model_contexts() -> dict[str, int]:
    global _OPENROUTER_MODELS_BY_MODEL
    if _OPENROUTER_MODELS_BY_MODEL is not None:
        return _OPENROUTER_MODELS_BY_MODEL

    if not _OPENROUTER_MODELS_PATH.exists():
        _OPENROUTER_MODELS_BY_MODEL = {}
        return _OPENROUTER_MODELS_BY_MODEL

    try:
        payload = json.loads(_OPENROUTER_MODELS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _OPENROUTER_MODELS_BY_MODEL = {}
        return _OPENROUTER_MODELS_BY_MODEL

    records = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        _OPENROUTER_MODELS_BY_MODEL = {}
        return _OPENROUTER_MODELS_BY_MODEL

    contexts: dict[str, int] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        context_length = _coerce_context_length(item.get("context_length"))
        if context_length is None:
            top_provider = item.get("top_provider")
            if isinstance(top_provider, dict):
                context_length = _coerce_context_length(
                    top_provider.get("context_length")
                )
        if context_length is None:
            continue

        for raw_id in (item.get("id"), item.get("canonical_slug")):
            if isinstance(raw_id, str) and raw_id.strip():
                key = _normalize_model_id_for_lookup(raw_id)
                contexts[key] = context_length

    _OPENROUTER_MODELS_BY_MODEL = contexts
    return contexts


def _coerce_context_length(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = int(float(stripped))
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _normalize_model_id_for_lookup(model: str) -> str:
    return _strip_openrouter_version_suffix(model.strip().lower())


def _strip_openrouter_version_suffix(model: str) -> str:
    return _OPENROUTER_VERSION_SUFFIX_RE.sub("", model)


def _model_keys_match(left: str, right: str) -> bool:
    if left == right:
        return True
    if "/" in left and "/" in right:
        l_provider, _, l_name = left.partition("/")
        r_provider, _, r_name = right.partition("/")
        if l_provider != r_provider:
            return False
        return _strip_openrouter_version_suffix(
            l_name
        ) == _strip_openrouter_version_suffix(r_name)
    return _strip_openrouter_version_suffix(left) == _strip_openrouter_version_suffix(
        right
    )
