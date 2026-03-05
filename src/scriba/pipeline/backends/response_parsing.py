"""Shared response parsing helpers for model clients."""

from __future__ import annotations

import json
import re
from typing import Any

from scriba.pipeline.backends.errors import ModelClientError


def parse_json_response_payload(*, status_code: int, body_text: str) -> dict[str, Any]:
    """Parse JSON response body and validate object shape."""
    try:
        data = json.loads(body_text)
    except ValueError as exc:
        preview = body_text.strip().replace("\n", " ")
        if len(preview) > 240:
            preview = f"{preview[:240]}..."
        raise ModelClientError(
            "Model response was not valid JSON "
            f"(status={status_code}): {preview or 'empty body'}"
        ) from exc

    if not isinstance(data, dict):
        raise ModelClientError(
            f"Model response JSON must be an object, got {type(data).__name__}."
        )

    provider_error = extract_provider_error_message(data)
    if provider_error:
        raise ModelClientError(f"Model API returned error payload: {provider_error}")

    return data


def coerce_completion_payload(payload: Any) -> dict[str, Any]:
    """Coerce provider SDK response objects into a dict payload."""
    if isinstance(payload, dict):
        return payload

    for method_name in ("model_dump", "dict", "to_dict"):
        method = getattr(payload, method_name, None)
        if callable(method):
            try:
                maybe_dict = method()
            except TypeError:
                continue
            if isinstance(maybe_dict, dict):
                return maybe_dict

    json_method = getattr(payload, "model_dump_json", None)
    if callable(json_method):
        try:
            serialized = json_method()
            maybe_dict = json.loads(str(serialized))
            if isinstance(maybe_dict, dict):
                return maybe_dict
        except Exception:
            pass

    raise ModelClientError(
        "Unsupported completion payload type from provider SDK: "
        f"{type(payload).__name__}"
    )


def extract_provider_error_message(payload: dict[str, Any]) -> str | None:
    """Extract provider-native error details if present in payload."""
    error = payload.get("error")
    if isinstance(error, str):
        return error.strip() or None
    if isinstance(error, dict):
        code = error.get("code")
        message = error.get("message") or error.get("detail") or error.get("type")
        if message is None:
            return f"{error}"
        if code:
            return f"{code}: {message}"
        return str(message)
    return None


def extract_completion_text(payload: dict[str, Any]) -> str:
    """Extract textual completion content across common response variants."""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
            if isinstance(message, dict) and "content" in message:
                return coerce_completion_content_to_text(message.get("content"))
            if "text" in first_choice:
                return coerce_completion_content_to_text(first_choice.get("text"))
            delta = first_choice.get("delta")
            if isinstance(delta, dict) and "content" in delta:
                return coerce_completion_content_to_text(delta.get("content"))

    if "output_text" in payload:
        return coerce_completion_content_to_text(payload.get("output_text"))

    if "content" in payload:
        return coerce_completion_content_to_text(payload.get("content"))

    keys = ", ".join(sorted(payload.keys())) or "none"
    raise ModelClientError(
        f"Model response missing completion content (available keys: {keys})"
    )


def coerce_completion_content_to_text(content: Any) -> str:
    """Convert content object variants into plain markdown text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text_value = item.get("text")
                if text_value is not None:
                    parts.append(str(text_value))
                    continue
                content_value = item.get("content")
                if content_value is not None:
                    parts.append(str(content_value))
                    continue
            parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "")
        if "content" in content:
            return str(content.get("content") or "")
    return str(content)


def coerce_usage_int(value: Any) -> int | None:
    """Coerce token usage values from provider payloads."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def sanitize_model_markdown(text: str) -> str:
    """Strip reasoning tags and return stable markdown text."""
    original = text
    cleaned = text.strip()

    if "</think>" in cleaned:
        cleaned = cleaned.rsplit("</think>", 1)[1].lstrip()
    elif "<think>" in cleaned:
        cleaned = cleaned.split("<think>", 1)[0].rstrip()

    cleaned = re.sub(
        r"<think>.*?</think>\s*", "", cleaned, flags=re.IGNORECASE | re.DOTALL
    )
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    if cleaned:
        return cleaned

    fallback = re.sub(r"</?think>", "", original, flags=re.IGNORECASE).strip()
    return fallback if fallback else original.strip()
