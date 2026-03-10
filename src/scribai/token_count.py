"""Token counting helpers with optional tiktoken support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


_CHARS_PER_TOKEN_HEURISTIC = 4.0
_DEFAULT_FALLBACK_ENCODING = "o200k_base"


@dataclass(frozen=True)
class TokenEstimate:
    """Token count estimate and provenance details."""

    count: int
    method: str
    encoding: str | None = None


def estimate_token_count(
    text: str,
    *,
    model: str | None = None,
    encoding_name: str | None = None,
) -> TokenEstimate:
    """Estimate token count for text.

    Uses tiktoken when available. Falls back to a char-based heuristic when
    tiktoken is unavailable or encoding lookup fails.
    """
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        return TokenEstimate(
            count=_heuristic_token_count(text),
            method="heuristic_char4",
            encoding=None,
        )

    encoding, method = _resolve_tiktoken_encoding(
        tiktoken=tiktoken,
        model=model,
        encoding_name=encoding_name,
    )
    if encoding is None:
        return TokenEstimate(
            count=_heuristic_token_count(text),
            method="heuristic_char4",
            encoding=None,
        )

    try:
        encode_ordinary = getattr(encoding, "encode_ordinary", None)
        if callable(encode_ordinary):
            tokens = encode_ordinary(text)
        else:
            tokens = encoding.encode(text)
        return TokenEstimate(
            count=len(tokens),
            method=method,
            encoding=getattr(encoding, "name", None),
        )
    except Exception:
        return TokenEstimate(
            count=_heuristic_token_count(text),
            method="heuristic_char4",
            encoding=None,
        )


def estimated_chars_for_tokens(tokens: int) -> int:
    """Convert token count into approximate character budget."""
    return max(0, int(round(tokens * _CHARS_PER_TOKEN_HEURISTIC)))


def _resolve_tiktoken_encoding(
    *,
    tiktoken: Any,
    model: str | None,
    encoding_name: str | None,
) -> tuple[Any | None, str]:
    if model:
        try:
            return tiktoken.encoding_for_model(model), "tiktoken_model"
        except Exception:
            pass

    if encoding_name:
        try:
            return tiktoken.get_encoding(encoding_name), "tiktoken_encoding"
        except Exception:
            pass

    try:
        return tiktoken.get_encoding(_DEFAULT_FALLBACK_ENCODING), "tiktoken_default"
    except Exception:
        return None, "heuristic_char4"


def _heuristic_token_count(text: str) -> int:
    return max(0, int(round(len(text) / _CHARS_PER_TOKEN_HEURISTIC)))
