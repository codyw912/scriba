"""Rate-limit and retry delay helpers for backend clients."""

from __future__ import annotations

from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import re
from typing import Mapping


_MAX_RATE_LIMIT_DELAY_S = 30.0
_MIN_DELAY_S = 0.05
_RETRY_AFTER_RE = re.compile(r"retry\s+after\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def backoff_delay_seconds(*, attempt: int, base_delay_s: float) -> float:
    """Exponential backoff delay for given attempt number."""
    if attempt <= 0:
        return max(_MIN_DELAY_S, base_delay_s)
    return max(_MIN_DELAY_S, base_delay_s * (2 ** (attempt - 1)))


def retry_delay_from_headers(headers: Mapping[str, str] | None) -> float | None:
    """Extract retry delay from standard and vendor rate-limit headers."""
    if not headers:
        return None

    retry_after = _parse_retry_after(headers.get("retry-after"))
    token_reset = _parse_seconds(headers.get("x-ratelimit-reset-tokens-minute"))
    req_reset = _parse_seconds(headers.get("x-ratelimit-reset-requests-day"))

    for value in (retry_after, token_reset, req_reset):
        if value is None:
            continue
        return _clamp_delay(value)

    return None


def retry_delay_from_error_text(error_text: str) -> float | None:
    """Extract retry delay hint from provider error text."""
    if not error_text:
        return None
    match = _RETRY_AFTER_RE.search(error_text)
    if match is None:
        return None
    parsed = _parse_seconds(match.group(1))
    if parsed is None:
        return None
    return _clamp_delay(parsed)


def choose_retry_delay(
    *,
    attempt: int,
    base_delay_s: float,
    headers: Mapping[str, str] | None = None,
    error_text: str = "",
) -> float:
    """Choose delay from headers/text hints, falling back to exponential backoff."""
    header_delay = retry_delay_from_headers(headers)
    if header_delay is not None:
        return header_delay

    text_delay = retry_delay_from_error_text(error_text)
    if text_delay is not None:
        return text_delay

    return backoff_delay_seconds(attempt=attempt, base_delay_s=base_delay_s)


def _parse_seconds(raw: str | None) -> float | None:
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if parsed <= 0:
        return None
    return parsed


def _parse_retry_after(raw: str | None) -> float | None:
    if raw is None:
        return None

    numeric = _parse_seconds(raw)
    if numeric is not None:
        return numeric

    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    now = datetime.now(UTC)
    delta = (dt - now).total_seconds()
    if delta <= 0:
        return None
    return delta


def _clamp_delay(seconds: float) -> float:
    return max(_MIN_DELAY_S, min(_MAX_RATE_LIMIT_DELAY_S, seconds))
