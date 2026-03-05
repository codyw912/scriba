"""Tests for backend rate-limit retry delay helpers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from scriba.pipeline.backends.rate_limit import (
    backoff_delay_seconds,
    choose_retry_delay,
    retry_delay_from_headers,
)


def test_retry_delay_prefers_retry_after_seconds() -> None:
    headers = {
        "retry-after": "2.5",
        "x-ratelimit-reset-tokens-minute": "8.0",
    }
    assert retry_delay_from_headers(headers) == 2.5


def test_retry_delay_parses_retry_after_http_date() -> None:
    target = (datetime.now(UTC) + timedelta(seconds=3)).strftime(
        "%a, %d %b %Y %H:%M:%S GMT"
    )
    delay = retry_delay_from_headers({"retry-after": target})
    assert delay is not None
    assert 0.05 <= delay <= 30.0


def test_choose_retry_delay_uses_header_hint_over_backoff() -> None:
    delay = choose_retry_delay(
        attempt=3,
        base_delay_s=0.75,
        headers={"x-ratelimit-reset-tokens-minute": "4.2"},
        error_text="",
    )
    assert delay == 4.2


def test_backoff_delay_is_exponential() -> None:
    assert backoff_delay_seconds(attempt=1, base_delay_s=0.75) == 0.75
    assert backoff_delay_seconds(attempt=2, base_delay_s=0.75) == 1.5
    assert backoff_delay_seconds(attempt=3, base_delay_s=0.75) == 3.0
