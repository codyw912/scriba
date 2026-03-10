"""Tests for backend transport client behavior."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from scribai.pipeline.backends import ModelEndpoint
from scribai.pipeline.backends.adapters.litellm_adapter import LiteLLMChatClient
from scribai.pipeline.backends.errors import ModelClientError, RateLimitError


class _FakeLiteLLMError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


def _endpoint() -> ModelEndpoint:
    return ModelEndpoint(
        role="normalize_text",
        backend_name="remote_text",
        base_url="http://127.0.0.1:8090",
        inference_url="http://127.0.0.1:8090/v1/chat/completions",
        model="qwen/qwen3.5-35b-a3b",
        api_key="",
        adapter="litellm",
        topology="remote",
        provider="openrouter",
    )


def test_litellm_client_parses_text_content_list() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "text", "text": "First line"},
                        {"type": "text", "text": "Second line"},
                    ]
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    client = LiteLLMChatClient(_endpoint())

    with patch(
        "scribai.pipeline.backends.adapters.litellm_adapter.litellm_completion",
        return_value=response,
    ):
        result = client.complete(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.0,
            request_timeout_s=10,
            max_output_tokens=128,
        )

    assert result.text == "First line\nSecond line"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.total_tokens == 15


def test_litellm_client_raises_for_error_payload() -> None:
    response = {"error": {"code": "rate_limited", "message": "try again later"}}
    client = LiteLLMChatClient(_endpoint())

    with patch(
        "scribai.pipeline.backends.adapters.litellm_adapter.litellm_completion",
        return_value=response,
    ):
        with pytest.raises(ModelClientError, match="rate_limited"):
            client.complete(
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.0,
                request_timeout_s=10,
                max_output_tokens=128,
            )


def test_litellm_client_raises_for_missing_content() -> None:
    client = LiteLLMChatClient(_endpoint())

    with patch(
        "scribai.pipeline.backends.adapters.litellm_adapter.litellm_completion",
        return_value={"id": "abc123"},
    ):
        with pytest.raises(ModelClientError, match="missing completion content"):
            client.complete(
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.0,
                request_timeout_s=10,
                max_output_tokens=128,
            )


def test_litellm_client_retries_retryable_http_status() -> None:
    client = LiteLLMChatClient(_endpoint())

    call_queue: list[object] = [
        _FakeLiteLLMError(
            "rate limited",
            status_code=429,
            headers={"x-ratelimit-reset-tokens-minute": "3.5"},
        ),
        {
            "choices": [
                {
                    "message": {"content": "ok"},
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        },
    ]

    def _fake_completion(**_: object) -> object:
        if not call_queue:
            raise AssertionError("No more queued responses")
        next_item = call_queue.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item

    with (
        patch(
            "scribai.pipeline.backends.adapters.litellm_adapter.litellm_completion",
            side_effect=_fake_completion,
        ),
        patch(
            "scribai.pipeline.backends.adapters.litellm_adapter.time.sleep",
            return_value=None,
        ) as sleep_mock,
    ):
        result = client.complete(
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.0,
            request_timeout_s=10,
            max_output_tokens=128,
        )

    assert result.text == "ok"
    sleep_mock.assert_called_once_with(3.5)


def test_litellm_client_raises_rate_limit_error_after_retries() -> None:
    call_queue: list[object] = [
        _FakeLiteLLMError(
            "rate limited",
            status_code=429,
            headers={"x-ratelimit-reset-tokens-minute": "2.0"},
        ),
        _FakeLiteLLMError(
            "rate limited",
            status_code=429,
            headers={"x-ratelimit-reset-tokens-minute": "2.0"},
        ),
        _FakeLiteLLMError(
            "rate limited",
            status_code=429,
            headers={"x-ratelimit-reset-tokens-minute": "2.0"},
        ),
    ]

    def _fake_completion(**_: object) -> object:
        if not call_queue:
            raise AssertionError("No more queued responses")
        next_item = call_queue.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item

    client = LiteLLMChatClient(_endpoint())

    with (
        patch(
            "scribai.pipeline.backends.adapters.litellm_adapter.litellm_completion",
            side_effect=_fake_completion,
        ),
        patch(
            "scribai.pipeline.backends.adapters.litellm_adapter.time.sleep",
            return_value=None,
        ),
    ):
        with pytest.raises(RateLimitError, match="HTTP 429") as exc_info:
            client.complete(
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.0,
                request_timeout_s=10,
                max_output_tokens=128,
            )

    assert exc_info.value.retry_after_s == 2.0
