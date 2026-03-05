"""Tests for backend transport client behavior."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from scriba.pipeline.backends import ModelEndpoint
from scriba.pipeline.backends.adapters.openai_http import OpenAIHTTPChatClient
from scriba.pipeline.backends.errors import ModelClientError, RateLimitError


class _DummyClient:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    def __enter__(self) -> "_DummyClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    def post(self, *args, **kwargs) -> httpx.Response:  # type: ignore[no-untyped-def]
        return self._response


class _SequenceClient:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)

    def __enter__(self) -> "_SequenceClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    def post(self, *args, **kwargs) -> httpx.Response:  # type: ignore[no-untyped-def]
        if not self._responses:
            raise AssertionError("No more queued responses")
        return self._responses.pop(0)


def _endpoint() -> ModelEndpoint:
    return ModelEndpoint(
        role="normalize_text",
        backend_name="remote_text",
        base_url="http://127.0.0.1:8090",
        inference_url="http://127.0.0.1:8090/v1/chat/completions",
        model="qwen/qwen3.5-35b-a3b",
        api_key="",
        adapter="openai_http",
        topology="remote",
        provider="openrouter",
    )


def test_openai_http_client_parses_text_content_list() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:8090/v1/chat/completions")
    response = httpx.Response(
        200,
        request=request,
        json={
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
        },
    )
    client = OpenAIHTTPChatClient(_endpoint())

    with patch(
        "scriba.pipeline.backends.adapters.openai_http.httpx.Client",
        return_value=_DummyClient(response),
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


def test_openai_http_client_raises_for_error_payload() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:8090/v1/chat/completions")
    response = httpx.Response(
        200,
        request=request,
        json={"error": {"code": "rate_limited", "message": "try again later"}},
    )
    client = OpenAIHTTPChatClient(_endpoint())

    with patch(
        "scriba.pipeline.backends.adapters.openai_http.httpx.Client",
        return_value=_DummyClient(response),
    ):
        with pytest.raises(ModelClientError, match="rate_limited"):
            client.complete(
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.0,
                request_timeout_s=10,
                max_output_tokens=128,
            )


def test_openai_http_client_raises_for_missing_content() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:8090/v1/chat/completions")
    response = httpx.Response(200, request=request, json={"id": "abc123"})
    client = OpenAIHTTPChatClient(_endpoint())

    with patch(
        "scriba.pipeline.backends.adapters.openai_http.httpx.Client",
        return_value=_DummyClient(response),
    ):
        with pytest.raises(ModelClientError, match="missing completion content"):
            client.complete(
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.0,
                request_timeout_s=10,
                max_output_tokens=128,
            )


def test_openai_http_client_raises_for_non_json_body() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:8090/v1/chat/completions")
    response = httpx.Response(
        200,
        request=request,
        content=b"<html>not-json</html>",
        headers={"Content-Type": "text/html"},
    )
    client = OpenAIHTTPChatClient(_endpoint())

    with patch(
        "scriba.pipeline.backends.adapters.openai_http.httpx.Client",
        return_value=_DummyClient(response),
    ):
        with pytest.raises(ModelClientError, match="not valid JSON"):
            client.complete(
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.0,
                request_timeout_s=10,
                max_output_tokens=128,
            )


def test_openai_http_client_retries_retryable_http_status() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:8090/v1/chat/completions")
    first = httpx.Response(
        429,
        request=request,
        headers={"x-ratelimit-reset-tokens-minute": "3.5"},
        json={"error": {"message": "rate limited"}},
    )
    second = httpx.Response(
        200,
        request=request,
        json={
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )
    client = OpenAIHTTPChatClient(_endpoint())
    sequence = _SequenceClient([first, second])

    with (
        patch(
            "scriba.pipeline.backends.adapters.openai_http.httpx.Client",
            return_value=sequence,
        ),
        patch(
            "scriba.pipeline.backends.adapters.openai_http.time.sleep",
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


def test_openai_http_client_raises_rate_limit_error_after_retries() -> None:
    request = httpx.Request("POST", "http://127.0.0.1:8090/v1/chat/completions")
    responses = [
        httpx.Response(
            429,
            request=request,
            headers={"x-ratelimit-reset-tokens-minute": "2.0"},
            json={"error": {"message": "rate limited"}},
        ),
        httpx.Response(
            429,
            request=request,
            headers={"x-ratelimit-reset-tokens-minute": "2.0"},
            json={"error": {"message": "rate limited"}},
        ),
        httpx.Response(
            429,
            request=request,
            headers={"x-ratelimit-reset-tokens-minute": "2.0"},
            json={"error": {"message": "rate limited"}},
        ),
    ]
    client = OpenAIHTTPChatClient(_endpoint())

    with (
        patch(
            "scriba.pipeline.backends.adapters.openai_http.httpx.Client",
            return_value=_SequenceClient(responses),
        ),
        patch(
            "scriba.pipeline.backends.adapters.openai_http.time.sleep",
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
