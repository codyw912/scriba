"""Cerebras SDK backend adapter and client."""

from __future__ import annotations

import time
from typing import Any

from scriba.pipeline.backends.adapters.base import BackendAdapter
from scriba.pipeline.backends.errors import (
    BackendError,
    ContextWindowError,
    ModelClientError,
    RateLimitError,
)
from scriba.pipeline.backends.rate_limit import choose_retry_delay
from scriba.pipeline.backends.metadata_cerebras import (
    lookup_context_length_from_cerebras,
    lookup_max_output_tokens_from_cerebras,
)
from scriba.pipeline.backends.response_parsing import (
    coerce_completion_payload,
    coerce_usage_int,
    extract_completion_text,
    extract_provider_error_message,
    sanitize_model_markdown,
)
from scriba.pipeline.backends.types import (
    ChunkingHints,
    CompletionResult,
    ModelEndpoint,
)


_MAX_MODEL_REQUEST_ATTEMPTS = 3
_MODEL_REQUEST_RETRY_BASE_DELAY_S = 0.75


class CerebrasSDKChatClient:
    """Cerebras SDK chat completion client."""

    def __init__(self, endpoint: ModelEndpoint) -> None:
        self.endpoint = endpoint

        try:
            from cerebras.cloud.sdk import Cerebras  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ModelClientError(
                "cerebras_sdk adapter requires the optional package "
                "'cerebras-cloud-sdk'. Install with: uv add cerebras-cloud-sdk"
            ) from exc

        self._client = Cerebras(api_key=endpoint.api_key or None)

    def complete(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float,
        request_timeout_s: int,
        max_output_tokens: int | None,
        reasoning_effort: str | None = None,
        reasoning_exclude: bool | None = None,
    ) -> CompletionResult:
        del request_timeout_s

        request_payload: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_output_tokens is not None:
            request_payload["max_tokens"] = max_output_tokens
        if reasoning_effort is not None:
            request_payload["reasoning_effort"] = reasoning_effort
        if reasoning_exclude is not None:
            request_payload["clear_thinking"] = bool(reasoning_exclude)

        started = time.perf_counter()
        response: Any = None
        for attempt in range(1, _MAX_MODEL_REQUEST_ATTEMPTS + 1):
            try:
                response = self._client.chat.completions.create(**request_payload)
                break
            except Exception as exc:  # pragma: no cover - external SDK behavior
                error_text = str(exc)
                if _looks_like_context_error(error_text):
                    raise ContextWindowError(error_text) from exc
                if _looks_like_retryable_provider_error(error_text):
                    retry_delay_s = choose_retry_delay(
                        attempt=attempt,
                        base_delay_s=_MODEL_REQUEST_RETRY_BASE_DELAY_S,
                        headers=_headers_from_exception(exc),
                        error_text=error_text,
                    )
                    if attempt < _MAX_MODEL_REQUEST_ATTEMPTS:
                        time.sleep(retry_delay_s)
                        continue
                    raise RateLimitError(
                        f"Cerebras SDK request failed: {error_text}",
                        retry_after_s=retry_delay_s,
                    ) from exc
                raise ModelClientError(
                    f"Cerebras SDK request failed: {error_text}"
                ) from exc

        if response is None:
            raise ModelClientError(
                "Cerebras SDK request failed without response payload"
            )

        elapsed_s = time.perf_counter() - started
        data = coerce_completion_payload(response)

        provider_error = extract_provider_error_message(data)
        if provider_error:
            raise ModelClientError(
                f"Model API returned error payload: {provider_error}"
            )

        text = extract_completion_text(data)
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        prompt_tokens = coerce_usage_int(
            usage.get("prompt_tokens") if isinstance(usage, dict) else None
        )
        completion_tokens = coerce_usage_int(
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        total_tokens = coerce_usage_int(
            usage.get("total_tokens") if isinstance(usage, dict) else None
        )
        if (
            total_tokens is None
            and prompt_tokens is not None
            and completion_tokens is not None
        ):
            total_tokens = prompt_tokens + completion_tokens

        return CompletionResult(
            text=sanitize_model_markdown(text),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_s=elapsed_s,
            requests=1,
            split_count=0,
        )


class CerebrasSDKBackendAdapter(BackendAdapter):
    """Cerebras SDK adapter (health checked on first request)."""

    def ensure_ready(self, *, model: str) -> None:
        del model
        if not self.config.api_key.strip():
            raise BackendError(
                f"Backend '{self.name}' requires non-empty api_key for cerebras_sdk"
            )

    def create_chat_client(self, *, endpoint: ModelEndpoint) -> CerebrasSDKChatClient:
        return CerebrasSDKChatClient(endpoint)

    def model_chunking_hints(self, *, model: str) -> ChunkingHints:
        context_length = lookup_context_length_from_cerebras(
            model=model,
            provider=self.config.provider,
        )
        max_output_tokens_limit = lookup_max_output_tokens_from_cerebras(
            model=model,
            provider=self.config.provider,
        )
        return ChunkingHints(
            context_length=context_length,
            context_length_source="cerebras_model_catalog" if context_length else None,
            max_output_tokens_limit=max_output_tokens_limit,
        )


def _headers_from_exception(exc: Exception) -> dict[str, str] | None:
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            try:
                return {str(k).lower(): str(v) for k, v in headers.items()}
            except Exception:
                pass

    headers_attr = getattr(exc, "headers", None)
    if headers_attr is not None:
        try:
            return {str(k).lower(): str(v) for k, v in headers_attr.items()}
        except Exception:
            return None
    return None


def _looks_like_retryable_provider_error(error_text: str) -> bool:
    text = error_text.lower()
    return (
        "too_many_requests" in text
        or "rate limit" in text
        or "queue_exceeded" in text
        or "timeout" in text
        or "temporarily unavailable" in text
        or "error code: 429" in text
        or "error code: 500" in text
        or "error code: 502" in text
        or "error code: 503" in text
        or "error code: 504" in text
    )


def _looks_like_context_error(response_text: str) -> bool:
    text = response_text.lower()
    return (
        "exceeds the available context size" in text
        or "context" in text
        and "exceed" in text
        or "n_ctx" in text
        or "cannot truncate prompt" in text
        or "maximum context" in text
    )
