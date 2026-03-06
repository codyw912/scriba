"""LiteLLM-based backend adapters and client."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from typing import Any

import httpx

from scriba.pipeline.backends.adapters.base import BackendAdapter
from scriba.pipeline.backends.errors import (
    BackendError,
    ContextWindowError,
    ModelClientError,
    ModelRequestTimeoutError,
    RateLimitError,
)
from scriba.pipeline.backends.metadata_cerebras import (
    lookup_context_length_from_cerebras,
    lookup_max_output_tokens_from_cerebras,
)
from scriba.pipeline.backends.metadata_openrouter import (
    lookup_context_length_from_openrouter,
)
from scriba.pipeline.backends.rate_limit import choose_retry_delay
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
from scriba.pipeline.profile import BackendConfig

try:
    from litellm import completion as litellm_completion
except ImportError:  # pragma: no cover - import path validated in runtime
    litellm_completion = None


DEFAULT_HEALTH_POST_PAYLOAD = {
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 1,
}

_MAX_MODEL_REQUEST_ATTEMPTS = 3
_RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
_MODEL_REQUEST_RETRY_BASE_DELAY_S = 0.75
_OPENAI_COMPATIBLE_PROVIDERS = {
    "local",
    "lmstudio",
    "llama_cpp",
    "vllm",
    "openai_compatible",
    "glm_ocr",
}
_PROVIDER_PREFIX_MAP = {
    "openrouter": "openrouter",
    "cerebras": "cerebras",
    "openai": "openai",
    "anthropic": "anthropic",
    "gemini": "gemini",
    "google": "gemini",
    "groq": "groq",
    "mistral": "mistral",
    "together": "together_ai",
    "together_ai": "together_ai",
    "fireworks": "fireworks_ai",
    "fireworks_ai": "fireworks_ai",
    "xai": "xai",
    "deepseek": "deepseek",
}
_KNOWN_MODEL_PREFIXES = set(_PROVIDER_PREFIX_MAP.values()) | {
    "vertex_ai",
    "azure",
    "ollama",
}


class LiteLLMChatClient:
    """LiteLLM chat completion client."""

    def __init__(self, endpoint: ModelEndpoint) -> None:
        self.endpoint = endpoint

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
        if litellm_completion is None:  # pragma: no cover - dependency guard
            raise ModelClientError(
                "litellm adapter requires the 'litellm' package. Install with: uv add litellm"
            )

        payload: dict[str, Any] = {
            "model": _resolve_litellm_model_name(
                provider=self.endpoint.provider,
                model=self.endpoint.model,
            ),
            "messages": messages,
            "temperature": temperature,
            "timeout": float(request_timeout_s),
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens

        if self.endpoint.api_key:
            payload["api_key"] = self.endpoint.api_key
        if self.endpoint.base_url:
            payload["api_base"] = self.endpoint.base_url

        if reasoning_effort is not None or reasoning_exclude is not None:
            reasoning_payload: dict[str, Any] = {}
            if reasoning_effort is not None:
                reasoning_payload["effort"] = reasoning_effort
            if reasoning_exclude is not None:
                reasoning_payload["exclude"] = reasoning_exclude
            if reasoning_payload:
                payload["reasoning"] = reasoning_payload

        started = time.perf_counter()
        response_payload: dict[str, Any] | None = None
        for attempt in range(1, _MAX_MODEL_REQUEST_ATTEMPTS + 1):
            try:
                raw = litellm_completion(**payload)
                response_payload = coerce_completion_payload(raw)
                break
            except Exception as exc:  # pragma: no cover - provider/network behavior
                error_text = str(exc)
                status_code = _status_code_from_exception(exc)

                if _looks_like_context_error(error_text):
                    raise ContextWindowError(error_text) from exc

                if _looks_like_timeout_error(error_text):
                    if attempt < _MAX_MODEL_REQUEST_ATTEMPTS:
                        time.sleep(_model_request_retry_delay_s(attempt))
                        continue
                    raise ModelRequestTimeoutError(
                        f"Model request timed out after {request_timeout_s}s"
                    ) from exc

                if status_code == 429:
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
                        f"HTTP 429: {error_text}",
                        retry_after_s=retry_delay_s,
                    ) from exc

                if (
                    status_code in _RETRYABLE_HTTP_STATUS_CODES
                    or _looks_like_retryable_provider_error(error_text)
                ) and attempt < _MAX_MODEL_REQUEST_ATTEMPTS:
                    retry_delay_s = choose_retry_delay(
                        attempt=attempt,
                        base_delay_s=_MODEL_REQUEST_RETRY_BASE_DELAY_S,
                        headers=_headers_from_exception(exc),
                        error_text=error_text,
                    )
                    time.sleep(retry_delay_s)
                    continue

                raise ModelClientError(error_text) from exc

        if response_payload is None:
            raise ModelClientError("Model request failed without response payload")

        provider_error = extract_provider_error_message(response_payload)
        if provider_error:
            raise ModelClientError(
                f"Model API returned error payload: {provider_error}"
            )

        elapsed_s = time.perf_counter() - started
        text = extract_completion_text(response_payload)

        usage = (
            response_payload.get("usage", {})
            if isinstance(response_payload, dict)
            else {}
        )
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


class LocalProcessLiteLLMBackendAdapter(BackendAdapter):
    """Local process backend started via command."""

    def __init__(self, *, name: str, config: BackendConfig) -> None:
        super().__init__(name=name, config=config)
        self._process: subprocess.Popen[bytes] | None = None

    def ensure_ready(self, *, model: str) -> None:
        self._ensure_process_running()
        self._wait_until_healthy(model=model)

    def create_chat_client(self, *, endpoint: ModelEndpoint) -> LiteLLMChatClient:
        return LiteLLMChatClient(endpoint)

    def model_chunking_hints(self, *, model: str) -> ChunkingHints:
        return _chunking_hints_for_provider(provider=self.config.provider, model=model)

    def _ensure_process_running(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return

        if not self.config.command:
            raise BackendError(f"Local-process backend '{self.name}' has no command")

        cmd = shlex.split(self.config.command)
        env = os.environ.copy()
        env.update(self.config.env)
        passthrough_logs = os.getenv("SCRIBA_BACKEND_PASSTHROUGH_LOGS", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=None if passthrough_logs else subprocess.DEVNULL,
                stderr=None if passthrough_logs else subprocess.DEVNULL,
                env=env,
            )
        except OSError as exc:
            raise BackendError(
                f"Failed to start backend '{self.name}' with command: {self.config.command}"
            ) from exc

    def _wait_until_healthy(self, *, model: str) -> None:
        deadline = time.monotonic() + float(self.config.startup_timeout_s)
        started = time.monotonic()
        last_error: str = "unknown"
        next_progress_update = started

        self._print_progress(elapsed_s=0.0, last_error=last_error, final=False)

        while time.monotonic() < deadline:
            if self._process is None:
                raise BackendError(
                    f"Local-process backend '{self.name}' is not running"
                )
            if self._process.poll() is not None:
                raise BackendError(
                    f"Local-process backend '{self.name}' exited before becoming healthy"
                )

            ok, error = _probe_health(config=self.config, model=model)
            if ok:
                self._print_progress(
                    elapsed_s=time.monotonic() - started,
                    last_error="ok",
                    final=True,
                )
                return
            last_error = error

            now = time.monotonic()
            if now >= next_progress_update:
                self._print_progress(
                    elapsed_s=now - started,
                    last_error=last_error,
                    final=False,
                )
                next_progress_update = now + 10.0

            time.sleep(0.5)

        raise BackendError(
            f"Backend '{self.name}' did not become healthy in {self.config.startup_timeout_s}s ({last_error})"
        )

    def _print_progress(
        self,
        *,
        elapsed_s: float,
        last_error: str,
        final: bool,
    ) -> None:
        elapsed_int = int(max(0, elapsed_s))
        timeout = max(1, int(self.config.startup_timeout_s))
        ratio = min(1.0, elapsed_s / float(timeout))
        filled = int(ratio * 20)
        bar = "#" * filled + "-" * (20 - filled)
        state = "ready" if final else "starting"
        print(
            (
                f"[backend:{self.name}] {state} "
                f"[{bar}] {elapsed_int}s/{timeout}s "
                f"last_probe='{last_error}'"
            ),
            file=sys.stderr,
        )

    def stop(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is not None:
            self._process = None
            return

        self._process.terminate()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5)
        finally:
            self._process = None


class AttachedOrRemoteLiteLLMBackendAdapter(BackendAdapter):
    """Attached or remote LiteLLM-backed adapter."""

    def ensure_ready(self, *, model: str) -> None:
        if not self.config.base_url.strip():
            return
        ok, error = _probe_health(config=self.config, model=model)
        if not ok:
            raise BackendError(f"Backend '{self.name}' is not healthy ({error})")

    def create_chat_client(self, *, endpoint: ModelEndpoint) -> LiteLLMChatClient:
        return LiteLLMChatClient(endpoint)

    def model_chunking_hints(self, *, model: str) -> ChunkingHints:
        return _chunking_hints_for_provider(provider=self.config.provider, model=model)


def _chunking_hints_for_provider(*, provider: str, model: str) -> ChunkingHints:
    context_length = lookup_context_length_from_openrouter(
        model=model,
        provider=provider,
    )
    context_source = "openrouter_models" if context_length else None

    max_output_tokens_limit: int | None = None
    if context_length is None:
        context_length = lookup_context_length_from_cerebras(
            model=model,
            provider=provider,
        )
        context_source = "cerebras_model_catalog" if context_length else None
        max_output_tokens_limit = lookup_max_output_tokens_from_cerebras(
            model=model,
            provider=provider,
        )

    return ChunkingHints(
        context_length=context_length,
        context_length_source=context_source,
        max_output_tokens_limit=max_output_tokens_limit,
    )


def _probe_health(*, config: BackendConfig, model: str) -> tuple[bool, str]:
    method = config.health_method.upper()
    url = f"{config.base_url.rstrip('/')}{config.health_path}"
    request_kwargs: dict[str, Any] = {}

    headers = dict(config.health_headers)
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    if headers:
        request_kwargs["headers"] = headers

    if method == "POST":
        payload = (
            dict(config.health_payload)
            if config.health_payload is not None
            else dict(DEFAULT_HEALTH_POST_PAYLOAD)
        )
        if "model" not in payload:
            payload["model"] = model
        request_kwargs["json"] = payload

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.request(method, url, **request_kwargs)
        if 200 <= response.status_code < 300:
            return True, "ok"
        return False, f"HTTP {response.status_code}"
    except httpx.HTTPError as exc:
        return False, str(exc)


def _resolve_litellm_model_name(*, provider: str, model: str) -> str:
    stripped_model = model.strip()
    if not stripped_model:
        return model

    if "/" in stripped_model:
        maybe_prefix = stripped_model.split("/", 1)[0].strip().lower()
        if maybe_prefix in _KNOWN_MODEL_PREFIXES:
            return stripped_model

    provider_key = provider.strip().lower()
    if provider_key in _OPENAI_COMPATIBLE_PROVIDERS:
        return f"openai/{stripped_model}"

    prefix = _PROVIDER_PREFIX_MAP.get(provider_key, provider_key)
    if not prefix:
        return stripped_model
    return f"{prefix}/{stripped_model}"


def _model_request_retry_delay_s(attempt: int) -> float:
    return choose_retry_delay(
        attempt=attempt,
        base_delay_s=_MODEL_REQUEST_RETRY_BASE_DELAY_S,
    )


def _status_code_from_exception(exc: Exception) -> int | None:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    if response is None:
        return None
    status_code = getattr(response, "status_code", None)
    if isinstance(status_code, int):
        return status_code
    return None


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


def _looks_like_timeout_error(error_text: str) -> bool:
    text = error_text.lower()
    return "timed out" in text or "timeout" in text or "deadline exceeded" in text


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
