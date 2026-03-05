"""OpenAI-compatible HTTP backend adapters and client."""

from __future__ import annotations

import shlex
import subprocess
import time
from typing import Any
import os
import sys

import httpx

from scriba.pipeline.backends.adapters.base import BackendAdapter
from scriba.pipeline.backends.errors import (
    BackendError,
    ContextWindowError,
    ModelClientError,
    ModelRequestTimeoutError,
    RateLimitError,
)
from scriba.pipeline.backends.rate_limit import choose_retry_delay
from scriba.pipeline.backends.metadata_openrouter import (
    lookup_context_length_from_openrouter,
)
from scriba.pipeline.backends.response_parsing import (
    coerce_usage_int,
    extract_completion_text,
    parse_json_response_payload,
    sanitize_model_markdown,
)
from scriba.pipeline.backends.types import (
    ChunkingHints,
    CompletionResult,
    ModelEndpoint,
)
from scriba.pipeline.profile import BackendConfig


DEFAULT_HEALTH_POST_PAYLOAD = {
    "messages": [{"role": "user", "content": "ping"}],
    "max_tokens": 1,
}

_MAX_MODEL_REQUEST_ATTEMPTS = 3
_RETRYABLE_HTTP_STATUS_CODES = {429, 500, 502, 503, 504}
_MODEL_REQUEST_RETRY_BASE_DELAY_S = 0.75


class OpenAIHTTPChatClient:
    """OpenAI-compatible chat completion client."""

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
        headers = {"Content-Type": "application/json"}
        if self.endpoint.api_key:
            headers["Authorization"] = f"Bearer {self.endpoint.api_key}"

        payload: dict[str, Any] = {
            "model": self.endpoint.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens
        if reasoning_effort is not None or reasoning_exclude is not None:
            reasoning_payload: dict[str, Any] = {}
            if reasoning_effort is not None:
                reasoning_payload["effort"] = reasoning_effort
            if reasoning_exclude is not None:
                reasoning_payload["exclude"] = reasoning_exclude
            if reasoning_payload:
                payload["reasoning"] = reasoning_payload

        started = time.perf_counter()
        data: dict[str, Any] | None = None
        for attempt in range(1, _MAX_MODEL_REQUEST_ATTEMPTS + 1):
            try:
                with httpx.Client(timeout=float(request_timeout_s)) as client:
                    response = client.post(
                        self.endpoint.inference_url,
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = parse_json_response_payload(
                        status_code=response.status_code,
                        body_text=response.text,
                    )
                break
            except httpx.TimeoutException as exc:
                if attempt < _MAX_MODEL_REQUEST_ATTEMPTS:
                    time.sleep(_model_request_retry_delay_s(attempt))
                    continue
                raise ModelRequestTimeoutError(
                    f"Model request timed out after {request_timeout_s}s"
                ) from exc
            except httpx.HTTPStatusError as exc:
                response_text = exc.response.text if exc.response is not None else ""
                if (
                    exc.response is not None
                    and exc.response.status_code == 400
                    and _looks_like_context_error(response_text)
                ):
                    raise ContextWindowError(response_text) from exc

                status_code = (
                    exc.response.status_code if exc.response is not None else None
                )
                if (
                    status_code in _RETRYABLE_HTTP_STATUS_CODES
                    and attempt < _MAX_MODEL_REQUEST_ATTEMPTS
                ):
                    retry_delay_s = choose_retry_delay(
                        attempt=attempt,
                        base_delay_s=_MODEL_REQUEST_RETRY_BASE_DELAY_S,
                        headers=exc.response.headers
                        if exc.response is not None
                        else None,
                        error_text=response_text,
                    )
                    time.sleep(retry_delay_s)
                    continue

                if status_code == 429:
                    retry_delay_s = choose_retry_delay(
                        attempt=attempt,
                        base_delay_s=_MODEL_REQUEST_RETRY_BASE_DELAY_S,
                        headers=exc.response.headers
                        if exc.response is not None
                        else None,
                        error_text=response_text,
                    )
                    raise RateLimitError(
                        f"HTTP 429: {response_text or str(exc)}",
                        retry_after_s=retry_delay_s,
                    ) from exc

                raise ModelClientError(
                    f"HTTP {status_code or 'unknown'}: {response_text or str(exc)}"
                ) from exc
            except ModelClientError:
                raise
            except Exception as exc:
                raise ModelClientError(str(exc)) from exc

        if data is None:
            raise ModelClientError("Model request failed without response payload")

        elapsed_s = time.perf_counter() - started
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


class LocalProcessOpenAIBackendAdapter(BackendAdapter):
    """Local process backend started via command."""

    def __init__(self, *, name: str, config: BackendConfig) -> None:
        super().__init__(name=name, config=config)
        self._process: subprocess.Popen[bytes] | None = None

    def ensure_ready(self, *, model: str) -> None:
        self._ensure_process_running()
        self._wait_until_healthy(model=model)

    def create_chat_client(self, *, endpoint: ModelEndpoint) -> OpenAIHTTPChatClient:
        return OpenAIHTTPChatClient(endpoint)

    def model_chunking_hints(self, *, model: str) -> ChunkingHints:
        context_length = lookup_context_length_from_openrouter(
            model=model,
            provider=self.config.provider,
        )
        return ChunkingHints(
            context_length=context_length,
            context_length_source="openrouter_models" if context_length else None,
        )

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


class AttachedOrRemoteOpenAIBackendAdapter(BackendAdapter):
    """Attached or remote OpenAI-compatible backend adapter."""

    def ensure_ready(self, *, model: str) -> None:
        ok, error = _probe_health(config=self.config, model=model)
        if not ok:
            raise BackendError(f"Backend '{self.name}' is not healthy ({error})")

    def create_chat_client(self, *, endpoint: ModelEndpoint) -> OpenAIHTTPChatClient:
        return OpenAIHTTPChatClient(endpoint)

    def model_chunking_hints(self, *, model: str) -> ChunkingHints:
        context_length = lookup_context_length_from_openrouter(
            model=model,
            provider=self.config.provider,
        )
        return ChunkingHints(
            context_length=context_length,
            context_length_source="openrouter_models" if context_length else None,
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


def _model_request_retry_delay_s(attempt: int) -> float:
    return choose_retry_delay(
        attempt=attempt,
        base_delay_s=_MODEL_REQUEST_RETRY_BASE_DELAY_S,
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
