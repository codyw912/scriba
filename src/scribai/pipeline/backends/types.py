"""Typed models and protocols for backend interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CompletionResult:
    """Normalized completion payload with telemetry metadata."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_s: float = 0.0
    requests: int = 1
    split_count: int = 0


@dataclass(frozen=True)
class ChunkingHints:
    """Model/provider hints used for dynamic chunk sizing."""

    context_length: int | None = None
    context_length_source: str | None = None
    reserve_tokens: int = 1024
    target_utilization: float = 0.45
    min_target_tokens: int = 600
    max_target_tokens: int = 20000
    overlap_ratio: float = 0.11
    min_overlap_tokens: int = 120
    fallback_target_tokens: int = 5000
    fallback_overlap_tokens: int = 400
    max_output_tokens_limit: int | None = None


class ChatModelClient(Protocol):
    """Transport-agnostic model client interface."""

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
        """Execute one chat-completion request."""


@dataclass(frozen=True)
class ModelEndpoint:
    """Resolved endpoint details for a logical model role."""

    role: str
    backend_name: str
    base_url: str
    inference_url: str
    model: str
    api_key: str
    adapter: str
    topology: str
    provider: str
    context_length: int | None = None
    context_length_source: str | None = None
    chunking_hints: ChunkingHints = field(default_factory=ChunkingHints)


@dataclass(frozen=True)
class ModelSession:
    """Resolved model session with endpoint metadata and chat client."""

    endpoint: ModelEndpoint
    client: ChatModelClient
