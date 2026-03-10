"""Backend orchestration and transport clients for pipeline model roles."""

from scribai.pipeline.backends.errors import (
    BackendError,
    ContextWindowError,
    ModelClientError,
    ModelRequestTimeoutError,
    RateLimitError,
)
from scribai.pipeline.backends.manager import ModelManager
from scribai.pipeline.backends.types import (
    ChatModelClient,
    ChunkingHints,
    CompletionResult,
    ModelEndpoint,
    ModelSession,
)

__all__ = [
    "BackendError",
    "ChatModelClient",
    "ChunkingHints",
    "CompletionResult",
    "ContextWindowError",
    "ModelClientError",
    "ModelEndpoint",
    "ModelManager",
    "ModelRequestTimeoutError",
    "RateLimitError",
    "ModelSession",
]
