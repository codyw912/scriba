"""Backend adapter implementations."""

from scribai.pipeline.backends.adapters.base import BackendAdapter
from scribai.pipeline.backends.adapters.litellm_adapter import (
    AttachedOrRemoteLiteLLMBackendAdapter,
    LocalProcessLiteLLMBackendAdapter,
)

__all__ = [
    "AttachedOrRemoteLiteLLMBackendAdapter",
    "BackendAdapter",
    "LocalProcessLiteLLMBackendAdapter",
]
