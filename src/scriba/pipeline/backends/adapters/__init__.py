"""Backend adapter implementations."""

from scriba.pipeline.backends.adapters.base import BackendAdapter
from scriba.pipeline.backends.adapters.litellm_adapter import (
    AttachedOrRemoteLiteLLMBackendAdapter,
    LocalProcessLiteLLMBackendAdapter,
)

__all__ = [
    "AttachedOrRemoteLiteLLMBackendAdapter",
    "BackendAdapter",
    "LocalProcessLiteLLMBackendAdapter",
]
