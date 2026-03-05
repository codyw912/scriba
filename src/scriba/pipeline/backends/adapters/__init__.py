"""Backend adapter implementations."""

from scriba.pipeline.backends.adapters.cerebras_sdk import (
    CerebrasSDKBackendAdapter,
)
from scriba.pipeline.backends.adapters.openai_http import (
    AttachedOrRemoteOpenAIBackendAdapter,
    LocalProcessOpenAIBackendAdapter,
)
from scriba.pipeline.backends.adapters.base import BackendAdapter

__all__ = [
    "AttachedOrRemoteOpenAIBackendAdapter",
    "BackendAdapter",
    "CerebrasSDKBackendAdapter",
    "LocalProcessOpenAIBackendAdapter",
]
