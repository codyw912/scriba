"""Base adapter interfaces."""

from __future__ import annotations

from scribai.pipeline.backends.types import (
    ChatModelClient,
    ChunkingHints,
    ModelEndpoint,
)
from scribai.pipeline.profile import BackendConfig


class BackendAdapter:
    """Abstract backend adapter."""

    def __init__(self, *, name: str, config: BackendConfig) -> None:
        self.name = name
        self.config = config

    def ensure_ready(self, *, model: str) -> None:
        """Ensure backend is running and healthy."""
        raise NotImplementedError

    def create_chat_client(self, *, endpoint: ModelEndpoint) -> ChatModelClient:
        """Create transport client for resolved endpoint."""
        raise NotImplementedError

    def model_chunking_hints(self, *, model: str) -> ChunkingHints:
        """Return model-aware chunking hints for this adapter/backend.

        First-class adapters should override this and provide model metadata
        through provider APIs/catalogs when possible.
        """
        del model
        return ChunkingHints()

    def stop(self) -> None:
        """Stop backend resources (no-op for attached/remote backends)."""
