"""Role-based backend orchestration manager."""

from __future__ import annotations

from scribai.pipeline.backends.adapters import (
    AttachedOrRemoteLiteLLMBackendAdapter,
    BackendAdapter,
    LocalProcessLiteLLMBackendAdapter,
)
from scribai.pipeline.backends.errors import BackendError
from scribai.pipeline.backends.types import ChunkingHints, ModelEndpoint, ModelSession
from scribai.pipeline.profile import BackendConfig, PipelineProfile, RoleBinding


class ModelManager:
    """Role-based model backend orchestration."""

    def __init__(self, profile: PipelineProfile) -> None:
        self.profile = profile
        self._adapters: dict[str, BackendAdapter] = {}
        self._active_backend_name: str | None = None
        self._model_chunking_cache: dict[tuple[str, str], ChunkingHints] = {}

    def __enter__(self) -> "ModelManager":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    def acquire(self, role_name: str) -> ModelSession:
        if role_name not in self.profile.roles:
            raise BackendError(f"Role '{role_name}' is not defined in profile")

        binding: RoleBinding = self.profile.roles[role_name]
        backend_name = binding.backend
        adapter = self._get_adapter(backend_name)

        if self._active_backend_name and self._active_backend_name != backend_name:
            current = self._adapters[self._active_backend_name]
            current.stop()
            self._active_backend_name = None

        adapter.ensure_ready(model=binding.model)
        self._active_backend_name = backend_name

        config = self.profile.backends[backend_name]
        chunking_hints = self._get_model_chunking_hints(
            config,
            adapter,
            binding.model,
        )

        endpoint = ModelEndpoint(
            role=role_name,
            backend_name=backend_name,
            base_url=config.base_url,
            inference_url=f"{config.base_url.rstrip('/')}{config.inference_path}",
            model=binding.model,
            api_key=config.api_key,
            adapter=config.adapter,
            topology=config.topology,
            provider=config.provider,
            context_length=chunking_hints.context_length,
            context_length_source=chunking_hints.context_length_source,
            chunking_hints=chunking_hints,
        )
        client = adapter.create_chat_client(endpoint=endpoint)
        return ModelSession(endpoint=endpoint, client=client)

    def close(self) -> None:
        for adapter in self._adapters.values():
            adapter.stop()
        self._active_backend_name = None

    def _get_adapter(self, backend_name: str) -> BackendAdapter:
        if backend_name in self._adapters:
            return self._adapters[backend_name]

        if backend_name not in self.profile.backends:
            raise BackendError(f"Backend '{backend_name}' is not defined")

        config = self.profile.backends[backend_name]
        if config.adapter == "litellm" and config.topology == "local_spawned":
            adapter: BackendAdapter = LocalProcessLiteLLMBackendAdapter(
                name=backend_name,
                config=config,
            )
        elif config.adapter == "litellm" and config.topology in {
            "local_attached",
            "remote",
        }:
            adapter = AttachedOrRemoteLiteLLMBackendAdapter(
                name=backend_name,
                config=config,
            )
        else:
            raise BackendError(
                "Backend '{}' has unsupported adapter/topology combination: {}/{}".format(
                    backend_name,
                    config.adapter,
                    config.topology,
                )
            )

        self._adapters[backend_name] = adapter
        return adapter

    def _get_model_chunking_hints(
        self,
        config: BackendConfig,
        adapter: BackendAdapter,
        model: str,
    ) -> ChunkingHints:
        cache_key = (config.provider, model)
        if cache_key in self._model_chunking_cache:
            return self._model_chunking_cache[cache_key]

        hints = adapter.model_chunking_hints(model=model)
        self._model_chunking_cache[cache_key] = hints
        return hints
