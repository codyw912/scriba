"""Tests for backend metadata resolution."""

from __future__ import annotations

from unittest.mock import patch

from scriba.pipeline.backends.adapters.litellm_adapter import (
    AttachedOrRemoteLiteLLMBackendAdapter,
)
from scriba.pipeline.backends.metadata_cerebras import (
    lookup_context_length_from_cerebras,
    lookup_max_output_tokens_from_cerebras,
)
from scriba.pipeline.profile import BackendConfig


def test_lookup_context_length_from_cerebras_defaults_to_free_tier(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SCRIBA_CEREBRAS_TIER", raising=False)
    value = lookup_context_length_from_cerebras(
        model="gpt-oss-120b",
        provider="cerebras",
    )
    assert value == 65000


def test_lookup_context_length_from_cerebras_paygo_tier(monkeypatch) -> None:
    monkeypatch.setenv("SCRIBA_CEREBRAS_TIER", "paygo")
    value = lookup_context_length_from_cerebras(
        model="llama3.1-8b",
        provider="cerebras",
    )
    assert value == 32000


def test_lookup_context_length_from_cerebras_ignores_other_provider() -> None:
    value = lookup_context_length_from_cerebras(
        model="gpt-oss-120b",
        provider="openrouter",
    )
    assert value is None


def test_lookup_max_output_tokens_from_cerebras_by_tier(monkeypatch) -> None:
    monkeypatch.setenv("SCRIBA_CEREBRAS_TIER", "free")
    free_value = lookup_max_output_tokens_from_cerebras(
        model="gpt-oss-120b",
        provider="cerebras",
    )
    monkeypatch.setenv("SCRIBA_CEREBRAS_TIER", "paygo")
    paygo_value = lookup_max_output_tokens_from_cerebras(
        model="gpt-oss-120b",
        provider="cerebras",
    )

    assert free_value == 32000
    assert paygo_value == 40000


def test_litellm_adapter_exposes_chunking_hints_from_openrouter_lookup() -> None:
    adapter = AttachedOrRemoteLiteLLMBackendAdapter(
        name="remote_text",
        config=BackendConfig(
            adapter="litellm",
            topology="remote",
            provider="openrouter",
            base_url="https://openrouter.ai/api/v1",
        ),
    )

    with patch(
        "scriba.pipeline.backends.adapters.litellm_adapter.lookup_context_length_from_openrouter",
        return_value=128000,
    ):
        hints = adapter.model_chunking_hints(model="qwen/qwen3.5-35b-a3b")

    assert hints.context_length == 128000
    assert hints.context_length_source == "openrouter_models"


def test_litellm_adapter_exposes_cerebras_chunking_hints_from_catalog(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SCRIBA_CEREBRAS_TIER", raising=False)

    adapter = AttachedOrRemoteLiteLLMBackendAdapter(
        name="remote_text",
        config=BackendConfig(
            adapter="litellm",
            topology="remote",
            provider="cerebras",
            base_url="https://api.cerebras.ai",
            api_key="token",
        ),
    )
    hints = adapter.model_chunking_hints(model="gpt-oss-120b")

    assert hints.context_length == 65000
    assert hints.context_length_source == "cerebras_model_catalog"
    assert hints.max_output_tokens_limit == 32000
