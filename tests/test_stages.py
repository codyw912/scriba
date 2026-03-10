"""Focused tests for stage handlers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scribai.pipeline.backends import (
    ChunkingHints,
    CompletionResult,
    ModelEndpoint,
    ModelSession,
    RateLimitError,
)
from scribai.pipeline.profile import StageConfig
from scribai.pipeline.stages import StageExecutionError, execute_stage


class _QueueChatClient:
    def __init__(self, responses: list[CompletionResult]) -> None:
        self._responses = list(responses)

    def complete(self, **_: object) -> CompletionResult:
        if not self._responses:
            raise AssertionError("No queued response")
        return self._responses.pop(0)


class _CaptureChatClient:
    def __init__(self) -> None:
        self.max_output_tokens: list[int | None] = []

    def complete(self, **kwargs: object) -> CompletionResult:
        raw = kwargs.get("max_output_tokens")
        token_limit = raw if isinstance(raw, int) or raw is None else None
        self.max_output_tokens.append(token_limit)
        return CompletionResult(text="# normalized\n")


class _RateLimitOnceClient:
    def __init__(self) -> None:
        self._raised = False

    def complete(self, **_: object) -> CompletionResult:
        if not self._raised:
            self._raised = True
            raise RateLimitError("rate limited", retry_after_s=0.01)
        return CompletionResult(
            text="# normalized\n", completion_tokens=20, latency_s=0.1
        )


def _prepare_run_dir(run_dir: Path) -> None:
    for name in ("raw", "chunks", "map", "reduce", "final", "logs"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    (run_dir / "final" / "sections").mkdir(parents=True, exist_ok=True)


def _write_chunks(run_dir: Path) -> None:
    chunks_dir = run_dir / "chunks"
    chunk_1 = "# Intro\n\nGET /v1/health\n"
    chunk_2 = "# Auth\n\nPOST /v1/login\n"
    (chunks_dir / "chunk-0001.md").write_text(chunk_1, encoding="utf-8")
    (chunks_dir / "chunk-0002.md").write_text(chunk_2, encoding="utf-8")

    manifest = {
        "chunk_count": 2,
        "chunks": [
            {"chunk_id": "chunk-0001", "file": "chunk-0001.md", "heading": "Intro"},
            {"chunk_id": "chunk-0002", "file": "chunk-0002.md", "heading": "Auth"},
        ],
    }
    (chunks_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _endpoint(
    *,
    role: str = "normalize_text",
    provider: str = "openrouter",
    context_length: int | None = None,
    context_length_source: str | None = None,
    chunking_hints: ChunkingHints | None = None,
) -> ModelEndpoint:
    return ModelEndpoint(
        role=role,
        backend_name="remote_text",
        base_url="http://127.0.0.1:8090",
        inference_url="http://127.0.0.1:8090/v1/chat/completions",
        model="qwen/qwen3.5-35b-a3b",
        api_key="",
        adapter="litellm",
        topology="remote",
        provider=provider,
        context_length=context_length,
        context_length_source=context_length_source,
        chunking_hints=chunking_hints or ChunkingHints(),
    )


def _session(
    *,
    endpoint: ModelEndpoint | None = None,
    responses: list[CompletionResult] | None = None,
) -> ModelSession:
    resolved = endpoint or _endpoint()
    client = _QueueChatClient(responses or [CompletionResult(text="# ok\n")])
    return ModelSession(endpoint=resolved, client=client)


def test_extract_clean_sectionize_markdown_input(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)

    input_path = tmp_path / "input.md"
    input_path.write_text(
        "# API\n\nGET /v1/health\n\nPage 1\n\n# API\n\nGET /v1/health\n",
        encoding="utf-8",
    )
    state = {"input_path": str(input_path)}

    extract_result = execute_stage(
        stage_name="extract",
        state=state,
        run_dir=run_dir,
        stage_config=StageConfig(),
        model_session=None,
    )
    assert extract_result["handler"] == "extract"
    assert (run_dir / "raw" / "extracted.md").exists()

    clean_result = execute_stage(
        stage_name="clean",
        state=state,
        run_dir=run_dir,
        stage_config=StageConfig(),
        model_session=None,
    )
    assert clean_result["handler"] == "clean"
    assert (run_dir / "raw" / "cleaned.md").exists()

    sectionize_result = execute_stage(
        stage_name="sectionize",
        state=state,
        run_dir=run_dir,
        stage_config=StageConfig(target_tokens=1000, overlap_tokens=100),
        model_session=None,
    )
    assert sectionize_result["handler"] == "sectionize"
    assert (run_dir / "chunks" / "manifest.json").exists()


def test_sectionize_prefers_explicit_tokens_over_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)

    source = run_dir / "raw" / "cleaned.md"
    source.write_text("# API\n" + "A" * 12000 + "\n", encoding="utf-8")
    state = {"input_path": str(source)}

    endpoint = _endpoint(
        context_length=32000,
        context_length_source="openrouter_models",
    )
    model_session = _session(endpoint=endpoint)

    result = execute_stage(
        stage_name="sectionize",
        state=state,
        run_dir=run_dir,
        stage_config=StageConfig(target_tokens=500, overlap_tokens=25),
        model_session=model_session,
    )
    manifest = json.loads(
        (run_dir / "chunks" / "manifest.json").read_text(encoding="utf-8")
    )

    assert result["target_tokens"] == 500
    assert result["overlap_tokens"] == 25
    assert result["target_tokens_source"] == "explicit"
    assert result["overlap_tokens_source"] == "explicit"
    assert manifest["target_tokens"] == 500
    assert manifest["target_tokens_source"] == "explicit"
    assert manifest["overlap_tokens"] == 25


def test_sectionize_uses_metadata_target_and_overlap_when_unset(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)

    source = run_dir / "raw" / "cleaned.md"
    source.write_text("# API\n" + "A" * 12000 + "\n", encoding="utf-8")
    state = {"input_path": str(source)}

    endpoint = _endpoint(
        context_length=20000,
        context_length_source="openrouter_models",
    )
    model_session = _session(endpoint=endpoint)

    result = execute_stage(
        stage_name="sectionize",
        state=state,
        run_dir=run_dir,
        stage_config=StageConfig(),
        model_session=model_session,
    )
    manifest = json.loads(
        (run_dir / "chunks" / "manifest.json").read_text(encoding="utf-8")
    )

    assert result["target_tokens"] == 8539
    assert result["overlap_tokens"] == 939
    assert result["target_tokens_source"] == "metadata"
    assert result["overlap_tokens_source"] == "metadata"
    assert manifest["target_tokens_source"] == "metadata"
    assert manifest["sectionize_context_length"] == 20000


def test_sectionize_caps_high_context_target_tokens(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)

    source = run_dir / "raw" / "cleaned.md"
    source.write_text("# API\n" + "A" * 12000 + "\n", encoding="utf-8")
    state = {"input_path": str(source)}

    endpoint = _endpoint(
        context_length=65000,
        context_length_source="cerebras_model_catalog",
    )
    model_session = _session(endpoint=endpoint)

    result = execute_stage(
        stage_name="sectionize",
        state=state,
        run_dir=run_dir,
        stage_config=StageConfig(),
        model_session=model_session,
    )

    assert result["target_tokens"] == 20000
    assert result["target_tokens_source"] == "metadata"


def test_sectionize_merges_small_adjacent_heading_chunks(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)

    source = run_dir / "raw" / "cleaned.md"
    source.write_text(
        "\n".join(
            [
                "# Overview",
                "Tiny section.",
                "",
                "## Auth",
                "Tiny section.",
                "",
                "## Errors",
                "Tiny section.",
                "",
                "## Limits",
                "Tiny section.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    state = {"input_path": str(source)}

    result = execute_stage(
        stage_name="sectionize",
        state=state,
        run_dir=run_dir,
        stage_config=StageConfig(target_tokens=500, overlap_tokens=0),
        model_session=None,
    )
    manifest = json.loads(
        (run_dir / "chunks" / "manifest.json").read_text(encoding="utf-8")
    )

    assert result["initial_chunk_count"] >= 4
    assert result["chunk_count"] == 1
    assert manifest["initial_chunk_count"] == result["initial_chunk_count"]
    assert manifest["chunk_count"] == 1


def test_sectionize_preserves_preface_before_first_heading(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)

    source = run_dir / "raw" / "cleaned.md"
    source.write_text(
        "\n".join(
            [
                "Talos API Reference",
                "Generated on 2026-03-04",
                "",
                "# Introduction",
                "This is the intro.",
            ]
        ),
        encoding="utf-8",
    )

    execute_stage(
        stage_name="sectionize",
        state={"input_path": str(source)},
        run_dir=run_dir,
        stage_config=StageConfig(target_tokens=500, overlap_tokens=0),
        model_session=None,
    )

    first_chunk = (run_dir / "chunks" / "chunk-0001.md").read_text("utf-8")
    assert "Talos API Reference" in first_chunk
    assert "Generated on 2026-03-04" in first_chunk


def test_sectionize_falls_back_to_static_defaults_when_no_metadata(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)

    source = run_dir / "raw" / "cleaned.md"
    source.write_text("# API\n" + "A" * 8000 + "\n", encoding="utf-8")
    state = {"input_path": str(source)}

    result = execute_stage(
        stage_name="sectionize",
        state=state,
        run_dir=run_dir,
        stage_config=StageConfig(),
        model_session=None,
    )
    manifest = json.loads(
        (run_dir / "chunks" / "manifest.json").read_text(encoding="utf-8")
    )

    assert result["target_tokens"] == 5000
    assert result["overlap_tokens"] == 400
    assert result["target_tokens_source"] == "fallback"
    assert result["overlap_tokens_source"] == "fallback"
    assert manifest["sectionize_context_length"] is None


def test_extract_pdf_uses_ocr_vision_endpoint_when_available(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)

    input_path = tmp_path / "input.pdf"
    input_path.write_bytes(b"%PDF-1.4\n")
    state = {"input_path": str(input_path)}

    endpoint = _endpoint(role="ocr_vision", provider="glm_ocr")
    model_session = _session(endpoint=endpoint)

    with patch(
        "scribai.pipeline.stages._extract_pdf_markdown_with_vision_endpoint",
        return_value="# OCR\n\nExtracted text\n",
    ) as vision_mock:
        result = execute_stage(
            stage_name="extract",
            state=state,
            run_dir=run_dir,
            stage_config=StageConfig(),
            model_session=model_session,
        )

    vision_mock.assert_called_once()
    assert result["extraction_mode"] == "ocr_vision"
    metadata = json.loads(
        (run_dir / "raw" / "extract_metadata.json").read_text("utf-8")
    )
    assert metadata["extraction_mode"] == "ocr_vision"
    assert metadata["ocr_endpoint"]["provider"] == "glm_ocr"


def test_extract_pdf_falls_back_when_ocr_vision_errors(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)

    input_path = tmp_path / "input.pdf"
    input_path.write_bytes(b"%PDF-1.4\n")
    state = {"input_path": str(input_path)}

    model_session = _session(endpoint=_endpoint(role="ocr_vision"))

    with (
        patch(
            "scribai.pipeline.stages._extract_pdf_markdown_with_vision_endpoint",
            side_effect=StageExecutionError("ocr unavailable"),
        ),
        patch(
            "scribai.pipeline.stages._extract_pdf_markdown",
            return_value="# Fallback\n\ntext\n",
        ),
    ):
        result = execute_stage(
            stage_name="extract",
            state=state,
            run_dir=run_dir,
            stage_config=StageConfig(),
            model_session=model_session,
        )

    assert result["extraction_mode"] == "pymupdf4llm_fallback"
    assert "warning" in result


def test_normalize_map_passthrough_writes_telemetry_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)
    _write_chunks(run_dir)

    result = execute_stage(
        stage_name="normalize_map",
        state={},
        run_dir=run_dir,
        stage_config=StageConfig(temperature=0.0),
        model_session=None,
    )

    assert result["handler"] == "normalize_map"
    assert result["processed"] == 2

    map_manifest = json.loads((run_dir / "map" / "manifest.json").read_text("utf-8"))
    assert map_manifest["processed"] == 2
    assert map_manifest["model"] == "passthrough"
    assert map_manifest["processed_telemetry"]["requests"] == 2


def test_normalize_map_with_session_uses_completion_usage(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)
    _write_chunks(run_dir)

    model_session = _session(
        responses=[
            CompletionResult(
                text="# Intro\nNormalized one\n",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                latency_s=2.0,
                requests=1,
                split_count=0,
            ),
            CompletionResult(
                text="# Auth\nNormalized two\n",
                prompt_tokens=120,
                completion_tokens=60,
                total_tokens=180,
                latency_s=3.0,
                requests=1,
                split_count=0,
            ),
        ]
    )

    result = execute_stage(
        stage_name="normalize_map",
        state={},
        run_dir=run_dir,
        stage_config=StageConfig(temperature=0.0),
        model_session=model_session,
    )

    assert result["processed"] == 2
    assert result["effective_tokens_per_second"] == 22.0

    map_manifest = json.loads((run_dir / "map" / "manifest.json").read_text("utf-8"))
    processed = map_manifest["processed_telemetry"]
    assert processed["prompt_tokens"] == 220
    assert processed["completion_tokens"] == 110
    assert processed["total_tokens"] == 330
    assert processed["chunks_with_usage"] == 2
    assert processed["effective_tokens_per_second"] == 22.0


def test_normalize_map_scales_chunk_max_output_tokens(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)
    _write_chunks(run_dir)

    capture_client = _CaptureChatClient()
    model_session = ModelSession(endpoint=_endpoint(), client=capture_client)

    execute_stage(
        stage_name="normalize_map",
        state={},
        run_dir=run_dir,
        stage_config=StageConfig(max_output_tokens=1024),
        model_session=model_session,
    )

    assert capture_client.max_output_tokens
    assert all(
        value is not None and value <= 1024
        for value in capture_client.max_output_tokens
    )
    assert any(
        value is not None and value < 1024 for value in capture_client.max_output_tokens
    )


def test_normalize_map_retries_rate_limit_with_shared_gate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)
    _write_chunks(run_dir)

    model_session = ModelSession(endpoint=_endpoint(), client=_RateLimitOnceClient())

    result = execute_stage(
        stage_name="normalize_map",
        state={},
        run_dir=run_dir,
        stage_config=StageConfig(workers=2),
        model_session=model_session,
    )

    assert result["processed"] == 2
    assert result["workers_used"] == 2


def test_normalize_map_respects_provider_output_cap_without_stage_cap(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)
    _write_chunks(run_dir)

    capture_client = _CaptureChatClient()
    endpoint = _endpoint(
        chunking_hints=ChunkingHints(max_output_tokens_limit=300),
    )
    model_session = ModelSession(endpoint=endpoint, client=capture_client)

    execute_stage(
        stage_name="normalize_map",
        state={},
        run_dir=run_dir,
        stage_config=StageConfig(),
        model_session=model_session,
    )

    assert capture_client.max_output_tokens
    assert all(
        value is not None and value <= 300 for value in capture_client.max_output_tokens
    )


def test_reduce_validate_export_happy_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)
    _write_chunks(run_dir)

    execute_stage(
        stage_name="normalize_map",
        state={},
        run_dir=run_dir,
        stage_config=StageConfig(),
        model_session=None,
    )

    reduce_result = execute_stage(
        stage_name="reduce",
        state={},
        run_dir=run_dir,
        stage_config=StageConfig(),
        model_session=None,
    )
    assert reduce_result["handler"] == "reduce"

    validate_result = execute_stage(
        stage_name="validate",
        state={},
        run_dir=run_dir,
        stage_config=StageConfig(fail_on_hard_errors=True),
        model_session=None,
    )
    assert validate_result["ok"] == "true"

    export_result = execute_stage(
        stage_name="export",
        state={},
        run_dir=run_dir,
        stage_config=StageConfig(multi_file=True),
        model_session=None,
    )
    assert export_result["export_entry"] == "final/index.md"
    assert (run_dir / "final" / "index.md").exists()
    assert (run_dir / "final" / "merged.md").exists()


def test_validate_accepts_markdown_wrapped_endpoints(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)

    (run_dir / "raw" / "cleaned.md").write_text(
        "# Source\n\nGET /v1/health\n\nPOST /v1/echo\n",
        encoding="utf-8",
    )
    (run_dir / "final" / "merged.md").write_text(
        "# Output\n\n### GET `/v1/health`\n\n**POST** `/v1/echo`\n",
        encoding="utf-8",
    )

    validate_result = execute_stage(
        stage_name="validate",
        state={},
        run_dir=run_dir,
        stage_config=StageConfig(fail_on_hard_errors=True),
        model_session=None,
    )

    assert validate_result["ok"] == "true"
    report = json.loads(
        (run_dir / "final" / "validation_report.json").read_text("utf-8")
    )
    assert report["checks"]["missing_endpoint_count"] == 0


def test_validate_raises_on_think_leak_when_strict(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _prepare_run_dir(run_dir)
    (run_dir / "final" / "merged.md").write_text(
        "# Output\n\n<think>internal</think>\n",
        encoding="utf-8",
    )
    (run_dir / "raw" / "cleaned.md").write_text(
        "# Source\n\nGET /v1/health\n",
        encoding="utf-8",
    )
    (run_dir / "map" / "chunk-0001.json").write_text(
        json.dumps(
            {"chunk_id": "chunk-0001", "normalized_markdown": "<think>x</think>"}
        ),
        encoding="utf-8",
    )

    with pytest.raises(StageExecutionError):
        execute_stage(
            stage_name="validate",
            state={},
            run_dir=run_dir,
            stage_config=StageConfig(fail_on_hard_errors=True),
            model_session=None,
        )
