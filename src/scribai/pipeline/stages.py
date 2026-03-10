"""Stage handlers for extract, clean, sectionize, map, reduce, validate, export."""

from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from tqdm.auto import tqdm

from scribai.pipeline.backends import (
    ChunkingHints,
    CompletionResult,
    ContextWindowError,
    ModelClientError,
    ModelEndpoint,
    ModelRequestTimeoutError,
    ModelSession,
    RateLimitError,
)
from scribai.pipeline.profile import StageConfig
from scribai.pipeline.rate_limit_gate import SharedRateLimitGate
from scribai.pipeline.state import utc_now_iso
from scribai.token_count import estimate_token_count, estimated_chars_for_tokens

MAP_PROMPT_VERSION = "v1"


class StageExecutionError(RuntimeError):
    """Raised when a stage cannot complete."""


def execute_stage(
    *,
    stage_name: str,
    state: dict[str, Any],
    run_dir: Path,
    stage_config: StageConfig,
    model_session: ModelSession | None,
) -> dict[str, Any]:
    """Execute one pipeline stage and return metadata for logs/state."""
    if stage_name == "extract":
        return _run_extract_stage(
            state=state,
            run_dir=run_dir,
            stage_config=stage_config,
            model_session=model_session,
        )
    if stage_name == "clean":
        return _run_clean_stage(run_dir=run_dir)
    if stage_name == "sectionize":
        return _run_sectionize_stage(
            run_dir=run_dir,
            stage_config=stage_config,
            model_session=model_session,
        )
    if stage_name == "normalize_map":
        return _run_normalize_map_stage(
            run_dir=run_dir,
            stage_config=stage_config,
            model_session=model_session,
        )
    if stage_name == "reduce":
        return _run_reduce_stage(run_dir=run_dir, stage_config=stage_config)
    if stage_name == "validate":
        return _run_validate_stage(run_dir=run_dir, stage_config=stage_config)
    if stage_name == "export":
        return _run_export_stage(run_dir=run_dir, stage_config=stage_config)

    raise StageExecutionError(f"Unsupported stage: {stage_name}")


def _run_extract_stage(
    *,
    state: dict[str, Any],
    run_dir: Path,
    stage_config: StageConfig,
    model_session: ModelSession | None,
) -> dict[str, Any]:
    input_path = Path(state["input_path"]).expanduser().resolve()
    if not input_path.exists() or not input_path.is_file():
        raise StageExecutionError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    text: str
    extraction_mode = "direct_text"
    extraction_warning: str | None = None

    if suffix in {".md", ".markdown", ".txt"}:
        text = input_path.read_text(encoding="utf-8")
        extraction_mode = "direct_text"
    elif suffix == ".pdf":
        endpoint = model_session.endpoint if model_session is not None else None
        if endpoint is not None and endpoint.role == "ocr_vision":
            try:
                text = _extract_pdf_markdown_with_vision_endpoint(
                    input_path=input_path,
                    model_session=model_session,
                    request_timeout_s=(stage_config.request_timeout_s or 120),
                    max_output_tokens=(stage_config.max_output_tokens or 4096),
                )
                extraction_mode = "ocr_vision"
            except StageExecutionError as exc:
                extraction_warning = str(exc)
                text = _extract_pdf_markdown(input_path)
                extraction_mode = "pymupdf4llm_fallback"
        else:
            text = _extract_pdf_markdown(input_path)
            extraction_mode = "pymupdf4llm"
    else:
        raise StageExecutionError(
            f"Unsupported input type '{suffix or 'unknown'}' for extract stage"
        )

    markdown_path = run_dir / "raw" / "extracted.md"
    metadata_path = run_dir / "raw" / "extract_metadata.json"

    _write_text_if_changed(markdown_path, _ensure_trailing_newline(text))
    metadata = {
        "input_path": str(input_path),
        "source_type": suffix.lstrip(".") or "unknown",
        "extraction_mode": extraction_mode,
        "markdown_chars": len(text),
        "generated_at": utc_now_iso(),
    }
    endpoint = model_session.endpoint if model_session is not None else None
    if endpoint is not None and extraction_mode == "ocr_vision":
        metadata["ocr_endpoint"] = {
            "role": endpoint.role,
            "backend": endpoint.backend_name,
            "adapter": endpoint.adapter,
            "topology": endpoint.topology,
            "provider": endpoint.provider,
            "model": endpoint.model,
        }
    if extraction_warning:
        metadata["warning"] = extraction_warning
    _write_json_if_changed(metadata_path, metadata)

    result = {
        "handler": "extract",
        "source": str(input_path),
        "output": str(markdown_path.relative_to(run_dir)),
        "markdown_chars": len(text),
        "extraction_mode": extraction_mode,
    }
    if extraction_warning:
        result["warning"] = extraction_warning
    return result


def _extract_pdf_markdown(input_path: Path) -> str:
    try:
        import pymupdf4llm
    except ImportError as exc:
        raise StageExecutionError(
            "PDF extraction requires pymupdf4llm. Install the dependency or use markdown/text input."
        ) from exc

    try:
        output = pymupdf4llm.to_markdown(str(input_path))
    except Exception as exc:  # pragma: no cover - external parser errors
        raise StageExecutionError(f"Failed to extract PDF markdown: {exc}") from exc
    return str(output)


def _extract_pdf_markdown_with_vision_endpoint(
    *,
    input_path: Path,
    model_session: ModelSession | None,
    request_timeout_s: int,
    max_output_tokens: int,
) -> str:
    if model_session is None:
        raise StageExecutionError("Vision OCR endpoint session is missing")

    try:
        try:
            import pymupdf  # type: ignore[import-not-found]
        except ImportError:
            import fitz as pymupdf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise StageExecutionError(
            "Vision OCR extraction requires pymupdf (PyMuPDF) dependency"
        ) from exc

    try:
        document = pymupdf.open(str(input_path))
    except Exception as exc:  # pragma: no cover
        raise StageExecutionError(
            f"Failed to open PDF for OCR extraction: {exc}"
        ) from exc

    if len(document) == 0:
        document.close()
        return ""

    blocks: list[str] = []
    for page_index in range(len(document)):
        page = document[page_index]
        try:
            pix = page.get_pixmap(alpha=False)
            png_bytes = pix.tobytes("png")
        except Exception as exc:  # pragma: no cover
            document.close()
            raise StageExecutionError(
                f"Failed to render PDF page {page_index + 1} for OCR extraction: {exc}"
            ) from exc

        messages = [
            {
                "role": "system",
                "content": (
                    "You extract markdown from document page images for technical documentation. "
                    "Preserve text, headings, lists, and code blocks faithfully. "
                    "Return markdown only."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract this PDF page to markdown. "
                            "Do not include commentary or reasoning tags."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_data_url(png_bytes),
                        },
                    },
                ],
            },
        ]

        try:
            completion = model_session.client.complete(
                messages=messages,
                temperature=0.0,
                request_timeout_s=request_timeout_s,
                max_output_tokens=max_output_tokens,
            )
        except Exception as exc:
            document.close()
            raise StageExecutionError(
                f"Vision OCR request failed on page {page_index + 1}: {exc}"
            ) from exc

        page_text = completion.text.strip()
        if page_text:
            blocks.append(f"<!-- page {page_index + 1} -->\n\n{page_text}")

    document.close()
    return _ensure_trailing_newline("\n\n".join(blocks).strip())


def _image_data_url(png_bytes: bytes) -> str:
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _run_clean_stage(*, run_dir: Path) -> dict[str, Any]:
    extracted_path = run_dir / "raw" / "extracted.md"
    if not extracted_path.exists():
        raise StageExecutionError("Clean stage requires raw/extracted.md")

    original = extracted_path.read_text(encoding="utf-8")
    cleaned, report = _clean_markdown(original)

    cleaned_path = run_dir / "raw" / "cleaned.md"
    report_path = run_dir / "raw" / "clean_report.json"
    _write_text_if_changed(cleaned_path, cleaned)
    _write_json_if_changed(report_path, report)

    return {
        "handler": "clean",
        "input": str(extracted_path.relative_to(run_dir)),
        "output": str(cleaned_path.relative_to(run_dir)),
        "removed_repeated_lines": report["removed_repeated_lines"],
        "removed_page_numbers": report["removed_page_numbers"],
    }


def _run_sectionize_stage(
    *, run_dir: Path, stage_config: StageConfig, model_session: ModelSession | None
) -> dict[str, Any]:
    source_path = run_dir / "raw" / "cleaned.md"
    if not source_path.exists():
        source_path = run_dir / "raw" / "extracted.md"
    if not source_path.exists():
        raise StageExecutionError(
            "Sectionize stage requires raw/cleaned.md or raw/extracted.md"
        )

    text = source_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    endpoint = model_session.endpoint if model_session is not None else None
    target_tokens, target_source = _resolve_sectionize_target_tokens(
        stage_config=stage_config,
        endpoint=endpoint,
    )
    overlap_tokens, overlap_source = _resolve_sectionize_overlap_tokens(
        stage_config=stage_config,
        endpoint=endpoint,
        target_tokens=target_tokens,
    )
    target_chars = max(1000, estimated_chars_for_tokens(target_tokens))
    overlap_lines = max(0, int(estimated_chars_for_tokens(overlap_tokens) / 80))

    chunk_ranges = _build_chunk_ranges(lines, target_chars, overlap_lines)
    initial_chunk_count = len(chunk_ranges)
    chunk_ranges = _merge_small_adjacent_chunk_ranges(
        lines=lines,
        chunk_ranges=chunk_ranges,
        target_chars=target_chars,
    )
    if not chunk_ranges:
        chunk_ranges = [(0, len(lines), "document")]

    manifest_chunks: list[dict[str, Any]] = []
    desired_files: set[str] = set()

    for index, (start_idx, end_idx, heading) in enumerate(chunk_ranges, start=1):
        chunk_text = _ensure_trailing_newline(
            "\n".join(lines[start_idx:end_idx]).strip()
        )
        checksum = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        file_name = f"chunk-{index:04d}.md"
        desired_files.add(file_name)
        chunk_path = chunks_dir / file_name
        _write_text_if_changed(chunk_path, chunk_text)

        manifest_chunks.append(
            {
                "chunk_id": f"chunk-{index:04d}",
                "file": file_name,
                "start_line": start_idx + 1,
                "end_line": end_idx,
                "line_count": max(0, end_idx - start_idx),
                "char_count": len(chunk_text),
                "sha256": checksum,
                "heading": heading,
            }
        )

    for existing in chunks_dir.glob("chunk-*.md"):
        if existing.name not in desired_files:
            existing.unlink(missing_ok=True)

    manifest = {
        "source_file": str(source_path.relative_to(run_dir)),
        "generated_at": utc_now_iso(),
        "target_tokens": target_tokens,
        "target_tokens_source": target_source,
        "overlap_tokens": overlap_tokens,
        "overlap_tokens_source": overlap_source,
        "sectionize_context_length": endpoint.context_length if endpoint else None,
        "initial_chunk_count": initial_chunk_count,
        "chunk_count": len(manifest_chunks),
        "chunks": manifest_chunks,
    }
    _write_json_if_changed(chunks_dir / "manifest.json", manifest)

    return {
        "handler": "sectionize",
        "source": str(source_path.relative_to(run_dir)),
        "initial_chunk_count": initial_chunk_count,
        "chunk_count": len(manifest_chunks),
        "target_tokens": target_tokens,
        "overlap_tokens": overlap_tokens,
        "target_tokens_source": target_source,
        "overlap_tokens_source": overlap_source,
    }


def _resolve_sectionize_target_tokens(
    *, stage_config: StageConfig, endpoint: ModelEndpoint | None
) -> tuple[int, str]:
    if stage_config.target_tokens is not None:
        return stage_config.target_tokens, "explicit"

    hints = endpoint.chunking_hints if endpoint is not None else ChunkingHints()
    hints = _hydrate_chunking_hints_from_endpoint(hints=hints, endpoint=endpoint)

    if hints.context_length:
        inferred = _infer_target_tokens_from_hints(hints)
        return inferred, "metadata"

    return hints.fallback_target_tokens, "fallback"


def _resolve_sectionize_overlap_tokens(
    *,
    stage_config: StageConfig,
    endpoint: ModelEndpoint | None,
    target_tokens: int,
) -> tuple[int, str]:
    if stage_config.overlap_tokens is not None:
        return stage_config.overlap_tokens, "explicit"

    hints = endpoint.chunking_hints if endpoint is not None else ChunkingHints()
    hints = _hydrate_chunking_hints_from_endpoint(hints=hints, endpoint=endpoint)

    if hints.context_length:
        return _infer_overlap_tokens_from_hints(target_tokens, hints), "metadata"

    return hints.fallback_overlap_tokens, "fallback"


def _infer_target_tokens_from_hints(hints: ChunkingHints) -> int:
    if not hints.context_length:
        return hints.fallback_target_tokens

    context_budget = max(0, hints.context_length - max(0, hints.reserve_tokens))
    inferred = int(context_budget * max(0.0, hints.target_utilization))

    if inferred < hints.min_target_tokens:
        inferred = hints.min_target_tokens
    if inferred > hints.max_target_tokens:
        inferred = hints.max_target_tokens
    if inferred > context_budget:
        inferred = context_budget
    return inferred


def _infer_overlap_tokens_from_hints(target_tokens: int, hints: ChunkingHints) -> int:
    inferred = int(target_tokens * max(0.0, hints.overlap_ratio))
    return max(hints.min_overlap_tokens, inferred)


def _hydrate_chunking_hints_from_endpoint(
    *,
    hints: ChunkingHints,
    endpoint: ModelEndpoint | None,
) -> ChunkingHints:
    if endpoint is None:
        return hints
    if hints.context_length is not None:
        return hints
    if endpoint.context_length is None:
        return hints
    return replace(
        hints,
        context_length=endpoint.context_length,
        context_length_source=endpoint.context_length_source,
    )


def _resolve_chunk_max_output_tokens(
    *,
    chunk_text: str,
    stage_max_output_tokens: int | None,
    model_name: str | None,
    hints: ChunkingHints | None,
) -> int | None:
    estimated_input_tokens = max(
        1,
        estimate_token_count(chunk_text, model=model_name).count,
    )
    suggested = max(96, int(round(estimated_input_tokens * 1.35)) + 64)

    provider_cap = hints.max_output_tokens_limit if hints is not None else None

    resolved = suggested
    if provider_cap is not None:
        resolved = min(resolved, provider_cap)
    if stage_max_output_tokens is not None:
        resolved = min(resolved, stage_max_output_tokens)
    return max(96, resolved)


def _merge_small_adjacent_chunk_ranges(
    *,
    lines: list[str],
    chunk_ranges: list[tuple[int, int, str]],
    target_chars: int,
) -> list[tuple[int, int, str]]:
    if len(chunk_ranges) < 2 or target_chars <= 0:
        return chunk_ranges

    merged: list[tuple[int, int, str]] = []
    current_start, current_end, current_heading = chunk_ranges[0]
    current_chars = _range_char_count(lines, current_start, current_end)

    for start, end, heading in chunk_ranges[1:]:
        next_chars = _range_char_count(lines, start, end)
        contiguous = current_end == start
        combined_chars = current_chars + next_chars
        current_underfilled = current_chars < int(target_chars * 0.6)
        next_underfilled = next_chars < int(target_chars * 0.45)

        if (
            contiguous
            and combined_chars <= target_chars
            and (current_underfilled or next_underfilled)
        ):
            current_end = end
            current_chars = combined_chars
            if heading != current_heading:
                current_heading = "multiple-sections"
            continue

        merged.append((current_start, current_end, current_heading))
        current_start, current_end, current_heading = start, end, heading
        current_chars = next_chars

    merged.append((current_start, current_end, current_heading))
    return merged


def _range_char_count(lines: list[str], start: int, end: int) -> int:
    if start >= end:
        return 0
    return sum(len(line) + 1 for line in lines[start:end])


def _run_normalize_map_stage(
    *,
    run_dir: Path,
    stage_config: StageConfig,
    model_session: ModelSession | None,
) -> dict[str, Any]:
    manifest_path = run_dir / "chunks" / "manifest.json"
    if not manifest_path.exists():
        raise StageExecutionError("normalize_map stage requires chunks/manifest.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunk_entries = manifest.get("chunks", [])
    if not isinstance(chunk_entries, list):
        raise StageExecutionError("chunks/manifest.json has invalid 'chunks' format")

    map_dir = run_dir / "map"
    map_dir.mkdir(parents=True, exist_ok=True)
    endpoint = model_session.endpoint if model_session is not None else None

    temperature = (
        stage_config.temperature if stage_config.temperature is not None else 0.0
    )
    request_timeout_s = (
        stage_config.request_timeout_s
        if stage_config.request_timeout_s is not None
        else 600
    )
    max_output_tokens = stage_config.max_output_tokens
    reasoning_effort = stage_config.reasoning_effort
    reasoning_exclude = stage_config.reasoning_exclude
    processed = 0
    skipped = 0

    processed_requests_total = 0
    processed_latency_s_total = 0.0
    processed_prompt_tokens_total = 0
    processed_completion_tokens_total = 0
    processed_total_tokens_total = 0
    processed_output_tokens_est_total = 0
    processed_chunks_with_usage = 0

    pending_tasks: list[dict[str, Any]] = []

    for chunk in chunk_entries:
        chunk_file = chunk.get("file", "")
        chunk_id = chunk.get("chunk_id", "")
        if not chunk_file or not chunk_id:
            raise StageExecutionError("Chunk entry missing file/chunk_id in manifest")

        chunk_path = run_dir / "chunks" / chunk_file
        if not chunk_path.exists():
            raise StageExecutionError(f"Chunk file missing: {chunk_path}")

        chunk_text = chunk_path.read_text(encoding="utf-8")
        chunk_max_output_tokens = _resolve_chunk_max_output_tokens(
            chunk_text=chunk_text,
            stage_max_output_tokens=max_output_tokens,
            model_name=endpoint.model if endpoint is not None else None,
            hints=endpoint.chunking_hints if endpoint is not None else None,
        )
        input_sha = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
        output_path = map_dir / f"{chunk_id}.json"

        output_model = endpoint.model if endpoint else "passthrough"
        if output_path.exists():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                existing.get("input_sha256") == input_sha
                and existing.get("model") == output_model
                and existing.get("prompt_version") == MAP_PROMPT_VERSION
            ):
                skipped += 1
                continue

        pending_tasks.append(
            {
                "chunk_id": chunk_id,
                "chunk_file": chunk_file,
                "heading": chunk.get("heading", ""),
                "chunk_text": chunk_text,
                "chunk_max_output_tokens": chunk_max_output_tokens,
                "input_sha": input_sha,
                "output_path": output_path,
                "output_model": output_model,
            }
        )

    workers = _resolve_map_workers(
        stage_config=stage_config,
        pending_count=len(pending_tasks),
        has_model_session=model_session is not None,
    )
    rate_limit_retries = _resolve_map_rate_limit_retries()
    rate_limit_gate = SharedRateLimitGate()

    def apply_result(result: dict[str, Any]) -> None:
        nonlocal processed
        nonlocal processed_requests_total
        nonlocal processed_latency_s_total
        nonlocal processed_prompt_tokens_total
        nonlocal processed_completion_tokens_total
        nonlocal processed_total_tokens_total
        nonlocal processed_output_tokens_est_total
        nonlocal processed_chunks_with_usage

        _write_json_if_changed(Path(result["output_path"]), result["payload"])
        chunk_telemetry = result["telemetry"]

        processed += 1
        processed_requests_total += int(chunk_telemetry["requests"])
        processed_latency_s_total += float(chunk_telemetry["latency_s"])
        processed_output_tokens_est_total += int(chunk_telemetry["output_tokens_est"])
        if chunk_telemetry["prompt_tokens"] is not None:
            processed_prompt_tokens_total += int(chunk_telemetry["prompt_tokens"])
            processed_chunks_with_usage += 1
        if chunk_telemetry["completion_tokens"] is not None:
            processed_completion_tokens_total += int(
                chunk_telemetry["completion_tokens"]
            )
        if chunk_telemetry["total_tokens"] is not None:
            processed_total_tokens_total += int(chunk_telemetry["total_tokens"])

    with _normalize_map_progress_bar(
        total=len(chunk_entries),
        initial=skipped,
        model=endpoint.model if endpoint else "passthrough",
        workers=workers,
    ) as progress_bar:
        if workers <= 1:
            for task in pending_tasks:
                result = _run_normalize_map_task(
                    task=task,
                    model_session=model_session,
                    temperature=temperature,
                    request_timeout_s=request_timeout_s,
                    reasoning_effort=reasoning_effort,
                    reasoning_exclude=reasoning_exclude,
                    rate_limit_gate=rate_limit_gate,
                    rate_limit_retries=rate_limit_retries,
                )
                apply_result(result)
                progress_bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _run_normalize_map_task,
                        task=task,
                        model_session=model_session,
                        temperature=temperature,
                        request_timeout_s=request_timeout_s,
                        reasoning_effort=reasoning_effort,
                        reasoning_exclude=reasoning_exclude,
                        rate_limit_gate=rate_limit_gate,
                        rate_limit_retries=rate_limit_retries,
                    )
                    for task in pending_tasks
                ]
                for future in as_completed(futures):
                    apply_result(future.result())
                    progress_bar.update(1)

    all_outputs_telemetry = _aggregate_map_telemetry(map_dir)

    map_manifest = {
        "generated_at": utc_now_iso(),
        "chunk_count": len(chunk_entries),
        "processed": processed,
        "skipped": skipped,
        "model": endpoint.model if endpoint else "passthrough",
        "prompt_version": MAP_PROMPT_VERSION,
        "processed_telemetry": {
            "requests": processed_requests_total,
            "latency_s": round(processed_latency_s_total, 3),
            "prompt_tokens": processed_prompt_tokens_total
            if processed_chunks_with_usage > 0
            else None,
            "completion_tokens": processed_completion_tokens_total
            if processed_chunks_with_usage > 0
            else None,
            "total_tokens": processed_total_tokens_total
            if processed_chunks_with_usage > 0
            else None,
            "output_tokens_est": processed_output_tokens_est_total,
            "chunks_with_usage": processed_chunks_with_usage,
            "effective_tokens_per_second": _effective_tokens_per_second(
                completion_tokens=processed_completion_tokens_total
                if processed_chunks_with_usage > 0
                else None,
                output_tokens_est=processed_output_tokens_est_total,
                latency_s=processed_latency_s_total,
            ),
        },
        "all_outputs_telemetry": all_outputs_telemetry,
        "workers_used": workers,
        "rate_limit_retries": rate_limit_retries,
    }
    _write_json_if_changed(map_dir / "manifest.json", map_manifest)

    return {
        "handler": "normalize_map",
        "chunk_count": len(chunk_entries),
        "processed": processed,
        "skipped": skipped,
        "model": endpoint.model if endpoint else "passthrough",
        "workers_used": workers,
        "effective_tokens_per_second": map_manifest["processed_telemetry"][
            "effective_tokens_per_second"
        ],
    }


def _run_normalize_map_task(
    *,
    task: dict[str, Any],
    model_session: ModelSession | None,
    temperature: float,
    request_timeout_s: int,
    reasoning_effort: str | None,
    reasoning_exclude: bool | None,
    rate_limit_gate: SharedRateLimitGate,
    rate_limit_retries: int,
) -> dict[str, Any]:
    attempts = 0
    while True:
        if model_session is not None:
            rate_limit_gate.wait_until_ready()
        try:
            if model_session is None:
                completion = CompletionResult(
                    text=str(task["chunk_text"]), latency_s=0.0
                )
            else:
                completion = _normalize_chunk_with_llm_with_context_fallback(
                    chunk_text=str(task["chunk_text"]),
                    heading=str(task["heading"]),
                    model_session=model_session,
                    temperature=temperature,
                    request_timeout_s=request_timeout_s,
                    max_output_tokens=int(task["chunk_max_output_tokens"])
                    if task["chunk_max_output_tokens"] is not None
                    else None,
                    reasoning_effort=reasoning_effort,
                    reasoning_exclude=reasoning_exclude,
                )
            break
        except RateLimitError as exc:
            attempts += 1
            retry_after_s = exc.retry_after_s if exc.retry_after_s is not None else 1.0
            rate_limit_gate.block_for(retry_after_s)
            if attempts > rate_limit_retries:
                raise StageExecutionError(str(exc)) from exc

    normalized_markdown = completion.text
    chunk_telemetry = _chunk_telemetry_from_completion(
        completion,
        normalized_markdown,
        model_name=str(task["output_model"]),
    )
    payload = {
        "chunk_id": task["chunk_id"],
        "source_file": task["chunk_file"],
        "heading": task["heading"],
        "input_sha256": task["input_sha"],
        "model": task["output_model"],
        "prompt_version": MAP_PROMPT_VERSION,
        "normalized_markdown": _ensure_trailing_newline(normalized_markdown.strip()),
        "generated_at": utc_now_iso(),
        "telemetry": chunk_telemetry,
    }

    return {
        "output_path": str(task["output_path"]),
        "payload": payload,
        "telemetry": chunk_telemetry,
    }


def _resolve_map_workers(
    *,
    stage_config: StageConfig,
    pending_count: int,
    has_model_session: bool,
) -> int:
    if pending_count <= 1 or not has_model_session:
        return 1
    requested = max(1, int(stage_config.workers))
    return min(requested, pending_count)


def _resolve_map_rate_limit_retries() -> int:
    raw = os.getenv("SCRIBAI_MAP_RATE_LIMIT_RETRIES", "2").strip()
    try:
        parsed = int(raw)
    except ValueError:
        return 2
    return max(0, parsed)


class _NoOpProgressBar:
    def __enter__(self) -> "_NoOpProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    def update(self, n: int = 1) -> None:
        del n


def _normalize_map_progress_bar(
    *,
    total: int,
    initial: int,
    model: str,
    workers: int,
) -> Any:
    if total <= 0:
        return _NoOpProgressBar()

    enabled_raw = os.getenv("SCRIBAI_PROGRESS", "1").strip().lower()
    if enabled_raw in {"0", "false", "no", "off"}:
        return _NoOpProgressBar()

    if not sys.stderr.isatty():
        return _NoOpProgressBar()

    label = model if len(model) <= 24 else f"...{model[-24:]}"
    progress = tqdm(
        total=total,
        initial=max(0, min(initial, total)),
        desc=f"normalize_map {label}",
        unit="chunk",
        dynamic_ncols=True,
        mininterval=0.1,
        leave=True,
    )
    if workers > 1:
        progress.set_postfix_str(f"workers={workers}")
    return progress


def _run_reduce_stage(*, run_dir: Path, stage_config: StageConfig) -> dict[str, Any]:
    del stage_config

    map_dir = run_dir / "map"
    if not map_dir.exists():
        raise StageExecutionError(
            "reduce stage requires map outputs, but map/ is missing"
        )

    map_files = sorted(path for path in map_dir.glob("chunk-*.json") if path.is_file())
    if not map_files:
        raise StageExecutionError("reduce stage found no chunk map outputs in map/")

    sections_dir = run_dir / "final" / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)

    section_entries: list[dict[str, Any]] = []
    merged_blocks: list[str] = []

    for path in map_files:
        item = json.loads(path.read_text(encoding="utf-8"))
        chunk_id = str(item.get("chunk_id", path.stem))
        heading = str(item.get("heading", "")).strip() or chunk_id
        markdown = str(item.get("normalized_markdown", "")).strip()
        if not markdown:
            continue

        section_file = f"{chunk_id}.md"
        section_path = sections_dir / section_file
        _write_text_if_changed(section_path, _ensure_trailing_newline(markdown))

        section_entries.append(
            {
                "chunk_id": chunk_id,
                "heading": heading,
                "file": f"sections/{section_file}",
            }
        )
        merged_blocks.append(markdown)

    if not section_entries:
        raise StageExecutionError("reduce stage produced no section outputs")

    merged_markdown = _ensure_trailing_newline(
        "\n\n".join(_dedupe_adjacent(merged_blocks))
    )
    index_lines = ["# Document Index", ""]
    for entry in section_entries:
        index_lines.append(f"- [{entry['heading']}]({entry['file']})")
    index_markdown = _ensure_trailing_newline("\n".join(index_lines))

    final_dir = run_dir / "final"
    _write_text_if_changed(final_dir / "merged.md", merged_markdown)
    _write_text_if_changed(final_dir / "index.md", index_markdown)

    reduce_manifest = {
        "generated_at": utc_now_iso(),
        "section_count": len(section_entries),
        "files": section_entries,
    }
    _write_json_if_changed(final_dir / "manifest.json", reduce_manifest)

    return {
        "handler": "reduce",
        "section_count": len(section_entries),
        "outputs": "final/index.md, final/merged.md, final/sections/*.md",
    }


def _run_validate_stage(*, run_dir: Path, stage_config: StageConfig) -> dict[str, Any]:
    merged_path = run_dir / "final" / "merged.md"
    if not merged_path.exists():
        raise StageExecutionError("validate stage requires final/merged.md")

    merged = merged_path.read_text(encoding="utf-8")
    source_path = run_dir / "raw" / "cleaned.md"
    if not source_path.exists():
        source_path = run_dir / "raw" / "extracted.md"
    source_text = (
        source_path.read_text(encoding="utf-8") if source_path.exists() else ""
    )

    code_fence_issues = _count_unbalanced_code_fences(merged)
    heading_jumps = _count_heading_jumps(merged)
    source_endpoints = _extract_endpoints(source_text)
    output_endpoints = _extract_endpoints(merged)
    missing_endpoints = sorted(source_endpoints - output_endpoints)
    merged_think_tag_count = _count_think_tags(merged)

    map_think_chunks: list[str] = []
    map_dir = run_dir / "map"
    if map_dir.exists():
        for map_file in sorted(map_dir.glob("chunk-*.json")):
            try:
                item = json.loads(map_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            normalized = str(item.get("normalized_markdown", ""))
            if _count_think_tags(normalized) > 0:
                map_think_chunks.append(str(item.get("chunk_id", map_file.stem)))

    hard_errors: list[str] = []
    warnings: list[str] = []

    if code_fence_issues:
        hard_errors.append(f"Unbalanced code fences detected: {code_fence_issues}")
    if missing_endpoints:
        hard_errors.append(
            "Missing endpoints in normalized output: "
            + ", ".join(missing_endpoints[:20])
        )
    if merged_think_tag_count > 0:
        hard_errors.append(
            f"Reasoning tags leaked into final merged output: {merged_think_tag_count}"
        )
    if map_think_chunks:
        hard_errors.append(
            "Reasoning tags leaked into map outputs: "
            + ", ".join(map_think_chunks[:20])
        )
    if heading_jumps:
        warnings.append(f"Heading depth jumps detected: {heading_jumps}")

    report = {
        "generated_at": utc_now_iso(),
        "source_file": str(source_path.relative_to(run_dir))
        if source_path.exists()
        else "",
        "merged_file": "final/merged.md",
        "checks": {
            "code_fence_unbalanced_count": code_fence_issues,
            "heading_jump_count": heading_jumps,
            "source_endpoint_count": len(source_endpoints),
            "output_endpoint_count": len(output_endpoints),
            "missing_endpoint_count": len(missing_endpoints),
            "merged_think_tag_count": merged_think_tag_count,
            "map_think_chunk_count": len(map_think_chunks),
        },
        "hard_errors": hard_errors,
        "warnings": warnings,
        "ok": len(hard_errors) == 0,
    }
    _write_json_if_changed(run_dir / "final" / "validation_report.json", report)

    if stage_config.fail_on_hard_errors and hard_errors:
        raise StageExecutionError("Validation failed with hard errors")

    return {
        "handler": "validate",
        "ok": str(report["ok"]).lower(),
        "hard_error_count": len(hard_errors),
        "warning_count": len(warnings),
    }


def _run_export_stage(*, run_dir: Path, stage_config: StageConfig) -> dict[str, Any]:
    final_dir = run_dir / "final"
    merged_path = final_dir / "merged.md"
    if not merged_path.exists():
        raise StageExecutionError("export stage requires final/merged.md")

    index_path = final_dir / "index.md"
    section_files = (
        sorted((final_dir / "sections").glob("*.md"))
        if (final_dir / "sections").exists()
        else []
    )
    validation_path = final_dir / "validation_report.json"
    validation_report = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.exists()
        else None
    )

    multi_file = (
        stage_config.multi_file if stage_config.multi_file is not None else True
    )
    export_file = "final/index.md" if multi_file else "final/export.md"

    if not multi_file:
        merged_text = merged_path.read_text(encoding="utf-8")
        _write_text_if_changed(
            final_dir / "export.md", _ensure_trailing_newline(merged_text.strip())
        )

    summary = {
        "generated_at": utc_now_iso(),
        "multi_file": multi_file,
        "export_entry": export_file,
        "artifacts": {
            "merged": "final/merged.md",
            "index": "final/index.md" if index_path.exists() else "",
            "sections": len(section_files),
            "validation": "final/validation_report.json" if validation_report else "",
        },
        "validation_ok": validation_report.get("ok") if validation_report else None,
    }
    _write_json_if_changed(final_dir / "export_summary.json", summary)

    return {
        "handler": "export",
        "multi_file": str(multi_file).lower(),
        "export_entry": export_file,
        "section_count": len(section_files),
    }


def _normalize_chunk_with_llm(
    *,
    chunk_text: str,
    heading: str,
    model_session: ModelSession,
    temperature: float,
    request_timeout_s: int,
    max_output_tokens: int | None,
    reasoning_effort: str | None,
    reasoning_exclude: bool | None,
) -> CompletionResult:
    system_prompt = (
        "You normalize OCR markdown for technical API documentation. "
        "Preserve all technical facts, identifiers, endpoint paths, and code. "
        "Do not invent content. Improve only markdown structure/readability. "
        "Return markdown only without code fences."
    )
    user_prompt = (
        f"Heading context: {heading or 'unknown'}\n\n"
        "Normalize the following chunk faithfully:\n\n"
        f"{chunk_text}"
    )
    try:
        return model_session.client.complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            request_timeout_s=request_timeout_s,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_exclude=reasoning_exclude,
        )
    except (ContextWindowError, ModelRequestTimeoutError):
        raise
    except RateLimitError:
        raise
    except ModelClientError as exc:
        raise StageExecutionError(str(exc)) from exc


def _normalize_chunk_with_llm_with_context_fallback(
    *,
    chunk_text: str,
    heading: str,
    model_session: ModelSession,
    temperature: float,
    request_timeout_s: int,
    max_output_tokens: int | None,
    reasoning_effort: str | None,
    reasoning_exclude: bool | None,
    depth: int = 0,
) -> CompletionResult:
    try:
        return _normalize_chunk_with_llm(
            chunk_text=chunk_text,
            heading=heading,
            model_session=model_session,
            temperature=temperature,
            request_timeout_s=request_timeout_s,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_exclude=reasoning_exclude,
        )
    except (ContextWindowError, ModelRequestTimeoutError) as exc:
        if depth >= 6:
            raise StageExecutionError(
                "Chunk exceeds model context after recursive splitting. "
                "Reduce sectionize target_tokens or increase backend context window."
            ) from exc

        if len(chunk_text) < 600:
            raise StageExecutionError(
                "Chunk too small to split further but still exceeds model context. "
                "Increase backend context window."
            ) from exc

        left, right = _split_chunk_for_context(chunk_text)
        if not left or not right:
            raise StageExecutionError(
                "Failed to split oversized chunk for context fallback."
            ) from exc

        left_result = _normalize_chunk_with_llm_with_context_fallback(
            chunk_text=left,
            heading=heading,
            model_session=model_session,
            temperature=temperature,
            request_timeout_s=request_timeout_s,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_exclude=reasoning_exclude,
            depth=depth + 1,
        )
        right_result = _normalize_chunk_with_llm_with_context_fallback(
            chunk_text=right,
            heading=heading,
            model_session=model_session,
            temperature=temperature,
            request_timeout_s=request_timeout_s,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_exclude=reasoning_exclude,
            depth=depth + 1,
        )

        combined_text = (
            _ensure_trailing_newline(left_result.text.strip())
            + "\n"
            + _ensure_trailing_newline(right_result.text.strip())
        )
        return CompletionResult(
            text=combined_text,
            prompt_tokens=_add_optional_int(
                left_result.prompt_tokens, right_result.prompt_tokens
            ),
            completion_tokens=_add_optional_int(
                left_result.completion_tokens, right_result.completion_tokens
            ),
            total_tokens=_add_optional_int(
                left_result.total_tokens, right_result.total_tokens
            ),
            latency_s=left_result.latency_s + right_result.latency_s,
            requests=left_result.requests + right_result.requests,
            split_count=left_result.split_count + right_result.split_count + 1,
        )


def _coerce_usage_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(float(stripped))
        except ValueError:
            return None
    return None


def _add_optional_int(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return int(left or 0) + int(right or 0)


def _chunk_telemetry_from_completion(
    completion: CompletionResult,
    normalized_markdown: str,
    *,
    model_name: str | None,
) -> dict[str, Any]:
    output_tokens_est = estimate_token_count(
        normalized_markdown,
        model=model_name,
    ).count
    return {
        "requests": completion.requests,
        "split_count": completion.split_count,
        "latency_s": round(completion.latency_s, 3),
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "total_tokens": completion.total_tokens,
        "output_tokens_est": output_tokens_est,
        "effective_tokens_per_second": _effective_tokens_per_second(
            completion_tokens=completion.completion_tokens,
            output_tokens_est=output_tokens_est,
            latency_s=completion.latency_s,
        ),
    }


def _effective_tokens_per_second(
    *,
    completion_tokens: int | None,
    output_tokens_est: int,
    latency_s: float,
) -> float | None:
    if latency_s <= 0:
        return None
    tokens = completion_tokens if completion_tokens is not None else output_tokens_est
    return round(tokens / latency_s, 3)


def _aggregate_map_telemetry(map_dir: Path) -> dict[str, Any]:
    files = sorted(map_dir.glob("chunk-*.json"))
    if not files:
        return {
            "chunks": 0,
            "requests": 0,
            "latency_s": 0.0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "output_tokens_est": 0,
            "chunks_with_usage": 0,
            "effective_tokens_per_second": None,
        }

    requests_total = 0
    latency_total = 0.0
    prompt_total = 0
    completion_total = 0
    tokens_total = 0
    output_est_total = 0
    usage_chunks = 0

    for file_path in files:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        telemetry = data.get("telemetry", {})
        if not isinstance(telemetry, dict):
            continue

        requests_total += int(telemetry.get("requests", 1) or 1)
        latency_total += float(telemetry.get("latency_s", 0.0) or 0.0)
        output_est_total += int(telemetry.get("output_tokens_est", 0) or 0)

        prompt_val = _coerce_usage_int(telemetry.get("prompt_tokens"))
        completion_val = _coerce_usage_int(telemetry.get("completion_tokens"))
        total_val = _coerce_usage_int(telemetry.get("total_tokens"))
        if prompt_val is not None:
            prompt_total += prompt_val
            usage_chunks += 1
        if completion_val is not None:
            completion_total += completion_val
        if total_val is not None:
            tokens_total += total_val

    return {
        "chunks": len(files),
        "requests": requests_total,
        "latency_s": round(latency_total, 3),
        "prompt_tokens": prompt_total if usage_chunks > 0 else None,
        "completion_tokens": completion_total if usage_chunks > 0 else None,
        "total_tokens": tokens_total if usage_chunks > 0 else None,
        "output_tokens_est": output_est_total,
        "chunks_with_usage": usage_chunks,
        "effective_tokens_per_second": _effective_tokens_per_second(
            completion_tokens=completion_total if usage_chunks > 0 else None,
            output_tokens_est=output_est_total,
            latency_s=latency_total,
        ),
    }


def _dedupe_adjacent(blocks: list[str]) -> list[str]:
    if not blocks:
        return []
    deduped = [blocks[0]]
    for block in blocks[1:]:
        if block != deduped[-1]:
            deduped.append(block)
    return deduped


def _split_chunk_for_context(chunk_text: str) -> tuple[str, str]:
    lines = chunk_text.splitlines()
    if len(lines) < 2:
        midpoint = len(chunk_text) // 2
        return chunk_text[:midpoint].strip(), chunk_text[midpoint:].strip()

    midpoint = len(lines) // 2
    split_idx = _find_split_index(lines, midpoint)
    if split_idx is None:
        split_idx = midpoint

    left = "\n".join(lines[:split_idx]).strip()
    right = "\n".join(lines[split_idx:]).strip()

    if not left or not right:
        midpoint_chars = len(chunk_text) // 2
        left = chunk_text[:midpoint_chars].strip()
        right = chunk_text[midpoint_chars:].strip()
    return left, right


def _find_split_index(lines: list[str], midpoint: int) -> int | None:
    for radius in range(0, max(midpoint, len(lines) - midpoint)):
        left_idx = midpoint - radius
        right_idx = midpoint + radius
        if 0 < left_idx < len(lines) and not lines[left_idx].strip():
            return left_idx
        if 0 < right_idx < len(lines) and not lines[right_idx].strip():
            return right_idx

    for radius in range(0, max(midpoint, len(lines) - midpoint)):
        left_idx = midpoint - radius
        right_idx = midpoint + radius
        if 0 < left_idx < len(lines) and re.match(
            r"^\s{0,3}#{1,6}\s+\S", lines[left_idx]
        ):
            return left_idx
        if 0 < right_idx < len(lines) and re.match(
            r"^\s{0,3}#{1,6}\s+\S", lines[right_idx]
        ):
            return right_idx

    return None


def _count_unbalanced_code_fences(markdown: str) -> int:
    fence_lines = [
        line for line in markdown.splitlines() if line.strip().startswith("```")
    ]
    return len(fence_lines) % 2


def _count_heading_jumps(markdown: str) -> int:
    jumps = 0
    previous_level: int | None = None
    for line in markdown.splitlines():
        match = re.match(r"^\s{0,3}(#{1,6})\s+\S", line)
        if not match:
            continue
        level = len(match.group(1))
        if previous_level is not None and level - previous_level > 1:
            jumps += 1
        previous_level = level
    return jumps


def _count_think_tags(markdown: str) -> int:
    text = markdown.lower()
    return text.count("<think>") + text.count("</think>")


def _extract_endpoints(markdown: str) -> set[str]:
    methods = "GET|POST|PUT|PATCH|DELETE|OPTIONS|HEAD"
    canonical_pattern = re.compile(
        rf"\b({methods})\s+(/[A-Za-z0-9._~!$&'()*+,;=:@%/-]+)",
        re.IGNORECASE,
    )
    tolerant_markdown_pattern = re.compile(
        rf"\b({methods})\b(?:\s|[*_`]|:|-)+(/[A-Za-z0-9._~!$&'()*+,;=:@%/-]+)",
        re.IGNORECASE,
    )

    endpoints: set[str] = set()
    for pattern in (canonical_pattern, tolerant_markdown_pattern):
        for method, path in pattern.findall(markdown):
            endpoints.add(f"{method.upper()} {path}")
    return endpoints


def _clean_markdown(content: str) -> tuple[str, dict[str, int]]:
    normalized = content.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n")
    normalized = re.sub(r"(?<=\w)-\n(?=[a-z])", "", normalized)

    lines = normalized.split("\n")
    repeated_lines = _detect_repeated_noise_lines(lines)

    cleaned_lines: list[str] = []
    removed_repeated_lines = 0
    removed_page_numbers = 0

    for line in lines:
        stripped = line.strip()
        if stripped and stripped in repeated_lines:
            removed_repeated_lines += 1
            continue
        if re.fullmatch(r"(?:page\s+)?\d{1,4}", stripped, re.IGNORECASE):
            removed_page_numbers += 1
            continue
        cleaned_lines.append(line.rstrip())

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = _ensure_trailing_newline(cleaned.strip())

    report = {
        "removed_repeated_lines": removed_repeated_lines,
        "removed_page_numbers": removed_page_numbers,
        "repeated_line_patterns": len(repeated_lines),
    }
    return cleaned, report


def _detect_repeated_noise_lines(lines: list[str]) -> set[str]:
    candidates: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) > 100:
            continue
        if stripped.startswith(("#", "```", "- ", "* ")):
            continue
        if re.match(r"^\d+[.)]\s", stripped):
            continue
        if "/" in stripped or "{" in stripped or "}" in stripped:
            continue
        candidates.append(stripped)

    threshold = max(8, len(lines) // 120)
    counts = Counter(candidates)
    return {text for text, count in counts.items() if count >= threshold}


def _build_chunk_ranges(
    lines: list[str],
    target_chars: int,
    overlap_lines: int,
) -> list[tuple[int, int, str]]:
    if not lines:
        return []

    heading_indices: list[int] = []
    for idx, line in enumerate(lines):
        if re.match(r"^\s{0,3}#{1,6}\s+\S", line):
            heading_indices.append(idx)

    section_ranges: list[tuple[int, int, str]] = []
    if heading_indices:
        first_heading = heading_indices[0]
        if first_heading > 0:
            section_ranges.append((0, first_heading, "document"))
        for i, start in enumerate(heading_indices):
            end = heading_indices[i + 1] if i + 1 < len(heading_indices) else len(lines)
            heading = _normalize_heading(lines[start])
            section_ranges.append((start, end, heading))
    else:
        section_ranges.append((0, len(lines), "document"))

    chunk_ranges: list[tuple[int, int, str]] = []
    for start, end, heading in section_ranges:
        current = start
        while current < end:
            consumed_chars = 0
            stop = current
            while stop < end:
                line_len = len(lines[stop]) + 1
                if consumed_chars + line_len > target_chars and stop > current:
                    break
                consumed_chars += line_len
                stop += 1

            if stop == current:
                stop = min(current + 1, end)

            chunk_ranges.append((current, stop, heading))
            if stop >= end:
                break

            current = max(stop - overlap_lines, current + 1)

    return chunk_ranges


def _normalize_heading(line: str) -> str:
    stripped = line.strip()
    stripped = re.sub(r"^#{1,6}\s+", "", stripped)
    return stripped if stripped else "section"


def _write_text_if_changed(path: Path, content: str) -> str:
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == content:
            return "unchanged"
        path.write_text(content, encoding="utf-8")
        return "updated"
    path.write_text(content, encoding="utf-8")
    return "created"


def _write_json_if_changed(path: Path, payload: dict[str, Any]) -> str:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    return _write_text_if_changed(path, rendered)


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"
