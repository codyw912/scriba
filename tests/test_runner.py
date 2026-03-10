"""Tests for pipeline runner orchestration."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scriba.pipeline import PipelineError, PipelineRunner, load_profile
from scriba.pipeline.backends import CompletionResult


class _FakeChatClient:
    def __init__(self, result: CompletionResult) -> None:
        self._result = result

    def complete(self, **_: object) -> CompletionResult:
        return self._result


def _write_profile(tmp_path: Path) -> Path:
    profile_path = tmp_path / "profile.yaml"
    artifacts_root = (tmp_path / "artifacts").as_posix()
    profile_path.write_text(
        f"version: 1\nartifacts:\n  root: {artifacts_root}\n",
        encoding="utf-8",
    )
    return profile_path


def _write_remote_profile(
    tmp_path: Path,
    *,
    api_key: str = "",
    fail_on_hard_errors: bool | None = None,
) -> Path:
    profile_path = tmp_path / "profile_remote.yaml"
    artifacts_root = (tmp_path / "artifacts").as_posix()
    lines = [
        "version: 1",
        "artifacts:",
        f"  root: {artifacts_root}",
        "backends:",
        "  remote_text:",
        "    adapter: litellm",
        "    topology: remote",
        "    provider: openrouter",
        "    model_origin: hosted_weights",
        "    base_url: https://openrouter.ai/api",
        f"    api_key: {api_key}",
        "roles:",
        "  normalize_text:",
        "    backend: remote_text",
        "    model: qwen/qwen3.5-35b-a3b",
    ]
    if fail_on_hard_errors is not None:
        lines.extend(
            [
                "stages:",
                "  validate:",
                f"    fail_on_hard_errors: {'true' if fail_on_hard_errors else 'false'}",
            ]
        )
    lines.append("")
    profile_path.write_text("\n".join(lines), encoding="utf-8")
    return profile_path


def _write_cerebras_profile(tmp_path: Path, *, api_key: str = '"token"') -> Path:
    profile_path = tmp_path / "profile_cerebras.yaml"
    artifacts_root = (tmp_path / "artifacts").as_posix()
    profile_path.write_text(
        "\n".join(
            [
                "version: 1",
                "artifacts:",
                f"  root: {artifacts_root}",
                "backends:",
                "  remote_text:",
                "    adapter: litellm",
                "    topology: remote",
                "    provider: cerebras",
                "    model_origin: hosted_weights",
                "    base_url: https://api.cerebras.ai",
                f"    api_key: {api_key}",
                "roles:",
                "  normalize_text:",
                "    backend: remote_text",
                "    model: llama3.1-8b",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return profile_path


def test_run_writes_state(tmp_path: Path) -> None:
    profile = load_profile(_write_profile(tmp_path))
    runner = PipelineRunner(profile)

    input_file = tmp_path / "sample.md"
    input_file.write_text(
        "# API\n\nGET /v1/health\n\n## Details\n\nService status endpoint.\n",
        encoding="utf-8",
    )

    state = runner.run(input_path=str(input_file), run_id="run-test", resume=False)
    assert state["run_id"] == "run-test"
    assert state["status"] == "completed"

    run_dir = tmp_path / "artifacts" / "run-test"
    assert (run_dir / "raw" / "extracted.md").exists()
    assert (run_dir / "chunks" / "manifest.json").exists()
    assert (run_dir / "map" / "manifest.json").exists()
    assert (run_dir / "final" / "merged.md").exists()
    assert (run_dir / "final" / "validation_report.json").exists()


def test_run_resume_returns_existing_state(tmp_path: Path) -> None:
    profile = load_profile(_write_profile(tmp_path))
    runner = PipelineRunner(profile)

    input_file = tmp_path / "sample.md"
    input_file.write_text("# Doc\n\nGET /v1/ping\n", encoding="utf-8")

    first = runner.run(input_path=str(input_file), run_id="run-resume", resume=False)
    second = runner.run(input_path=str(input_file), run_id="run-resume", resume=True)
    assert first["run_id"] == second["run_id"]
    assert second["status"] == "completed"


def test_status_errors_when_missing(tmp_path: Path) -> None:
    profile = load_profile(_write_profile(tmp_path))
    runner = PipelineRunner(profile)

    with pytest.raises(PipelineError):
        runner.status(run_id="missing")


def test_role_for_extract_uses_ocr_vision_for_pdf_inputs(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.yaml"
    artifacts_root = (tmp_path / "artifacts").as_posix()
    profile_path.write_text(
        "\n".join(
            [
                "version: 1",
                "artifacts:",
                f"  root: {artifacts_root}",
                "backends:",
                "  ocr_backend:",
                "    adapter: litellm",
                "    topology: local_attached",
                "    provider: lmstudio",
                "    base_url: http://127.0.0.1:8090",
                "roles:",
                "  ocr_vision:",
                "    backend: ocr_backend",
                "    model: glm-4.5v",
                "",
            ]
        ),
        encoding="utf-8",
    )
    profile = load_profile(profile_path)
    runner = PipelineRunner(profile)

    role = runner._role_for_stage(
        stage_name="extract",
        state={"input_path": str(tmp_path / "sample.pdf")},
    )
    assert role == "ocr_vision"

    role_non_pdf = runner._role_for_stage(
        stage_name="extract",
        state={"input_path": str(tmp_path / "sample.md")},
    )
    assert role_non_pdf is None


def test_doctor_warns_when_remote_api_key_is_empty(tmp_path: Path) -> None:
    profile = load_profile(_write_remote_profile(tmp_path, api_key='""'))
    runner = PipelineRunner(profile)

    input_file = tmp_path / "sample.md"
    input_file.write_text("# Doc\n\nGET /v1/ping\n", encoding="utf-8")

    report = runner.doctor(input_path=str(input_file))
    assert report["ok"] is True
    warnings = report["warnings"]
    assert any("api_key is empty" in warning for warning in warnings)


def test_doctor_no_remote_api_key_warning_when_env_expands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-token")
    profile = load_profile(
        _write_remote_profile(tmp_path, api_key='"${OPENROUTER_API_KEY}"')
    )
    runner = PipelineRunner(profile)

    input_file = tmp_path / "sample.md"
    input_file.write_text("# Doc\n\nGET /v1/ping\n", encoding="utf-8")

    report = runner.doctor(input_path=str(input_file))
    warnings = report["warnings"]
    assert not any("api_key is empty" in warning for warning in warnings)


def test_run_remote_profile_with_mocked_health_and_completion(tmp_path: Path) -> None:
    profile = load_profile(_write_remote_profile(tmp_path, api_key='"token"'))
    runner = PipelineRunner(profile)

    input_file = tmp_path / "sample.md"
    input_file.write_text("# API\n\nGET /v1/ping\n", encoding="utf-8")

    with (
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter._probe_health",
            return_value=(True, "ok"),
        ),
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter.LiteLLMChatClient.complete",
            return_value=CompletionResult(
                text="# API\n\nGET /v1/ping\n",
                prompt_tokens=100,
                completion_tokens=40,
                total_tokens=140,
                latency_s=2.0,
                requests=1,
                split_count=0,
            ),
        ),
    ):
        state = runner.run(
            input_path=str(input_file), run_id="run-remote", resume=False
        )

    assert state["status"] == "completed"
    run_dir = tmp_path / "artifacts" / "run-remote"
    map_manifest_path = run_dir / "map" / "manifest.json"
    assert map_manifest_path.exists()
    map_manifest = json.loads(map_manifest_path.read_text(encoding="utf-8"))
    assert map_manifest["processed"] >= 1
    assert map_manifest["processed_telemetry"]["chunks_with_usage"] >= 1


def test_run_sectionize_uses_metadata_from_normalize_role(tmp_path: Path) -> None:
    profile = load_profile(_write_remote_profile(tmp_path, api_key='"token"'))
    runner = PipelineRunner(profile)

    input_file = tmp_path / "sample.md"
    input_file.write_text("# API\n\nGET /v1/ping\n", encoding="utf-8")

    with (
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter._probe_health",
            return_value=(True, "ok"),
        ),
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter.lookup_context_length_from_openrouter",
            return_value=20000,
        ),
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter.LiteLLMChatClient.complete",
            return_value=CompletionResult(
                text="# API\n\nGET /v1/ping\n",
                prompt_tokens=80,
                completion_tokens=30,
                total_tokens=110,
                latency_s=1.0,
                requests=1,
                split_count=0,
            ),
        ),
    ):
        state = runner.run(
            input_path=str(input_file), run_id="run-sectionize-metadata", resume=False
        )

    sectionize_details = state["stages"]["sectionize"]["details"]
    assert sectionize_details["target_tokens_source"] == "metadata"
    assert sectionize_details["overlap_tokens_source"] == "metadata"
    assert sectionize_details["target_tokens"] == 8539
    assert sectionize_details["overlap_tokens"] == 939

    run_dir = tmp_path / "artifacts" / "run-sectionize-metadata"
    manifest = json.loads((run_dir / "chunks" / "manifest.json").read_text("utf-8"))
    assert manifest["sectionize_context_length"] == 20000


def test_run_cerebras_profile_uses_litellm_adapter(tmp_path: Path) -> None:
    profile = load_profile(_write_cerebras_profile(tmp_path, api_key='"token"'))
    runner = PipelineRunner(profile)

    input_file = tmp_path / "sample.md"
    input_file.write_text("# API\n\nGET /v1/ping\n", encoding="utf-8")

    with (
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter._probe_health",
            return_value=(True, "ok"),
        ),
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter.LiteLLMChatClient.complete",
            return_value=CompletionResult(
                text="# API\n\nGET /v1/ping\n",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
                latency_s=0.5,
                requests=1,
                split_count=0,
            ),
        ),
    ):
        state = runner.run(
            input_path=str(input_file),
            run_id="run-cerebras-sdk-test",
            resume=False,
        )

    assert state["status"] == "completed"


def test_run_marks_completed_with_validation_errors_when_validation_finds_hard_errors(
    tmp_path: Path,
) -> None:
    profile = load_profile(
        _write_remote_profile(
            tmp_path,
            api_key='"token"',
            fail_on_hard_errors=False,
        )
    )
    runner = PipelineRunner(profile)

    input_file = tmp_path / "sample.md"
    input_file.write_text("# API\n\nGET /v1/ping\n", encoding="utf-8")

    with (
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter._probe_health",
            return_value=(True, "ok"),
        ),
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter.LiteLLMChatClient.complete",
            return_value=CompletionResult(
                text="# API\n\nNo endpoint preserved here.\n",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
                latency_s=0.5,
                requests=1,
                split_count=0,
            ),
        ),
    ):
        state = runner.run(
            input_path=str(input_file),
            run_id="run-validation-errors",
            resume=False,
        )

    assert state["status"] == "completed_with_validation_errors"
    run_dir = tmp_path / "artifacts" / "run-validation-errors"
    validation = json.loads(
        (run_dir / "final" / "validation_report.json").read_text(encoding="utf-8")
    )
    assert validation["ok"] is False
    assert len(validation["hard_errors"]) > 0
    assert (run_dir / "final" / "export_summary.json").exists()


def test_run_marks_failed_runtime_when_strict_validation_raises(
    tmp_path: Path,
) -> None:
    profile = load_profile(
        _write_remote_profile(
            tmp_path,
            api_key='"token"',
            fail_on_hard_errors=True,
        )
    )
    runner = PipelineRunner(profile)

    input_file = tmp_path / "sample.md"
    input_file.write_text("# API\n\nGET /v1/ping\n", encoding="utf-8")

    with (
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter._probe_health",
            return_value=(True, "ok"),
        ),
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter.LiteLLMChatClient.complete",
            return_value=CompletionResult(
                text="# API\n\nNo endpoint preserved here.\n",
                prompt_tokens=50,
                completion_tokens=20,
                total_tokens=70,
                latency_s=0.5,
                requests=1,
                split_count=0,
            ),
        ),
    ):
        with pytest.raises(PipelineError, match="Stage 'validate' failed"):
            runner.run(
                input_path=str(input_file),
                run_id="run-strict-validation-errors",
                resume=False,
            )

    state = runner.status(run_id="run-strict-validation-errors")
    assert state["status"] == "failed_runtime"
    run_dir = tmp_path / "artifacts" / "run-strict-validation-errors"
    assert (run_dir / "final" / "merged.md").exists()
    assert (run_dir / "final" / "validation_report.json").exists()
    assert not (run_dir / "final" / "export_summary.json").exists()
