"""CLI tests for scriba scaffold commands."""

from pathlib import Path
from unittest.mock import patch

from scriba.cli import _reasoning_efficiency_warning, main


def _write_profile(tmp_path: Path) -> Path:
    profile_path = tmp_path / "profile.yaml"
    artifacts_root = (tmp_path / "artifacts").as_posix()
    profile_path.write_text(
        f"version: 1\nartifacts:\n  root: {artifacts_root}\n",
        encoding="utf-8",
    )
    return profile_path


def test_cli_doctor(tmp_path: Path, capsys) -> None:
    profile = _write_profile(tmp_path)
    input_file = tmp_path / "sample.md"
    input_file.write_text("# Doc\n\nGET /v1/ping\n", encoding="utf-8")

    code = main(
        [
            "doctor",
            "--profile",
            str(profile),
            "--input",
            str(input_file),
        ]
    )

    assert code == 0
    assert '"ok": true' in capsys.readouterr().out


def test_cli_run_then_status(tmp_path: Path, capsys) -> None:
    profile = _write_profile(tmp_path)
    input_file = tmp_path / "sample.md"
    input_file.write_text("# Doc\n\nGET /v1/ping\n", encoding="utf-8")
    run_id = "run-cli"

    run_code = main(
        [
            "run",
            "--profile",
            str(profile),
            "--input",
            str(input_file),
            "--run-id",
            run_id,
        ]
    )
    assert run_code == 0
    run_streams = capsys.readouterr()
    assert "map telemetry:" in run_streams.err
    assert "source=output_tokens_est" in run_streams.err

    status_code = main(
        [
            "status",
            "--profile",
            str(profile),
            "--run-id",
            run_id,
        ]
    )
    assert status_code == 0
    assert '"run_id": "run-cli"' in capsys.readouterr().out


def test_cli_doctor_missing_input_returns_error(tmp_path: Path, capsys) -> None:
    profile = _write_profile(tmp_path)
    missing = tmp_path / "missing.md"

    code = main(
        [
            "doctor",
            "--profile",
            str(profile),
            "--input",
            str(missing),
        ]
    )

    assert code == 2
    assert "Input file not found" in capsys.readouterr().err


def test_reasoning_efficiency_warning_thresholds() -> None:
    assert (
        _reasoning_efficiency_warning(completion_tokens=2000, output_tokens_est=100)
        is not None
    )
    assert (
        _reasoning_efficiency_warning(completion_tokens=500, output_tokens_est=100)
        is None
    )
    assert (
        _reasoning_efficiency_warning(completion_tokens=None, output_tokens_est=100)
        is None
    )


def test_cli_run_with_passthrough_preset_and_artifacts_override(
    tmp_path: Path, capsys
) -> None:
    input_file = tmp_path / "sample.md"
    input_file.write_text("# Doc\n\nGET /v1/ping\n", encoding="utf-8")

    artifacts_root = tmp_path / "custom-artifacts"
    run_id = "run-preset"
    run_code = main(
        [
            "run",
            "--preset",
            "passthrough",
            "--input",
            str(input_file),
            "--artifacts-root",
            str(artifacts_root),
            "--run-id",
            run_id,
        ]
    )

    assert run_code == 0
    capsys.readouterr()

    status_code = main(
        [
            "status",
            "--preset",
            "passthrough",
            "--run-id",
            run_id,
            "--artifacts-root",
            str(artifacts_root),
        ]
    )
    assert status_code == 0
    assert '"run_id": "run-preset"' in capsys.readouterr().out


def test_cli_run_defaults_to_auto_preset_requires_provider_key(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    input_file = tmp_path / "sample.md"
    input_file.write_text("# Doc\n\nGET /v1/ping\n", encoding="utf-8")

    artifacts_root = tmp_path / "default-preset-artifacts"
    run_code = main(
        [
            "run",
            "--input",
            str(input_file),
            "--artifacts-root",
            str(artifacts_root),
        ]
    )

    assert run_code == 2
    assert "No provider API key detected" in capsys.readouterr().err


def test_cli_run_defaults_to_auto_preset_with_openrouter_key(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-token")

    input_file = tmp_path / "sample.md"
    input_file.write_text("# Doc\n\nGET /v1/ping\n", encoding="utf-8")
    run_id = "run-default-auto"

    with (
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter._probe_health",
            return_value=(True, "ok"),
        ),
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter.litellm_completion",
            return_value={
                "choices": [{"message": {"content": "# Doc\n\nGET /v1/ping\n"}}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        ),
    ):
        run_code = main(
            [
                "run",
                "--input",
                str(input_file),
                "--artifacts-root",
                str(tmp_path / "auto-artifacts"),
                "--run-id",
                run_id,
            ]
        )

    assert run_code == 0
    assert f'"run_id": "{run_id}"' in capsys.readouterr().out


def test_cli_text_model_override_requires_normalize_role(
    tmp_path: Path,
    capsys,
) -> None:
    profile = _write_profile(tmp_path)
    input_file = tmp_path / "sample.md"
    input_file.write_text("# Doc\n\nGET /v1/ping\n", encoding="utf-8")

    code = main(
        [
            "run",
            "--profile",
            str(profile),
            "--input",
            str(input_file),
            "--text-model",
            "qwen/qwen3.5-35b-a3b",
        ]
    )

    assert code == 2
    assert "no normalize_text role" in capsys.readouterr().err
