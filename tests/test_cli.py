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


def _write_scriba_config(home: Path, content: str) -> Path:
    config_path = home / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(content, encoding="utf-8")
    return config_path


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


def test_cli_run_passthrough_uses_runtime_native_home(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    scriba_home = tmp_path / "home"
    monkeypatch.setenv("SCRIBA_HOME", str(scriba_home))

    input_file = tmp_path / "sample.md"
    input_file.write_text("# Doc\n\nGET /v1/ping\n", encoding="utf-8")
    run_id = "run-home-default"

    run_code = main(
        [
            "run",
            "--preset",
            "passthrough",
            "--input",
            str(input_file),
            "--run-id",
            run_id,
        ]
    )

    assert run_code == 0
    state_path = scriba_home / "artifacts" / run_id / "state.json"
    assert state_path.exists()
    assert '"run_id": "run-home-default"' in capsys.readouterr().out


def test_cli_run_output_copies_final_directory(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SCRIBA_HOME", str(tmp_path / "home"))
    input_file = tmp_path / "sample.md"
    input_file.write_text("# Doc\n\nGET /v1/ping\n", encoding="utf-8")

    output_dir = tmp_path / "copied-final"
    run_code = main(
        [
            "run",
            "--preset",
            "passthrough",
            "--input",
            str(input_file),
            "--run-id",
            "run-copy-output",
            "--output",
            str(output_dir),
        ]
    )

    assert run_code == 0
    assert (output_dir / "merged.md").exists()
    assert (output_dir / "index.md").exists()
    assert (output_dir / "sections").is_dir()
    assert "final outputs copied to:" in capsys.readouterr().err


def test_cli_run_output_rejects_file_destination(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SCRIBA_HOME", str(tmp_path / "home"))
    input_file = tmp_path / "sample.md"
    input_file.write_text("# Doc\n\nGET /v1/ping\n", encoding="utf-8")
    output_file = tmp_path / "out.md"
    output_file.write_text("existing\n", encoding="utf-8")

    run_code = main(
        [
            "run",
            "--preset",
            "passthrough",
            "--input",
            str(input_file),
            "--run-id",
            "run-copy-output-error",
            "--output",
            str(output_file),
        ]
    )

    assert run_code == 2
    assert "--output destination must be a directory path" in capsys.readouterr().err


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


def test_cli_uses_config_default_preset_and_model(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    scriba_home = tmp_path / "home"
    monkeypatch.setenv("SCRIBA_HOME", str(scriba_home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-token")
    _write_scriba_config(
        scriba_home,
        "\n".join(
            [
                "version: 1",
                "defaults:",
                "  preset: openrouter",
                "models:",
                "  openrouter: openrouter/custom-model",
                "",
            ]
        ),
    )

    input_file = tmp_path / "sample.md"
    input_file.write_text("# Doc\n\nGET /v1/ping\n", encoding="utf-8")

    captured_model: dict[str, str] = {}

    def _capture_completion(**kwargs):  # type: ignore[no-untyped-def]
        captured_model["model"] = str(kwargs["model"])
        return {
            "choices": [{"message": {"content": "# Doc\n\nGET /v1/ping\n"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    with (
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter._probe_health",
            return_value=(True, "ok"),
        ),
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter.litellm_completion",
            side_effect=_capture_completion,
        ),
    ):
        run_code = main(
            ["run", "--input", str(input_file), "--run-id", "run-config-model"]
        )

    assert run_code == 0
    assert captured_model["model"] == "openrouter/custom-model"
    assert '"run_id": "run-config-model"' in capsys.readouterr().out


def test_cli_status_uses_config_artifacts_root_without_provider_key(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    scriba_home = tmp_path / "home"
    monkeypatch.setenv("SCRIBA_HOME", str(scriba_home))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CEREBRAS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    artifacts_root = tmp_path / "shared-artifacts"
    _write_scriba_config(
        scriba_home,
        "\n".join(
            [
                "version: 1",
                "defaults:",
                f"  artifacts_root: {artifacts_root}",
                "",
            ]
        ),
    )

    input_file = tmp_path / "sample.md"
    input_file.write_text("# Doc\n\nGET /v1/ping\n", encoding="utf-8")
    run_id = "run-config-artifacts"

    run_code = main(
        [
            "run",
            "--preset",
            "passthrough",
            "--input",
            str(input_file),
            "--run-id",
            run_id,
        ]
    )
    assert run_code == 0
    capsys.readouterr()

    status_code = main(["status", "--run-id", run_id])

    assert status_code == 0
    assert f'"run_id": "{run_id}"' in capsys.readouterr().out


def test_cli_config_provider_priority_controls_auto_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scriba_home = tmp_path / "home"
    monkeypatch.setenv("SCRIBA_HOME", str(scriba_home))
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-token")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-token")
    _write_scriba_config(
        scriba_home,
        "\n".join(
            [
                "version: 1",
                "defaults:",
                "  provider_priority:",
                "    - openai",
                "    - openrouter",
                "",
            ]
        ),
    )

    input_file = tmp_path / "sample.md"
    input_file.write_text("# Doc\n\nGET /v1/ping\n", encoding="utf-8")

    captured_model: dict[str, str] = {}

    def _capture_completion(**kwargs):  # type: ignore[no-untyped-def]
        captured_model["model"] = str(kwargs["model"])
        return {
            "choices": [{"message": {"content": "# Doc\n\nGET /v1/ping\n"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    with (
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter._probe_health",
            return_value=(True, "ok"),
        ),
        patch(
            "scriba.pipeline.backends.adapters.litellm_adapter.litellm_completion",
            side_effect=_capture_completion,
        ),
    ):
        run_code = main(["run", "--input", str(input_file), "--run-id", "run-priority"])

    assert run_code == 0
    assert captured_model["model"] == "openai/gpt-4o-mini"


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
