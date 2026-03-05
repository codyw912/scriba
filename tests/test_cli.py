"""CLI tests for scriba scaffold commands."""

from pathlib import Path

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
