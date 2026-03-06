"""CLI entrypoint for scriba."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from scriba.pipeline import PipelineError, PipelineRunner, ProfileError, load_profile
from scriba.pipeline.profile import (
    DEFAULT_STAGE_ORDER,
    ArtifactsConfig,
    BackendConfig,
    PipelineProfile,
    RoleBinding,
    StageConfig,
)


PRESET_CHOICES: tuple[str, ...] = (
    "auto",
    "passthrough",
    "openrouter",
    "cerebras",
    "openai",
)

PRESET_PROVIDER_CONFIG: dict[str, dict[str, str]] = {
    "openrouter": {
        "env_key": "OPENROUTER_API_KEY",
        "provider": "openrouter",
        "default_model": "qwen/qwen3.5-35b-a3b",
    },
    "cerebras": {
        "env_key": "CEREBRAS_API_KEY",
        "provider": "cerebras",
        "default_model": "gpt-oss-120b",
    },
    "openai": {
        "env_key": "OPENAI_API_KEY",
        "provider": "openai",
        "default_model": "gpt-4o-mini",
    },
}

AUTO_PRESET_PRIORITY: tuple[str, ...] = ("openrouter", "cerebras", "openai")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scriba",
        description="CLI-first document normalization pipeline",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a pipeline profile on an input")
    run_profile_group = run_parser.add_mutually_exclusive_group(required=False)
    run_profile_group.add_argument("--profile", help="Path to YAML profile")
    run_profile_group.add_argument(
        "--preset",
        choices=sorted(PRESET_CHOICES),
        help="Built-in profile preset",
    )
    run_parser.add_argument("--input", required=True, help="Path to local input file")
    run_parser.add_argument("--run-id", default=None, help="Optional explicit run id")
    run_parser.add_argument(
        "--artifacts-root",
        default=None,
        help="Override artifacts root directory",
    )
    run_parser.add_argument(
        "--text-model",
        default=None,
        help="Override normalize_text model for this run",
    )
    run_parser.add_argument(
        "--ocr-model",
        default=None,
        help="Override ocr_vision model for this run",
    )
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing run state when available",
    )

    status_parser = subparsers.add_parser("status", help="Read state for a run id")
    status_profile_group = status_parser.add_mutually_exclusive_group(required=False)
    status_profile_group.add_argument("--profile", help="Path to YAML profile")
    status_profile_group.add_argument(
        "--preset",
        choices=sorted(PRESET_CHOICES),
        help="Built-in profile preset",
    )
    status_parser.add_argument("--run-id", required=True, help="Run id to inspect")
    status_parser.add_argument(
        "--artifacts-root",
        default=None,
        help="Override artifacts root directory",
    )

    doctor_parser = subparsers.add_parser("doctor", help="Validate profile and input")
    doctor_profile_group = doctor_parser.add_mutually_exclusive_group(required=False)
    doctor_profile_group.add_argument("--profile", help="Path to YAML profile")
    doctor_profile_group.add_argument(
        "--preset",
        choices=sorted(PRESET_CHOICES),
        help="Built-in profile preset",
    )
    doctor_parser.add_argument(
        "--input", required=True, help="Path to local input file"
    )
    doctor_parser.add_argument(
        "--artifacts-root",
        default=None,
        help="Override artifacts root directory",
    )
    doctor_parser.add_argument(
        "--text-model",
        default=None,
        help="Override normalize_text model for this check",
    )
    doctor_parser.add_argument(
        "--ocr-model",
        default=None,
        help="Override ocr_vision model for this check",
    )

    eval_parser = subparsers.add_parser("eval", help="Evaluation commands")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command")
    eval_subparsers.add_parser("quick", help="Run quick fixture evaluation flow")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == "run":
            profile = _load_profile_for_command(
                profile=args.profile,
                preset=args.preset,
                artifacts_root=args.artifacts_root,
                text_model=args.text_model,
                ocr_model=args.ocr_model,
                enforce_model_backend=True,
            )
            runner = PipelineRunner(profile)
            state = runner.run(
                input_path=args.input,
                run_id=args.run_id,
                resume=args.resume,
            )
            print(json.dumps(state, indent=2, sort_keys=True))
            _print_map_telemetry_summary(
                profile_root=profile.artifacts.root, state=state
            )
            return 0

        if args.command == "status":
            profile = _load_profile_for_command(
                profile=args.profile,
                preset=args.preset,
                artifacts_root=args.artifacts_root,
                text_model=None,
                ocr_model=None,
                enforce_model_backend=False,
            )
            runner = PipelineRunner(profile)
            state = runner.status(run_id=args.run_id)
            print(json.dumps(state, indent=2, sort_keys=True))
            return 0

        if args.command == "doctor":
            profile = _load_profile_for_command(
                profile=args.profile,
                preset=args.preset,
                artifacts_root=args.artifacts_root,
                text_model=args.text_model,
                ocr_model=args.ocr_model,
                enforce_model_backend=True,
            )
            runner = PipelineRunner(profile)
            report = runner.doctor(input_path=args.input)
            print(json.dumps(report, indent=2, sort_keys=True))
            if report.get("ok"):
                return 0
            for error in report.get("errors", []):
                print(str(error), file=sys.stderr)
            return 2

        if args.command == "eval" and args.eval_command == "quick":
            print("quick eval scaffold ready")
            return 0

        parser.print_help()
        return 1

    except (ProfileError, PipelineError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _print_map_telemetry_summary(*, profile_root: Path, state: dict[str, Any]) -> None:
    run_id = str(state.get("run_id", "")).strip()
    if not run_id:
        return

    manifest_path = (
        profile_root.expanduser().resolve() / run_id / "map" / "manifest.json"
    )
    if not manifest_path.exists() or not manifest_path.is_file():
        return

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return

    processed = manifest.get("processed_telemetry")
    if not isinstance(processed, dict):
        return

    chunk_count = _as_int(manifest.get("chunk_count"), 0)
    processed_chunks = _as_int(manifest.get("processed"), 0)
    requests = _as_int(processed.get("requests"), 0)
    latency_s = _as_float(processed.get("latency_s"), 0.0)
    output_tokens_est = _as_int(processed.get("output_tokens_est"), 0)
    usage_chunks = _as_int(processed.get("chunks_with_usage"), 0)
    tok_s = processed.get("effective_tokens_per_second")

    prompt_tokens = _as_optional_int(processed.get("prompt_tokens"))
    completion_tokens = _as_optional_int(processed.get("completion_tokens"))
    total_tokens = _as_optional_int(processed.get("total_tokens"))

    token_source = (
        "usage.completion_tokens"
        if completion_tokens is not None
        else "output_tokens_est"
    )

    print(
        (
            "map telemetry: "
            f"processed={processed_chunks}/{chunk_count} "
            f"requests={requests} "
            f"latency_s={latency_s:.3f} "
            f"tok_s={_format_metric(tok_s)} "
            f"source={token_source} "
            f"usage_chunks={usage_chunks} "
            f"prompt={_format_metric(prompt_tokens)} "
            f"completion={_format_metric(completion_tokens)} "
            f"total={_format_metric(total_tokens)} "
            f"output_est={output_tokens_est}"
        ),
        file=sys.stderr,
    )

    warning = _reasoning_efficiency_warning(
        completion_tokens=completion_tokens,
        output_tokens_est=output_tokens_est,
    )
    if warning:
        print(warning, file=sys.stderr)

    print(f"map manifest: {manifest_path}", file=sys.stderr)


def _load_profile_for_command(
    *,
    profile: str | None,
    preset: str | None,
    artifacts_root: str | None,
    text_model: str | None,
    ocr_model: str | None,
    enforce_model_backend: bool,
) -> PipelineProfile:
    if profile:
        loaded = load_profile(profile)
    else:
        loaded = _load_preset_profile(
            preset=(preset or "auto"),
            enforce_model_backend=enforce_model_backend,
        )

    return _apply_profile_overrides(
        loaded,
        artifacts_root=artifacts_root,
        text_model=text_model,
        ocr_model=ocr_model,
    )


def _load_preset_profile(
    *, preset: str, enforce_model_backend: bool
) -> PipelineProfile:
    if preset == "passthrough":
        return load_profile(
            _resolve_repo_profile_path("profiles/pipeline.profile.example.yaml")
        )

    if preset == "auto":
        selected = _auto_select_provider_preset()
        if selected is None:
            if enforce_model_backend:
                raise ProfileError(_missing_provider_error_message())
            return load_profile(
                _resolve_repo_profile_path("profiles/pipeline.profile.example.yaml")
            )
        return _build_remote_preset_profile(selected, source_label="auto")

    if preset in PRESET_PROVIDER_CONFIG:
        return _build_remote_preset_profile(preset, source_label=preset)

    raise ProfileError(f"Unknown preset: {preset}")


def _auto_select_provider_preset() -> str | None:
    forced_provider = os.getenv("SCRIBA_PROVIDER", "").strip().lower()
    if forced_provider:
        if forced_provider not in PRESET_PROVIDER_CONFIG:
            allowed = ", ".join(sorted(PRESET_PROVIDER_CONFIG.keys()))
            raise ProfileError(
                f"SCRIBA_PROVIDER must be one of: {allowed}. Got: {forced_provider}"
            )
        return forced_provider

    for candidate in AUTO_PRESET_PRIORITY:
        config = PRESET_PROVIDER_CONFIG[candidate]
        env_key = config["env_key"]
        if os.getenv(env_key, "").strip():
            return candidate
    return None


def _missing_provider_error_message() -> str:
    required_env = ", ".join(
        config["env_key"] for config in PRESET_PROVIDER_CONFIG.values()
    )
    return (
        "No provider API key detected for automatic preset selection. "
        "Set one of the following environment variables or use --profile/--preset passthrough: "
        f"{required_env}"
    )


def _build_remote_preset_profile(preset: str, *, source_label: str) -> PipelineProfile:
    config = PRESET_PROVIDER_CONFIG.get(preset)
    if config is None:
        raise ProfileError(f"Unknown preset: {preset}")

    env_key = config["env_key"]
    api_key = os.getenv(env_key, "").strip()
    if not api_key:
        raise ProfileError(
            f"Preset '{preset}' requires environment variable '{env_key}' to be set."
        )

    provider = config["provider"]
    model = config["default_model"]
    return PipelineProfile(
        version=1,
        artifacts=ArtifactsConfig(root=Path("./artifacts"), run_id="auto"),
        roles={
            "normalize_text": RoleBinding(
                backend="remote_text",
                model=model,
            )
        },
        backends={
            "remote_text": BackendConfig(
                adapter="litellm",
                topology="remote",
                provider=provider,
                model_origin="hosted_weights",
                base_url="",
                inference_path="/chat/completions",
                health_path="/models",
                startup_timeout_s=30,
                api_key=api_key,
            )
        },
        stages=_default_stages_for_preset(),
        source_path=Path(f"<preset:{source_label}:{provider}>"),
    )


def _default_stages_for_preset() -> dict[str, StageConfig]:
    stages = {stage_name: StageConfig() for stage_name in DEFAULT_STAGE_ORDER}
    stages["normalize_map"] = StageConfig(
        enabled=True,
        temperature=0.0,
        request_timeout_s=900,
        max_output_tokens=1024,
        reasoning_effort="none",
        reasoning_exclude=True,
    )
    stages["validate"] = StageConfig(enabled=True, fail_on_hard_errors=True)
    stages["export"] = StageConfig(enabled=True, multi_file=True)
    return stages


def _resolve_repo_profile_path(rel: str) -> str:
    candidate_cwd = Path(rel)
    if candidate_cwd.exists() and candidate_cwd.is_file():
        return str(candidate_cwd)

    candidate_repo = _repo_root() / rel
    if candidate_repo.exists() and candidate_repo.is_file():
        return str(candidate_repo)

    raise ProfileError(f"Preset profile not found: expected {rel}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _apply_profile_overrides(
    profile: PipelineProfile,
    *,
    artifacts_root: str | None,
    text_model: str | None,
    ocr_model: str | None,
) -> PipelineProfile:
    updated = profile

    if artifacts_root:
        updated = replace(
            updated,
            artifacts=replace(
                updated.artifacts,
                root=Path(artifacts_root).expanduser(),
            ),
        )

    roles = dict(updated.roles)
    roles_changed = False

    if text_model is not None:
        normalize = roles.get("normalize_text")
        if normalize is None:
            raise ProfileError(
                "Cannot apply --text-model because profile has no normalize_text role."
            )
        roles["normalize_text"] = replace(normalize, model=text_model)
        roles_changed = True

    if ocr_model is not None:
        ocr = roles.get("ocr_vision")
        if ocr is None:
            normalize = roles.get("normalize_text")
            if normalize is None:
                raise ProfileError(
                    "Cannot apply --ocr-model because profile has no ocr_vision or normalize_text role."
                )
            roles["ocr_vision"] = RoleBinding(
                backend=normalize.backend,
                model=ocr_model,
            )
        else:
            roles["ocr_vision"] = replace(ocr, model=ocr_model)
        roles_changed = True

    if roles_changed:
        updated = replace(updated, roles=roles)

    return updated


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_metric(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _reasoning_efficiency_warning(
    *,
    completion_tokens: int | None,
    output_tokens_est: int,
) -> str | None:
    if completion_tokens is None:
        return None
    if output_tokens_est <= 0:
        return None

    ratio = completion_tokens / float(output_tokens_est)
    if ratio < 10.0:
        return None

    return (
        "map warning: completion/output ratio is high "
        f"({ratio:.1f}x); consider lowering `max_output_tokens` or disabling reasoning/thinking mode on the backend"
    )


if __name__ == "__main__":
    raise SystemExit(main())
