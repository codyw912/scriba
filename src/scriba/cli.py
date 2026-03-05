"""CLI entrypoint for scriba."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from scriba.pipeline import PipelineError, PipelineRunner, ProfileError, load_profile


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scriba",
        description="CLI-first document normalization pipeline",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a pipeline profile on an input")
    run_parser.add_argument("--profile", required=True, help="Path to YAML profile")
    run_parser.add_argument("--input", required=True, help="Path to local input file")
    run_parser.add_argument("--run-id", default=None, help="Optional explicit run id")
    run_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume existing run state when available",
    )

    status_parser = subparsers.add_parser("status", help="Read state for a run id")
    status_parser.add_argument("--profile", required=True, help="Path to YAML profile")
    status_parser.add_argument("--run-id", required=True, help="Run id to inspect")

    doctor_parser = subparsers.add_parser("doctor", help="Validate profile and input")
    doctor_parser.add_argument("--profile", required=True, help="Path to YAML profile")
    doctor_parser.add_argument(
        "--input", required=True, help="Path to local input file"
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
            profile = load_profile(args.profile)
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
            profile = load_profile(args.profile)
            runner = PipelineRunner(profile)
            state = runner.status(run_id=args.run_id)
            print(json.dumps(state, indent=2, sort_keys=True))
            return 0

        if args.command == "doctor":
            profile = load_profile(args.profile)
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
